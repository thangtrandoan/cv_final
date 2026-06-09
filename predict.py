from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageEnhance

from models.detector import TinyGridDetector
from utils.json_utils import write_json
from utils.loss import LEVEL_SPECS
from utils.nms import nms


TTA_BRIGHTNESS_FACTORS = [0.85, 1.15]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
LETTERBOX_FILL = tuple(int(round(value * 255)) for value in (0.485, 0.456, 0.406))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyGridDetector inference.")
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("./models/best.pth"))
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--conf_threshold", type=float, default=0.45)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--max_detections_per_image", type=int, default=20)
    parser.add_argument("--pre_nms_topk", type=int, default=1000)
    parser.add_argument("--preprocess", choices=("auto", "letterbox", "stretch"), default="auto")
    parser.add_argument("--disable_tta", action="store_true")
    parser.add_argument("--tta_brightness", nargs="*", type=float, default=TTA_BRIGHTNESS_FACTORS)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--no_channels_last", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def letterbox_image(image: Image.Image, img_size: int) -> tuple[Image.Image, float, int, int]:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(img_size / width, img_size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = image.resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), LETTERBOX_FILL)
    pad_x = (img_size - resized_w) // 2
    pad_y = (img_size - resized_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def image_to_tensor(image: Image.Image, img_size: int, preprocess: str) -> tuple[torch.Tensor, float, int, int]:
    if preprocess == "letterbox":
        image, scale, pad_x, pad_y = letterbox_image(image, img_size)
    else:
        original_w, original_h = image.size
        image = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        scale = img_size / max(original_w, 1)
        pad_x = 0
        pad_y = 0
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(img_size, img_size, 3)
    tensor = data.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD, scale, pad_x, pad_y


@torch.no_grad()
def predict_image(
    model: TinyGridDetector,
    image: Image.Image,
    image_id: str,
    class_names: list[str],
    img_size: int,
    conf_threshold: float,
    nms_threshold: float,
    pre_nms_topk: int,
    device: torch.device,
    preprocess: str,
    channels_last: bool,
) -> dict[str, object]:
    original_w, original_h = image.size
    tensor, scale, pad_x, pad_y = image_to_tensor(image, img_size, preprocess)
    tensor = tensor.to(device)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    outputs = model(tensor)

    output_boxes: list[dict[str, object]] = []
    for level, (cls_logits, reg_preds, cnt_logits) in outputs.items():
        stride = LEVEL_SPECS[level]["stride"]
        cls_scores = torch.sigmoid(cls_logits[0])
        cnt_scores = torch.sigmoid(cnt_logits[0])
        scores_per_class = torch.sqrt((cls_scores * cnt_scores).clamp(min=0.0))
        scores, class_ids = scores_per_class.reshape(len(class_names), -1).max(dim=0)
        keep = scores >= conf_threshold
        if not keep.any():
            continue

        _, height, width = cls_logits.shape[1:]
        shifts_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        distances = reg_preds[0].permute(1, 2, 0).reshape(-1, 4) * stride
        boxes = torch.stack(
            (
                points[:, 0] - distances[:, 0],
                points[:, 1] - distances[:, 1],
                points[:, 0] + distances[:, 2],
                points[:, 1] + distances[:, 3],
            ),
            dim=1,
        )
        if preprocess == "letterbox":
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        else:
            resize_scale = torch.tensor(
                [original_w / img_size, original_h / img_size, original_w / img_size, original_h / img_size],
                device=device,
                dtype=torch.float32,
            )
            boxes = boxes * resize_scale
        boxes = boxes.clamp(min=0)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=original_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=original_h)

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        if pre_nms_topk > 0 and scores.numel() > pre_nms_topk:
            scores, topk_indices = scores.topk(pre_nms_topk)
            boxes = boxes[topk_indices]
            class_ids = class_ids[topk_indices]
        for class_id in class_ids.unique():
            class_mask = class_ids == class_id
            selected = nms(boxes[class_mask], scores[class_mask], nms_threshold)
            class_boxes = boxes[class_mask][selected]
            class_scores_selected = scores[class_mask][selected]
            for box, score in zip(class_boxes, class_scores_selected):
                xmin, ymin, xmax, ymax = box.tolist()
                if xmax <= xmin or ymax <= ymin:
                    continue
                output_boxes.append(
                    {
                        "class": class_names[int(class_id.item())],
                        "confidence": round(float(score.item()), 6),
                        "bbox": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                    }
                )

    output_boxes.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": output_boxes}


def merge_boxes(
    image_id: str,
    boxes: list[dict[str, object]],
    class_names: list[str],
    nms_threshold: float,
    max_detections_per_image: int,
    device: torch.device,
) -> dict[str, object]:
    if not boxes:
        return {"image_id": image_id, "boxes": []}

    merged: list[dict[str, object]] = []
    for class_name in class_names:
        class_boxes = [box for box in boxes if box["class"] == class_name]
        if not class_boxes:
            continue
        box_tensor = torch.tensor([box["bbox"] for box in class_boxes], dtype=torch.float32, device=device)
        score_tensor = torch.tensor([box["confidence"] for box in class_boxes], dtype=torch.float32, device=device)
        selected = nms(box_tensor, score_tensor, nms_threshold)
        for index in selected.tolist():
            merged.append(class_boxes[index])

    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return {"image_id": image_id, "boxes": merged[:max_detections_per_image]}


def predict_with_tta(
    model: TinyGridDetector,
    image_path: Path,
    class_names: list[str],
    img_size: int,
    conf_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    pre_nms_topk: int,
    brightness_factors: list[float],
    device: torch.device,
    preprocess: str,
    use_tta: bool,
    channels_last: bool,
) -> dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    original_w, _ = image.size
    all_boxes: list[dict[str, object]] = []

    variants: list[tuple[Image.Image, bool]] = [(image, False)]
    if use_tta:
        variants.append((image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), True))
        for factor in brightness_factors:
            variants.append((ImageEnhance.Brightness(image).enhance(factor), False))

    for variant, flipped in variants:
        prediction = predict_image(
            model,
            variant,
            image_path.name,
            class_names,
            img_size,
            conf_threshold,
            nms_threshold,
            pre_nms_topk,
            device,
            preprocess,
            channels_last,
        )
        for box in prediction["boxes"]:
            box = dict(box)
            if flipped:
                xmin, ymin, xmax, ymax = box["bbox"]
                box["bbox"] = [round(original_w - xmax, 2), ymin, round(original_w - xmin, 2), ymax]
            all_boxes.append(box)

    return merge_boxes(
        image_path.name,
        all_boxes,
        class_names,
        nms_threshold,
        max_detections_per_image,
        device,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = int(checkpoint.get("img_size", args.img_size))
    preprocess = args.preprocess
    if preprocess == "auto":
        preprocess = str(checkpoint.get("preprocess", "stretch"))
    use_tta = not args.disable_tta
    channels_last = device.type == "cuda" and not args.no_channels_last
    print(
        "Starting prediction "
        f"device={device} "
        f"image_dir={args.image_dir} "
        f"checkpoint={args.checkpoint} "
        f"preprocess={preprocess} "
        f"conf_threshold={args.conf_threshold} "
        f"max_detections_per_image={args.max_detections_per_image} "
        f"pre_nms_topk={args.pre_nms_topk} "
        f"tta={use_tta} "
        f"tta_brightness={args.tta_brightness} "
        f"channels_last={channels_last} "
        f"output={args.output}",
        flush=True,
    )

    model = TinyGridDetector(
        num_classes=len(class_names),
        pretrained_backbone=False,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_paths = sorted(
        [
            path
            for path in args.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    predictions = []
    for index, image_path in enumerate(image_paths, start=1):
        predictions.append(
            predict_with_tta(
                model,
                image_path,
                class_names,
                img_size,
                args.conf_threshold,
                args.nms_threshold,
                args.max_detections_per_image,
                args.pre_nms_topk,
                args.tta_brightness,
                device,
                preprocess,
                use_tta,
                channels_last,
            )
        )
        if args.progress_every > 0 and (index == len(image_paths) or index % args.progress_every == 0):
            print(f"predicted={index}/{len(image_paths)}", flush=True)
    write_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
