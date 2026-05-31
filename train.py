from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.detector import TinyGridDetector
from utils.anchors import kmeans_anchors
from utils.box_ops import cxcywh_to_xyxy
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.loss import DetectionLoss, decode_raw_predictions
from utils.nms import nms


MULTI_SCALE_MIN = 320
MULTI_SCALE_MAX = 640
EARLY_STOPPING_PATIENCE = 50
MAP_CONF_THRESHOLD = 0.05
MAP_NMS_THRESHOLD = 0.50
MAP_MAX_DETECTIONS_PER_IMAGE = 100


def format_duration(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyGridDetector.")
    parser.add_argument("--train_data", required=True, type=Path)
    parser.add_argument("--val_data", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--val_image_dir", required=True, type=Path)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models/"))
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_noobj", type=float, default=2.0)
    parser.add_argument("--multi_scale_min", type=int, default=MULTI_SCALE_MIN)
    parser.add_argument("--multi_scale_max", type=int, default=MULTI_SCALE_MAX)
    parser.add_argument("--early_stopping_patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--map_conf_threshold", type=float, default=MAP_CONF_THRESHOLD)
    parser.add_argument("--map_nms_threshold", type=float, default=MAP_NMS_THRESHOLD)
    parser.add_argument(
        "--map_max_detections_per_image",
        type=int,
        default=MAP_MAX_DETECTIONS_PER_IMAGE,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_from_best", action="store_true")
    parser.add_argument("--resume_checkpoint", type=Path)
    return parser.parse_args()


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def bbox_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_weights_from_dataset(dataset: ObjectDetectionDataset) -> torch.Tensor:
    counts = torch.ones(len(dataset.class_names), dtype=torch.float32)
    for targets in dataset.targets_by_image.values():
        for item in targets:
            counts[item["class_id"]] += 1
    weights = counts.sum() / (counts * len(counts))
    return weights / weights.mean()


def run_epoch(
    model: TinyGridDetector,
    dataloader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        raw = model(images)
        loss, metrics = criterion(raw, targets)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1

    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    anchors: torch.Tensor,
    num_classes: int,
    img_size: int,
    device: torch.device,
    conf_threshold: float,
    nms_threshold: float,
    max_detections_per_image: int,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    anchors = anchors.to(device)
    gt_by_class: dict[int, dict[int, list[dict[str, object]]]] = {
        class_id: {} for class_id in range(num_classes)
    }
    pred_by_class: dict[int, list[dict[str, object]]] = {class_id: [] for class_id in range(num_classes)}
    image_index = 0

    for images, targets in dataloader:
        images = images.to(device)
        raw = model(images)
        decoded = decode_raw_predictions(raw, anchors)

        for batch_idx, image_targets in enumerate(targets):
            current_image_index = image_index + batch_idx
            for item in image_targets:
                xmin, ymin, xmax, ymax = [float(value) / img_size for value in item["bbox"]]
                class_id = int(item["class_id"])
                gt_by_class[class_id].setdefault(current_image_index, []).append(
                    {"bbox": torch.tensor([xmin, ymin, xmax, ymax]), "matched": False}
                )

            image_pred = decoded[batch_idx]
            boxes_cxcywh = image_pred[..., 0:4].reshape(-1, 4)
            object_scores = torch.sigmoid(image_pred[..., 4].reshape(-1))
            class_probs = F.softmax(image_pred[..., 5:].reshape(-1, num_classes), dim=1)
            class_scores, class_ids = class_probs.max(dim=1)
            scores = object_scores * class_scores

            keep = scores >= conf_threshold
            boxes_cxcywh = boxes_cxcywh[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
            if boxes_cxcywh.numel() == 0:
                continue

            boxes = cxcywh_to_xyxy(boxes_cxcywh).clamp(0.0, 1.0)
            image_predictions: list[dict[str, object]] = []
            for class_id in class_ids.unique():
                class_mask = class_ids == class_id
                selected = nms(boxes[class_mask], scores[class_mask], nms_threshold)
                for box, score in zip(boxes[class_mask][selected], scores[class_mask][selected]):
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    image_predictions.append(
                        {
                            "image_id": current_image_index,
                            "class_id": int(class_id.item()),
                            "confidence": float(score.item()),
                            "bbox": box.detach().cpu(),
                        }
                    )

            image_predictions.sort(key=lambda item: float(item["confidence"]), reverse=True)
            for pred in image_predictions[:max_detections_per_image]:
                pred_by_class[int(pred["class_id"])].append(pred)

        image_index += images.shape[0]

    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    for class_id in range(num_classes):
        class_gt = gt_by_class[class_id]
        num_gt = sum(len(items) for items in class_gt.values())
        class_preds = sorted(
            pred_by_class[class_id], key=lambda item: float(item["confidence"]), reverse=True
        )
        tp_flags = []
        fp_flags = []

        for pred in class_preds:
            candidates = class_gt.get(int(pred["image_id"]), [])
            best_iou = 0.0
            best_index = -1
            for index, gt in enumerate(candidates):
                if bool(gt["matched"]):
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0 and best_iou >= iou_threshold:
                candidates[best_index]["matched"] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        cumulative_tp = []
        cumulative_fp = []
        tp_sum = 0
        fp_sum = 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp
            fp_sum += fp
            cumulative_tp.append(tp_sum)
            cumulative_fp.append(fp_sum)

        recalls = [value / num_gt if num_gt else 0.0 for value in cumulative_tp]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(cumulative_tp, cumulative_fp)]
        if num_gt:
            aps.append(compute_ap(recalls, precisions))

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += num_gt

    map_50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "map_50": map_50,
        "micro_precision": total_tp / max(total_tp + total_fp, 1),
        "micro_recall": total_tp / total_gt if total_gt else 0.0,
        "num_predictions": float(sum(len(items) for items in pred_by_class.values())),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    class_names: list[str],
    anchors: torch.Tensor,
    img_size: int,
    grid_size: int,
    epoch: int,
    best_val_loss: float,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "class_names": class_names,
            "anchors": anchors.cpu().tolist(),
            "img_size": img_size,
            "grid_size": grid_size,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_metric": best_metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    best_path = args.checkpoint_dir / "best.pth"
    resume_path = args.resume_checkpoint
    if args.resume_from_best:
        resume_path = best_path
    if args.resume_from_best and args.resume_checkpoint:
        raise ValueError("Use either --resume_from_best or --resume_checkpoint, not both.")
    checkpoint = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

    train_dataset = ObjectDetectionDataset(args.train_data, args.image_dir, img_size=args.img_size, train=True)
    val_dataset = ObjectDetectionDataset(args.val_data, args.val_image_dir, img_size=args.img_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    anchors = kmeans_anchors(args.train_data, k=3)
    model = TinyGridDetector(
        num_classes=len(train_dataset.class_names),
        num_anchors=anchors.shape[0],
        pretrained_backbone=True,
    ).to(device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)
    class_weights = class_weights_from_dataset(train_dataset).to(device)
    criterion = DetectionLoss(
        anchors=anchors.to(device),
        img_size=args.img_size,
        grid_size=args.img_size // 32,
        num_classes=len(train_dataset.class_names),
        class_weights=class_weights,
        lambda_noobj=args.lambda_noobj,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 1
    best_val_loss = float("inf")
    best_map = float("-inf")
    epochs_without_improvement = 0
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_map = float(checkpoint.get("best_metric", float("-inf")))
        print(
            "Resuming training "
            f"checkpoint={resume_path} "
            f"start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.4f} "
            f"best_mAP@0.5={best_map:.4f}",
            flush=True,
        )
    multi_scale_sizes = list(range(args.multi_scale_min, args.multi_scale_max + 1, 32))
    if not multi_scale_sizes:
        raise ValueError("Multi-scale range must include at least one size.")
    print(
        "Starting training "
        f"device={device} "
        f"epochs={args.epochs} "
        f"batch_size={args.batch_size} "
        f"gpus={gpu_count} "
        f"architecture=resnet50 "
        f"pretrained_backbone=True "
        f"multi_scale=True "
        f"eval_map=True "
        f"early_stopping_patience={args.early_stopping_patience} "
        f"train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} "
        f"checkpoint={best_path}",
        flush=True,
    )
    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.perf_counter()
        current_img_size = random.choice(multi_scale_sizes)
        train_dataset.transform.img_size = current_img_size
        criterion.img_size = current_img_size
        criterion.grid_size = current_img_size // 32
        print(f"epoch={epoch:03d} multi_scale_img_size={current_img_size}", flush=True)

        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_dataset.transform.img_size = args.img_size
        criterion.img_size = args.img_size
        criterion.grid_size = args.img_size // 32
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        map_metrics = evaluate_map(
            model=model,
            dataloader=val_loader,
            anchors=anchors,
            num_classes=len(train_dataset.class_names),
            img_size=args.img_size,
            device=device,
            conf_threshold=args.map_conf_threshold,
            nms_threshold=args.map_nms_threshold,
            max_detections_per_image=args.map_max_detections_per_image,
        )

        improved = map_metrics["map_50"] > best_map
        if improved:
            best_val_loss = val_loss
            best_map = map_metrics["map_50"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                train_dataset.class_names,
                anchors,
                args.img_size,
                args.img_size // 32,
                epoch,
                best_val_loss,
                best_metric=best_map,
            )
        else:
            epochs_without_improvement += 1

        message = (
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"best_val_loss={best_val_loss:.4f} "
            f"val_obj_conf={val_metrics['obj_conf']:.4f} "
            f"val_noobj_conf={val_metrics['noobj_conf']:.4f}"
        )
        message += (
            f" val_mAP@0.5={map_metrics['map_50']:.4f} "
            f"best_mAP@0.5={best_map:.4f} "
            f"val_precision={map_metrics['micro_precision']:.4f} "
            f"val_recall={map_metrics['micro_recall']:.4f} "
            f"val_predictions={int(map_metrics['num_predictions'])} "
            f"patience={epochs_without_improvement}/{args.early_stopping_patience} "
            f"epoch_time={format_duration(time.perf_counter() - epoch_start_time)}"
        )
        print(message)

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch={epoch:03d} "
                f"best_mAP@0.5={best_map:.4f} "
                f"best_checkpoint={best_path}",
                flush=True,
            )
            break

    print(f"Best checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
