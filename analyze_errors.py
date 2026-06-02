from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze object detection errors.")
    parser.add_argument("--ground_truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("error_analysis.json"))
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--overlap_threshold", type=float, default=0.1)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def object_size_bucket(box: list[float], image: dict[str, Any]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    image_area = float(image["width"] * image["height"])
    ratio = area / max(image_area, 1.0)
    if ratio < 0.02:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


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


def analyze(ground_truth: dict[str, Any], predictions: list[dict[str, Any]], iou_threshold: float, overlap_threshold: float) -> dict[str, Any]:
    classes = ground_truth["classes"]
    image_info = {item["id"]: item for item in ground_truth["images"]}
    gt_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in ground_truth["annotations"]:
        image = image_info[ann["image_id"]]
        gt_by_image[ann["image_id"]].append(
            {
                "class": ann["class"],
                "bbox": [float(value) for value in ann["bbox"]],
                "matched": False,
                "size": object_size_bucket([float(value) for value in ann["bbox"]], image),
            }
        )

    pred_by_image = {entry["image_id"]: entry.get("boxes", []) for entry in predictions}

    per_class: dict[str, dict[str, Any]] = {
        class_name: {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp_background": 0,
            "fp_duplicate": 0,
            "fp_wrong_class": 0,
            "fp_localization": 0,
            "fn": 0,
            "ious_tp": [],
            "conf_tp": [],
            "conf_fp": [],
            "size_gt": Counter(),
            "size_tp": Counter(),
            "size_fn": Counter(),
        }
        for class_name in classes
    }
    confusion = Counter()
    worst_images = Counter()
    all_class_predictions: dict[str, list[dict[str, Any]]] = {class_name: [] for class_name in classes}

    for image_id, gts in gt_by_image.items():
        for gt in gts:
            per_class[gt["class"]]["gt"] += 1
            per_class[gt["class"]]["size_gt"][gt["size"]] += 1

        preds = sorted(pred_by_image.get(image_id, []), key=lambda item: item["confidence"], reverse=True)
        for pred in preds:
            pred_class = pred["class"]
            pred_box = [float(value) for value in pred["bbox"]]
            confidence = float(pred["confidence"])
            per_class[pred_class]["pred"] += 1

            same_class_candidates = [gt for gt in gts if gt["class"] == pred_class]
            best_same_iou = 0.0
            best_same = None
            for gt in same_class_candidates:
                iou = bbox_iou(pred_box, gt["bbox"])
                if iou > best_same_iou:
                    best_same_iou = iou
                    best_same = gt

            all_candidates = gts
            best_any_iou = 0.0
            best_any = None
            for gt in all_candidates:
                iou = bbox_iou(pred_box, gt["bbox"])
                if iou > best_any_iou:
                    best_any_iou = iou
                    best_any = gt

            is_tp = best_same is not None and not best_same["matched"] and best_same_iou >= iou_threshold
            if is_tp:
                best_same["matched"] = True
                per_class[pred_class]["tp"] += 1
                per_class[pred_class]["ious_tp"].append(best_same_iou)
                per_class[pred_class]["conf_tp"].append(confidence)
                per_class[pred_class]["size_tp"][best_same["size"]] += 1
                all_class_predictions[pred_class].append({"confidence": confidence, "tp": 1, "image_id": image_id})
                continue

            per_class[pred_class]["conf_fp"].append(confidence)
            all_class_predictions[pred_class].append({"confidence": confidence, "tp": 0, "image_id": image_id})
            worst_images[image_id] += 1

            if best_same is not None and best_same_iou >= iou_threshold and best_same["matched"]:
                per_class[pred_class]["fp_duplicate"] += 1
            elif best_same is not None and best_same_iou >= overlap_threshold:
                per_class[pred_class]["fp_localization"] += 1
            elif best_any is not None and best_any_iou >= overlap_threshold:
                per_class[pred_class]["fp_wrong_class"] += 1
                confusion[(best_any["class"], pred_class)] += 1
            else:
                per_class[pred_class]["fp_background"] += 1

        for gt in gts:
            if not gt["matched"]:
                per_class[gt["class"]]["fn"] += 1
                per_class[gt["class"]]["size_fn"][gt["size"]] += 1
                worst_images[image_id] += 1

    result_classes = {}
    for class_name in classes:
        stats = per_class[class_name]
        tp = stats["tp"]
        fp = stats["pred"] - tp
        gt_count = stats["gt"]
        preds = sorted(all_class_predictions[class_name], key=lambda item: item["confidence"], reverse=True)
        tp_sum = 0
        fp_sum = 0
        recalls = []
        precisions = []
        for pred in preds:
            if pred["tp"]:
                tp_sum += 1
            else:
                fp_sum += 1
            recalls.append(tp_sum / gt_count if gt_count else 0.0)
            precisions.append(tp_sum / max(tp_sum + fp_sum, 1))

        result_classes[class_name] = {
            "ap": round(compute_ap(recalls, precisions), 6),
            "gt": gt_count,
            "pred": stats["pred"],
            "tp": tp,
            "fp": fp,
            "fn": stats["fn"],
            "precision": round(tp / max(stats["pred"], 1), 6),
            "recall": round(tp / max(gt_count, 1), 6),
            "mean_tp_iou": round(sum(stats["ious_tp"]) / max(len(stats["ious_tp"]), 1), 6),
            "mean_tp_conf": round(sum(stats["conf_tp"]) / max(len(stats["conf_tp"]), 1), 6),
            "mean_fp_conf": round(sum(stats["conf_fp"]) / max(len(stats["conf_fp"]), 1), 6),
            "fp_breakdown": {
                "background": stats["fp_background"],
                "duplicate": stats["fp_duplicate"],
                "wrong_class": stats["fp_wrong_class"],
                "localization": stats["fp_localization"],
            },
            "size_gt": dict(stats["size_gt"]),
            "size_tp": dict(stats["size_tp"]),
            "size_fn": dict(stats["size_fn"]),
        }

    return {
        "classes": result_classes,
        "confusion_top": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in confusion.most_common(20)
        ],
        "worst_images": [
            {"image_id": image_id, "errors": count}
            for image_id, count in worst_images.most_common(30)
        ],
    }


def main() -> None:
    args = parse_args()
    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)
    result = analyze(ground_truth, predictions, args.iou_threshold, args.overlap_threshold)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
