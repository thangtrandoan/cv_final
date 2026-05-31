from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.detector import TinyGridDetector
from utils.anchors import DEFAULT_ANCHORS, kmeans_anchors
from utils.dataset import ObjectDetectionDataset, collate_fn
from utils.loss import DetectionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyGridDetector.")
    parser.add_argument("--train_data", required=True, type=Path)
    parser.add_argument("--val_data", required=True, type=Path)
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--val_image_dir", required=True, type=Path)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./models/"))
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_kmeans_anchors", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    class_names: list[str],
    anchors: torch.Tensor,
    img_size: int,
    grid_size: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": model_to_save.state_dict(),
            "class_names": class_names,
            "anchors": anchors.cpu().tolist(),
            "img_size": img_size,
            "grid_size": grid_size,
            "best_metric": best_metric,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

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

    anchors = kmeans_anchors(args.train_data, k=3) if args.use_kmeans_anchors else DEFAULT_ANCHORS.clone()
    model = TinyGridDetector(num_classes=len(train_dataset.class_names), num_anchors=anchors.shape[0]).to(device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)
    class_weights = class_weights_from_dataset(train_dataset).to(device)
    criterion = DetectionLoss(
        anchors=anchors.to(device),
        img_size=args.img_size,
        grid_size=args.img_size // 32,
        num_classes=len(train_dataset.class_names),
        class_weights=class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_loss = float("inf")
    best_path = args.checkpoint_dir / "best.pth"
    print(
        "Starting training "
        f"device={device} "
        f"epochs={args.epochs} "
        f"batch_size={args.batch_size} "
        f"gpus={gpu_count} "
        f"train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} "
        f"checkpoint={best_path}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                model,
                train_dataset.class_names,
                anchors,
                args.img_size,
                args.img_size // 32,
                best_metric=-best_val_loss,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"best_val_loss={best_val_loss:.4f}"
        )

    print(f"Best checkpoint saved to {best_path}")


if __name__ == "__main__":
    main()
