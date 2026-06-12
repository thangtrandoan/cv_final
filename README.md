# FCOS Object Detection

Project nay cai dat mot one-stage object detector tu dau cho 5 lop:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

Code khong dung YOLOv5/v8, Detectron2, MMDetection, Faster R-CNN hay SSD co san. Project dung PyTorch va backbone ResNet50 pretrained lam bo trich xuat dac trung.

## Cai Dat

```bash
pip install -r requirements.txt
```

## Cau Truc

```text
models/
  detector.py
utils/
  box_ops.py
  dataset.py
  json_utils.py
  loss.py
  nms.py
  transforms.py
train.py
predict.py
requirements.txt
```

## Du Lieu

Du lieu theo cau truc:

```text
public/
  classes.json
  train/images/
  val/images/
  annotations/train.json
  annotations/val.json
  tools/evaluate_predictions.py
```

Annotation dung bbox dang:

```text
[xmin, ymin, xmax, ymax]
```

Toa do bbox la toa do tren anh goc.

## Mo Hinh

`TinyGridDetector` la detector anchor-free theo huong FCOS:

- Backbone: ResNet50 pretrained.
- Neck: FPN/BiFPN nhe.
- Head: classification tower, box regression tower, centerness head.
- Output moi level gom:
  - class logits
  - box distances `[left, top, right, bottom]`
  - centerness/objectness

Checkpoint moi co the luu them cac tuy chon kien truc:

- `use_p2`
- `use_p6`
- `use_scales`
- `preprocess`
- `model_type`

`predict.py` tu doc metadata trong checkpoint de khoi tao dung kien truc.

## Tien Xu Ly Va Augment

Pipeline du lieu co:

- Doc JSON annotation va nhieu object trong mot anh.
- Letterbox resize de giu ti le anh.
- Normalize theo ImageNet mean/std.
- Horizontal flip.
- Color jitter.
- Multi-scale training.

## Loss

Loss gom cac thanh phan:

- Focal loss cho classification.
- CIoU loss cho box regression.
- BCEWithLogits cho centerness.
- Class weight tu thong ke tap train, co boost nhe cho `chair`.

## Train

Lenh bat buoc theo de bai:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Checkpoint tot nhat duoc luu tai:

```text
./models/best.pth
```

Checkpoint moi nhat duoc luu tai:

```text
./models/last.pth
```

Resume:

```bash
python train.py ... --resume_from_best
python train.py ... --resume_from_last
```

Lenh train khuyen dung tren Kaggle:

```bash
python train.py \
  --train_data /kaggle/input/datasets/trandthang/final-public/public/annotations/train.json \
  --val_data /kaggle/input/datasets/trandthang/final-public/public/annotations/val.json \
  --image_dir /kaggle/input/datasets/trandthang/final-public/public/train/images \
  --val_image_dir /kaggle/input/datasets/trandthang/final-public/public/val/images \
  --checkpoint_dir ./models/ \
  --img_size 640 \
  --batch_size 16 \
  --val_batch_size 32 \
  --lr 1.5e-4 \
  --scheduler onecycle \
  --eval_map_every 5 \
  --early_stopping_patience 10
```

Neu can quay ve baseline cu:

```bash
--disable_p6 --disable_level_scales
```

## Predict

Lenh bat buoc theo de bai:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Mac dinh:

- `--checkpoint ./models/best.pth`
- `--conf_threshold 0.05`
- `--max_detections_per_image 30`
- TTA bat mac dinh

Co the tat TTA de chay nhanh:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json \
  --disable_tta
```

## Output

`predictions.json` la mot mang JSON:

```json
[
  {
    "image_id": "example.jpg",
    "boxes": [
      {
        "class": "person",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

Moi anh trong `image_dir` deu co mot phan tu output. Anh khong co detection se co:

```json
{"image_id": "example.jpg", "boxes": []}
```

NMS duoc cai dat trong `utils/nms.py` va duoc chay rieng theo tung class.

## Evaluate

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```
