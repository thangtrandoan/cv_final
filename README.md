# TinyGridDetector Object Detection

Project nay cai dat mot one-stage detector nho theo huong YOLO-like cho bo du lieu `public`. Mo hinh phat hien 5 lop: `person`, `car`, `dog`, `cat`, `chair`.

## Cau truc

```text
models/
  detector.py
utils/
  anchors.py
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

## Cai dat

```bash
pip install -r requirements.txt
```

## Du lieu

Dataset nam trong `public/`:

- `public/annotations/train.json`: ground truth train.
- `public/annotations/val.json`: ground truth validation.
- `public/train/images`: anh train.
- `public/val/images`: anh validation.

Annotation dung bbox dang `[xmin, ymin, xmax, ymax]`. Dataset co anh khong co object, cac anh nay van duoc dua vao train de model hoc background.

## Mo hinh

`TinyGridDetector` la custom CNN backbone tu cac khoi `Conv2d + BatchNorm2d + LeakyReLU`, theo sau la detection head:

- Input: `3 x 416 x 416`
- Output: `[B, 30, 13, 13]`
- Grid: `13 x 13`
- Anchor: `3`
- Moi anchor du doan: `tx, ty, tw, th, objectness, class_logits`

Project khong dung YOLOv5/v8, Detectron2, MMDetection hay detection model co san trong torchvision.

## Loss

Loss gom:

- SmoothL1 cho bbox tren anchor co object.
- BCEWithLogits cho objectness.
- BCEWithLogits cho no-object voi trong so nho hon.
- CrossEntropy cho class, co class weight tinh tu tap train.

Tong loss:

```text
loss = 5.0 * box_loss + 1.0 * obj_loss + 0.3 * noobj_loss + 1.0 * cls_loss
```

## Train

Lenh theo yeu cau:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Co the them KMeans anchors:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --use_kmeans_anchors
```

Checkpoint tot nhat duoc luu tai `models/best.pth` va gom `model_state_dict`, `class_names`, `anchors`, `img_size`, `grid_size`, `best_metric`.

## Predict

Lenh theo yeu cau:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json
```

Default:

- `--checkpoint ./models/best.pth`
- `--img_size 416`
- `--conf_threshold 0.20`
- `--nms_threshold 0.50`

Output la JSON co du moi anh trong `image_dir`. Anh khong co detection se co `"boxes": []`. Bbox duoc scale nguoc ve toa do anh goc.

## Evaluate

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

NMS duoc chay rieng theo tung class de tranh loai nham box khac lop.
