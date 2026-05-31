# Ke hoach trien khai bai cuoi ky Object Detection

## 1. Muc tieu bai toan

Bai toan yeu cau cai dat mot mo hinh phat hien doi tuong tu dau tren bo du lieu `public`.

Mo hinh can phat hien 5 lop:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

Dau vao la anh tu nhien kich thuoc khong co dinh. Dau ra la file `predictions.json` gom danh sach bounding box, lop du doan va confidence cho tung anh.

Yeu cau quan trong:

- Khong dung YOLOv5/v8, Detectron2, MMDetection.
- Khong dung Faster R-CNN/SSD co san trong torchvision.
- Duoc dung PyTorch va cac lop mang co ban.
- Co the dung backbone pretrained ImageNet neu giang vien cho phep.
- Phai tu cai dat cac phan chinh: dataset, augmentation, detection head, loss, inference, confidence threshold va NMS.

---

## 2. Phan tich bo du lieu

### 2.1. Cau truc du lieu

```text
public/
  classes.json
  annotations/
    train.json
    val.json
    oracle_train_predictions.json
    oracle_val_predictions.json
  train/
    images/
  val/
    images/
  tools/
    evaluate_predictions.py
```

Trong do:

- `train.json`: ground truth cho tap train.
- `val.json`: ground truth cho tap validation.
- `oracle_train_predictions.json`, `oracle_val_predictions.json`: prediction mau dung 100%, confidence = 1.0.
- `evaluate_predictions.py`: script cham diem mAP@0.5.

### 2.2. Thong ke du lieu

| Tap | So anh | So annotation | So anh co nhan | Box/anh co nhan | Anh khong co nhan |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 7,500 | 10,642 | 5,000 | 2.13 | 2,500 |
| val | 1,500 | 2,021 | 1,000 | 2.02 | 500 |

Nhan xet:

- 1/3 so anh khong co doi tuong.
- Moi anh co nhan trung binh chi khoang 2 doi tuong.
- Dataset vua phai, phu hop voi one-stage detector nho.
- Can giu anh khong co nhan trong train de model hoc truong hop background.

### 2.3. Phan bo lop

| Lop | Train | Val | Tong |
| --- | ---: | ---: | ---: |
| `person` | 5,829 | 1,074 | 6,903 |
| `chair` | 1,613 | 282 | 1,895 |
| `car` | 1,339 | 283 | 1,622 |
| `dog` | 1,028 | 206 | 1,234 |
| `cat` | 833 | 176 | 1,009 |

Nhan xet:

- Lop `person` chiem ty le rat lon.
- Lop `cat`, `dog`, `car`, `chair` it hon nhieu.
- Neu train binh thuong, model co the thien ve du doan `person`.
- Can dung class weight hoac focal loss nhe cho classification/objectness.

### 2.4. Kich thuoc anh

- Train: width tu 142 den 500 px, height tu 71 den 500 px.
- Val: width tu 191 den 500 px, height tu 112 den 500 px.
- Da so anh co kich thuoc nho/vua, width toi da 500 px.

Nhan xet:

- Khong can dung input qua lon nhu 640.
- Nen chon `img_size = 416` de can bang giua toc do va do chinh xac.
- Neu GPU yeu, co the dung `img_size = 320`.

---

## 3. Huong mo hinh de xuat

Nen chon mo hinh one-stage detector kieu YOLO nho, tu cai dat.

Ten goi de dat trong bao cao/code:

```text
TinyGridDetector
```

Hoac:

```text
MiniYOLO-5Class
```

### Ly do chon huong YOLO-like

- Phu hop dataset co so lop it.
- De tu cai dat hon Faster R-CNN/SSD.
- Co day du thanh phan de an diem: backbone, detection head, bbox regression, objectness, classification, NMS.
- De giai thich khi bi phong van code.
- Chay duoc voi 7,500 anh train.

---

## 4. Thiet ke input va output

### 4.1. Input

Anh duoc resize ve:

```text
416 x 416
```

Sau do normalize ve `[0, 1]`.

Co the dung ImageNet normalization neu dung pretrained backbone:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Neu dung custom CNN tu dau, co the chi normalize ve `[0, 1]` truoc.

### 4.2. Output

Dung grid size:

```text
S = 13
```

Dung so anchor:

```text
A = 3
```

So lop:

```text
C = 5
```

Moi anchor du doan:

```text
tx, ty, tw, th, objectness, class_logits
```

Tong so gia tri moi anchor:

```text
5 + C = 10
```

Output cua model:

```text
[B, A * (5 + C), S, S]
```

Cu the:

```text
[B, 30, 13, 13]
```

Sau reshape:

```text
[B, 13, 13, 3, 10]
```

---

## 5. Anchor box

### 5.1. Ban dau nen dung anchor thu cong

Co the dat anchor theo ti le kich thuoc anh:

```python
anchors = [
    [0.08, 0.12],
    [0.18, 0.25],
    [0.35, 0.45],
]
```

Trong do moi anchor la `[w, h]` da normalize theo kich thuoc anh.

### 5.2. Ban cai tien nen tinh anchor bang KMeans

Vi dataset da co train annotation, nen co the lay tat ca bbox train va gom cum kich thuoc `(w, h)`.

Cach lam:

1. Doc tat ca bbox trong `train.json`.
2. Tinh:
   - `w = (xmax - xmin) / image_width`
   - `h = (ymax - ymin) / image_height`
3. Chay KMeans voi `k = 3` hoac `k = 5`.
4. Sap xep anchor theo dien tich tang dan.
5. Luu vao config.

Neu muon dat diem cao hon, nen cai dat KMeans anchor vi day la cai tien hop ly va de giai thich.

---

## 6. Xu ly du lieu

### 6.1. Dataset class

File nen tao:

```text
utils/dataset.py
```

Dataset can tra ve:

```python
image, target
```

Trong do:

- `image`: tensor `[3, 416, 416]`
- `target`: danh sach object cua anh

Moi object gom:

```python
{
    "class_id": int,
    "bbox": [xmin, ymin, xmax, ymax]
}
```

Hoac bbox normalized:

```python
{
    "class_id": int,
    "bbox": [cx, cy, w, h]
}
```

### 6.2. Xu ly anh khong co nhan

Dataset co 2,500 anh train khong co annotation. Khong duoc bo qua.

Voi anh khong co nhan:

```python
targets = []
```

Khi encode target YOLO:

- Tat ca objectness = 0.
- Khong tinh box loss.
- Khong tinh class loss.
- Chi tinh no-object loss.

Day la diem rat quan trong vi 1/3 dataset la anh background.

### 6.3. Collate function

Vi moi anh co so luong object khac nhau, can viet `collate_fn`.

Vi du:

```python
def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets
```

---

## 7. Tang cuong du lieu

### 7.1. Bat buoc nen co

- Resize ve 416x416.
- Random horizontal flip.
- Color jitter nhe.

### 7.2. Horizontal flip bbox

Neu anh co kich thuoc goc `W`, bbox cu la:

```text
[xmin, ymin, xmax, ymax]
```

Sau khi flip ngang:

```text
new_xmin = W - xmax
new_xmax = W - xmin
```

`ymin`, `ymax` giu nguyen.

### 7.3. Nen tranh trong ban dau

Khong nen lam random crop ngay tu dau vi de lam mat object hoac tao bbox loi.

Thu tu khuyen nghi:

1. Ban dau: resize + horizontal flip.
2. Khi model chay dung: them color jitter.
3. Sau khi co baseline tot: them random scale/crop neu can.

---

## 8. Kien truc mo hinh

### 8.1. Ban an toan: Custom CNN Backbone

File:

```text
models/detector.py
```

Kien truc:

```text
Input 3 x 416 x 416
↓
ConvBlock 3 -> 32
MaxPool
↓
ConvBlock 32 -> 64
MaxPool
↓
ConvBlock 64 -> 128
MaxPool
↓
ConvBlock 128 -> 256
MaxPool
↓
ConvBlock 256 -> 512
MaxPool
↓
ConvBlock 512 -> 1024
↓
Detection Head
Conv 1024 -> 512
Conv 512 -> 30
↓
Output B x 30 x 13 x 13
```

`ConvBlock`:

```text
Conv2d
BatchNorm2d
LeakyReLU
```

Uu diem:

- Tu cai dat ro rang.
- De giai thich.
- Phu hop voi yeu cau "lam tu dau".

Nhuoc diem:

- Co the mAP khong cao bang pretrained backbone.

### 8.2. Ban cai tien: ResNet18 pretrained backbone neu duoc phep

Neu giang vien cho phep dung pretrained ImageNet:

```text
ResNet18 pretrained
bo avgpool va fc
them detection head tu cai
```

Luu y:

- Khong dung detection model co san.
- Chi dung ResNet18 nhu feature extractor.
- Target assignment, loss, NMS van tu cai dat.

Trong README co the ghi:

```text
Mo hinh chi su dung ResNet18 pretrained lam backbone trich xuat dac trung. 
Phan detection head, target assignment, loss function, inference va NMS duoc tu cai dat.
```

---

## 9. Target encoding

Voi moi bbox:

1. Chuyen bbox tu `[xmin, ymin, xmax, ymax]` sang normalized center format:

```text
cx = (xmin + xmax) / 2 / image_width
cy = (ymin + ymax) / 2 / image_height
bw = (xmax - xmin) / image_width
bh = (ymax - ymin) / image_height
```

2. Tim cell:

```text
grid_x = int(cx * S)
grid_y = int(cy * S)
```

3. Chon anchor tot nhat dua tren IoU giua bbox size va anchor size.

4. Gan target:

```text
target[grid_y, grid_x, anchor, 0:4] = [cx, cy, bw, bh]
target[grid_y, grid_x, anchor, 4] = 1
target[grid_y, grid_x, anchor, 5] = class_id
```

5. Neu co 2 object roi vao cung cell va cung anchor:

- Giu object co IoU voi anchor cao hon, hoac
- Ghi de bang object moi.

Voi dataset trung binh 2 box/anh, truong hop nay co the khong qua nhieu.

---

## 10. Loss function

Tong loss:

```text
loss = lambda_box * box_loss 
     + lambda_obj * obj_loss 
     + lambda_noobj * noobj_loss 
     + lambda_cls * cls_loss
```

Goi y trong so:

```python
lambda_box = 5.0
lambda_obj = 1.0
lambda_noobj = 0.3
lambda_cls = 1.0
```

Ly do giam `lambda_noobj`:

- Moi anh co rat nhieu anchor khong co object.
- Dataset co 1/3 anh khong co nhan.
- Neu no-object loss qua lon, model co the hoc cach khong du doan gi de giam loss.

### 10.1. Box loss

Ban dau dung:

```python
SmoothL1Loss
```

Chi tinh tren anchor co object.

Sau do co the cai tien thanh:

```text
1 - IoU
```

Hoac:

```text
GIoU loss
```

### 10.2. Objectness loss

Dung:

```python
BCEWithLogitsLoss
```

Tach rieng:

- object anchor
- no-object anchor

### 10.3. Classification loss

Dung:

```python
CrossEntropyLoss
```

Chi tinh tren anchor co object.

Vi lop `person` nhieu hon cac lop khac, nen co the dung class weight.

Goi y class weight tinh tu train:

```python
class_counts = {
    "person": 5829,
    "chair": 1613,
    "car": 1339,
    "dog": 1028,
    "cat": 833,
}
```

Co the dat weight nguoc tan suat, nhung nen lam mem bang sqrt:

```text
weight_c = sqrt(total / count_c)
```

Sau do normalize ve trung binh 1.

---

## 11. Inference

Trong `predict.py`, can thuc hien:

1. Load model tu `models/best.pth`.
2. Doc tat ca anh trong `--image_dir`.
3. Resize anh ve 416x416.
4. Chay model.
5. Decode output thanh bbox.
6. Tinh confidence:

```text
confidence = sigmoid(objectness) * max_softmax(class_logits)
```

7. Loc theo nguong confidence.
8. Chay NMS theo tung class.
9. Scale bbox ve kich thuoc anh goc.
10. Ghi `predictions.json`.

### 11.1. Confidence threshold

Nen tune tren validation.

Gia tri nen thu:

```text
0.10, 0.15, 0.20, 0.25, 0.30
```

Goi y ban dau:

```python
conf_threshold = 0.20
```

Vi mAP can recall tot, khong nen de threshold qua cao.

### 11.2. NMS threshold

Nen thu:

```text
0.4, 0.5, 0.6
```

Goi y ban dau:

```python
nms_threshold = 0.5
```

### 11.3. Anh khong phat hien gi

Van phai ghi:

```json
{
  "image_id": "ten_anh.jpg",
  "boxes": []
}
```

Khong duoc bo qua anh.

---

## 12. NMS tu cai dat

File:

```text
utils/nms.py
```

Quy trinh:

1. Sap xep box theo confidence giam dan.
2. Lay box co confidence cao nhat.
3. Tinh IoU voi cac box con lai.
4. Loai cac box co IoU lon hon threshold.
5. Lap lai den khi het box.

NMS can chay rieng cho tung class:

```python
for class_id in range(num_classes):
    boxes_of_class = ...
    keep = nms(boxes_of_class, scores_of_class)
```

Khong nen chay NMS chung moi class, vi box `person` va `chair` co the trung nhau nhung deu hop le.

---

## 13. Training pipeline

File:

```text
train.py
```

Lenh bat buoc:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

### 13.1. Hyperparameter goi y

Ban dau:

```python
img_size = 416
grid_size = 13
batch_size = 16
epochs = 80
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
```

Neu loss giam cham:

```python
lr = 3e-4
```

Neu bi overfit:

```python
weight_decay = 5e-4
```

Neu GPU yeu:

```python
img_size = 320
batch_size = 16
```

### 13.2. Luu best model

Moi epoch chay validation, tinh loss hoac mAP.

Tot nhat:

- Sau moi epoch, predict tren val.
- Chay evaluate script de lay mAP.
- Neu mAP cao nhat thi luu `models/best.pth`.

Neu muon don gian:

- Luu theo `val_loss`.
- Cuoi cung chay evaluation de chon checkpoint.

Nhung de diem cao, nen luu theo `val_mAP@0.5`.

---

## 14. Validation va cham diem

Sau khi train, chay:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json
```

Sau do cham:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

Can kiem tra:

- JSON co du 1,500 anh val.
- Moi anh co field `image_id` va `boxes`.
- Bbox nam trong anh goc.
- Confidence trong `[0, 1]`.
- Class dung 5 lop.

---

## 15. Su dung oracle prediction de debug

Dataset co:

```text
oracle_train_predictions.json
oracle_val_predictions.json
```

Cach dung:

1. Chay evaluate tren oracle val:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions public/annotations/oracle_val_predictions.json \
  --output oracle_val_score.json
```

2. Neu score dung gan 1.0, script cham dang hoat dong binh thuong.
3. Lay format oracle de so sanh voi prediction cua minh.
4. Dam bao prediction cua minh co cau truc JSON giong oracle.

Oracle chi dung de debug format, khong duoc dung lam ket qua nop cho hidden test.

---

## 16. Chien luoc xu ly imbalance

Vi `person` qua nhieu, can tranh model chi hoc tot `person`.

Nen ap dung theo muc do:

### Muc 1: Dung class weight trong classification loss

Tinh weight tu tan suat class:

```text
person: thap nhat
cat/dog: cao hon
```

Dung trong `CrossEntropyLoss(weight=class_weights)`.

### Muc 2: Data augmentation giu nguyen cho lop it

Khong nen oversampling phuc tap ngay tu dau.

### Muc 3: Theo doi AP tung lop

Neu evaluate script co AP per class, xem lop nao yeu.

Neu `cat`/`dog` yeu:

- Giam confidence threshold.
- Tang augmentation nhe.
- Dung class weight.

---

## 17. Chien luoc xu ly anh khong co object

Vi train co 2,500 anh khong co nhan, day la diem rat quan trong.

Neu train sai, model co the:

- Du doan qua nhieu false positive tren anh background.
- Hoac hoc cach khong du doan gi.

Khuyen nghi:

- Giu anh background trong train.
- Dung no-object loss voi trong so thap, vi negative anchor rat nhieu.
- Khi validation, dam bao anh khong co object van output `boxes: []`.

Neu false positive nhieu:

- Tang confidence threshold.
- Giam NMS threshold khong giai quyet false positive, chi giai quyet box trung lap.
- Tang `lambda_noobj` tu 0.3 len 0.5.

Neu recall thap, model it du doan:

- Giam confidence threshold.
- Giam `lambda_noobj`.
- Tang `lambda_box` hoac train lau hon.

---

## 18. Lo trinh lam bai

### Buoc 1: Kiem tra format du lieu

Viet script nho de:

- Doc `classes.json`.
- Doc `train.json`, `val.json`.
- Dem so anh, so bbox.
- Ve thu bbox len 10 anh.

Muc tieu:

- Hieu dung format.
- Kiem tra bbox co nam trong anh khong.
- Xac nhan class mapping.

### Buoc 2: Cai dataset va transform

Can lam duoc:

```python
dataset = DetectionDataset(...)
image, target = dataset[0]
```

Kiem tra:

- image shape dung.
- bbox sau resize/flip dung.
- anh khong co bbox khong bi loi.

### Buoc 3: Cai target encoder

Input:

```python
targets = list object cua tung anh
```

Output:

```python
target_tensor = [S, S, A, 6]
```

Trong do:

```text
0:4 bbox
4 objectness
5 class_id
```

Hoac tach rieng:

```text
box_target
obj_target
cls_target
```

### Buoc 4: Cai model

Chay thu:

```python
x = torch.randn(2, 3, 416, 416)
y = model(x)
print(y.shape)
```

Ket qua phai la:

```text
[2, 30, 13, 13]
```

### Buoc 5: Cai loss

Train thu 1 batch:

- Loss khong NaN.
- Backward duoc.
- Grad khong loi.

### Buoc 6: Overfit 10 anh

Lay 10-20 anh train, train 100-200 epoch.

Neu model dung, no phai:

- Loss giam manh.
- Du doan duoc bbox tren anh train.
- mAP tren subset cao.

Neu khong overfit duoc, can debug truoc khi train full.

### Buoc 7: Train full

Chay train tren toan bo train.

Goi y:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --epochs 80 \
  --batch_size 16 \
  --img_size 416
```

### Buoc 8: Tune threshold

Chay val voi nhieu threshold:

```text
conf = 0.10, 0.15, 0.20, 0.25, 0.30
nms = 0.4, 0.5, 0.6
```

Chon combo co mAP val cao nhat, ghi vao README va dat default trong `predict.py`.

---

## 19. Cau truc thu muc nop bai

Nen to chuc:

```text
my_submission/
├── models/
│   ├── detector.py
│   └── best.pth
├── utils/
│   ├── dataset.py
│   ├── transforms.py
│   ├── anchors.py
│   ├── loss.py
│   ├── box_ops.py
│   ├── nms.py
│   └── json_utils.py
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

Khong can nop lai `public/` neu de bai noi khong can nop.

---

## 20. Yeu cau voi predict.py

Lenh bat buoc:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

`predict.py` nen co default:

```python
--checkpoint ./models/best.pth
--img_size 416
--conf_threshold 0.20
--nms_threshold 0.50
```

Nhung vi lenh cham chi truyen `--image_dir` va `--output`, cac tham so con lai phai co default.

Dau ra phai la:

```json
[
  {
    "image_id": "img_00090df9158f.jpg",
    "boxes": [
      {
        "class": "dog",
        "confidence": 0.95,
        "bbox": [126, 86, 500, 375]
      }
    ]
  }
]
```

Luu y:

- `image_id` chi la ten file, khong phai duong dan day du.
- Bbox la toa do tren anh goc.
- Anh khong co box van phai xuat `"boxes": []`.

---

## 21. Yeu cau voi train.py

Lenh bat buoc:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

`train.py` can:

1. Tao dataset train/val.
2. Tao dataloader voi `collate_fn`.
3. Tao model.
4. Train nhieu epoch.
5. Validate.
6. Luu checkpoint tot nhat vao:

```text
./models/best.pth
```

Checkpoint nen gom:

```python
{
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "anchors": anchors,
    "img_size": img_size,
    "grid_size": grid_size,
    "best_metric": best_map
}
```

---

## 22. Noi dung README can co

README nen gom:

1. Gioi thieu bai toan.
2. Mo ta dataset.
3. Cau truc thu muc.
4. Cai dat moi truong.
5. Lenh huan luyen.
6. Lenh suy luan.
7. Mo ta mo hinh.
8. Mo ta loss.
9. Mo ta NMS.
10. Ket qua validation.
11. Ghi chu ve pretrained backbone neu co.

Vi du lenh cai dat:

```bash
pip install -r requirements.txt
```

Vi du train:

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Vi du predict:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output val_predictions.json
```

Vi du evaluate:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

---

## 23. Nhung loi can tranh

1. Bo qua anh khong co annotation.
2. Prediction khong co du moi anh.
3. Bbox xuat theo anh resize thay vi anh goc.
4. Bbox sai format COCO `[x, y, w, h]` thay vi `[xmin, ymin, xmax, ymax]`.
5. Class name sai thu tu hoac sai chu thuong.
6. Confidence khong qua sigmoid/softmax nen vuot ngoai `[0, 1]`.
7. NMS chay chung tat ca class.
8. Khong luu `models/best.pth`.
9. `predict.py` yeu cau tham so phu khien lenh cham khong chay duoc.
10. Khong test bang `evaluate_predictions.py` truoc khi nop.
11. Dung oracle prediction sai muc dich.
12. Khong xu ly nhieu object trong cung mot anh.
13. Khong xu ly anh co 0 object.

---

## 24. Chien luoc diem so

### Baseline can dat

- Custom CNN + YOLO-like head.
- Resize + horizontal flip.
- SmoothL1 + BCE + CE.
- NMS tu cai.
- Tune threshold.

Muc tieu:

```text
mAP@0.5 >= 0.45
```

### Ban tot hon

- KMeans anchors.
- Class weight.
- ResNet18 pretrained backbone neu duoc phep.
- Tune threshold tren val.

Muc tieu:

```text
mAP@0.5 >= 0.60
```

### Ban manh hon

- ResNet18/ResNet34 pretrained.
- IoU/GIoU loss.
- Multi-scale training.
- Better augmentation.
- Ensemble khong khuyen khich vi kho giai thich va co the nang.

Muc tieu:

```text
mAP@0.5 >= 0.75
```

---

## 25. Ke hoach thuc hien de xuat

Thu tu nen lam:

```text
Ngay 1:
- Doc du lieu.
- Ve bbox mau.
- Cai Dataset va collate_fn.
- Cai resize + horizontal flip.

Ngay 2:
- Cai model TinyGridDetector.
- Cai target encoder.
- Cai loss.
- Train thu 1 batch.

Ngay 3:
- Overfit 10-20 anh.
- Cai decode bbox.
- Cai NMS.
- Xuat predictions.json.

Ngay 4:
- Train full dataset.
- Chay evaluation tren val.
- Sua loi bbox/format neu co.

Ngay 5:
- Them KMeans anchors.
- Them class weight.
- Tune confidence/NMS.
- Viet README.

Ngay 6 neu con thoi gian:
- Thu ResNet18 pretrained backbone.
- So sanh voi custom CNN.
- Chon checkpoint tot nhat.
```

---

## 26. Ket luan

Voi thong ke dataset hien tai, huong phu hop nhat la:

```text
Mini YOLO anchor-based one-stage detector
```

Cau hinh khuyen nghi:

```text
Input size: 416 x 416
Grid size: 13 x 13
Anchors: 3 hoac 5
Classes: 5
Backbone: Custom CNN, hoac ResNet18 pretrained neu duoc phep
Loss: SmoothL1/IoU + BCEWithLogits + CrossEntropy co class weight
Inference: confidence threshold + NMS theo tung class
Output: predictions.json du moi anh, bbox theo toa do anh goc
```

Diem can chu y nhat trong dataset nay:

- 1/3 anh khong co object, nen phai xu ly negative image tot.
- Lop `person` ap dao, nen can class weight hoac focal loss.
- Anh nho/vua, nen 416 la du hop ly.
- File prediction phai co du moi anh, ke ca anh khong detect gi.
