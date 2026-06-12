# Bài tập: Phát hiện đối tượng từ đầu

## 1. Mục tiêu

Cài đặt một mô hình phát hiện đối tượng (*object detection*) từ đầu và đánh giá trên bộ dữ liệu ảnh được cung cấp.

Sinh viên cần tự xây dựng quy trình huấn luyện và suy luận cho bài toán phát hiện đối tượng, bao gồm:

- Đọc dữ liệu và tiền xử lý ảnh/nhãn.
- Tăng cường dữ liệu.
- Mạng trích xuất đặc trưng, ví dụ mạng tích chập CNN.
- Đầu dự đoán phát hiện đối tượng.
- Hàm mất mát.
- Suy luận, ngưỡng độ tin cậy và khử trùng hộp bao bằng NMS (*Non-Maximum Suppression*).

Không được sử dụng các bộ phát hiện đối tượng hoàn chỉnh như:

- YOLOv5/v8
- Detectron2
- MMDetection
- Faster R-CNN/SSD có sẵn trong `torchvision`

Được phép dùng PyTorch, các lớp mạng cơ bản, và mạng trích xuất đặc trưng đã huấn luyện trước như resnet50, efficeientnet,...

---

## 2. Bộ dữ liệu

Bộ dữ liệu gồm ảnh tự nhiên có đối tượng thuộc 5 lớp:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

### 2.1. Cấu trúc thư mục

```text
public/
├── classes.json
├── train/
│   └── images/
├── val/
│   └── images/
├── annotations/
│   ├── train.json
│   └── val.json
└── tools/
    └── evaluate_predictions.py
```

Trong thư mục `public/` chỉ có tập huấn luyện và tập kiểm định.

Khi chấm tự động, hệ thống mới cung cấp thư mục ảnh kiểm tra ẩn cho `predict.py`; nhãn của tập kiểm tra ẩn được giữ riêng và không công bố.

---

## 3. Định dạng nhãn

Tệp `train.json` và `val.json` có dạng:

```json
{
  "classes": ["person", "car", "dog", "cat", "chair"],
  "images": [
    {
      "id": "img_a13f42c9d8b0.jpg",
      "file_name": "train/images/img_a13f42c9d8b0.jpg",
      "width": 500,
      "height": 375
    }
  ],
  "annotations": [
    {
      "image_id": "img_a13f42c9d8b0.jpg",
      "class": "person",
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```

Quy ước hộp bao:

```text
bbox = [xmin, ymin, xmax, ymax]
```

Tọa độ được tính theo điểm ảnh trên ảnh gốc.

---

## 4. Yêu cầu kỹ thuật

### 4.1. Quy trình dữ liệu

Sinh viên cần cài đặt:

- Bộ đọc dữ liệu.
- Thay đổi kích thước ảnh và chuẩn hóa giá trị điểm ảnh.
- Xử lý nhiều đối tượng trong cùng một ảnh.
- Tăng cường dữ liệu, tối thiểu gồm lật ngang ảnh.

Khuyến khích cài đặt thêm:

- Cắt ngẫu nhiên.
- Thay đổi màu sắc.
- Huấn luyện với nhiều kích thước ảnh.

---

### 4.2. Mô hình phát hiện đối tượng

Mô hình cần dự đoán:

- Hộp bao.
- Nhãn lớp.
- Điểm độ tin cậy hoặc điểm có đối tượng.

Sinh viên có thể chọn một trong các hướng:

- Mô hình dùng hộp neo (*anchor-based*).
- Mô hình không dùng hộp neo (*anchor-free*).
- Mô hình dựa trên lưới kiểu YOLO nhỏ.
- Mô hình kiểu SSD tự cài đặt.

---

### 4.3. Hàm mất mát

Hàm mất mát cần có các thành phần phù hợp với thiết kế mô hình:

- Mất mát phân lớp.
- Mất mát định vị hộp bao.
- Mất mát độ tin cậy hoặc điểm có đối tượng nếu mô hình có thành phần này.

Khuyến khích dùng:

- Cross Entropy
- BCE
- Smooth L1
- IoU / GIoU / DIoU

---

### 4.4. Suy luận

Sinh viên cần cài đặt:

- Ngưỡng độ tin cậy.
- NMS theo từng lớp.
- Chuyển hộp bao về tọa độ ảnh gốc.

---

## 5. Yêu cầu nộp bài

Sinh viên nộp tệp nén của thư mục `<my_submission>/` gồm:

```text
<my_submission>/
├── public/              # vị trí của public/, không cần nộp lại thư mục public
├── models/
├── utils/
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

---

## 6. Lệnh bắt buộc

### 6.1. Lệnh huấn luyện

```bash
python train.py \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/
```

Lệnh huấn luyện trên phải chạy được và lưu mô hình tốt nhất vào:

```text
./models/best.pth
```

Sinh viên có thể thêm tham số khác nếu cần.

---

### 6.2. Lệnh suy luận

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

---

## 7. Nội dung README.md cần có

Tệp `README.md` cần nêu rõ:

- Cách cài đặt môi trường.
- Cách huấn luyện.
- Cách chạy suy luận.
- Vị trí đặt mô hình hoặc trọng số mô hình.

---

## 8. Định dạng kết quả dự đoán

Tệp `predictions.json` phải là một mảng JSON:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
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

Quy định:

- `image_id` là tên file ảnh trong thư mục ảnh cần suy luận.
- `class` thuộc 5 lớp quy định.
- `confidence` là độ tin cậy, nằm trong đoạn `[0, 1]`.
- `bbox` là `[xmin, ymin, xmax, ymax]`.
- Tọa độ `bbox` tính theo điểm ảnh trên ảnh gốc.
- Ảnh không phát hiện đối tượng nào vẫn cần xuất:

```json
{
  "image_id": "example.jpg",
  "boxes": []
}
```

Trong `public/annotations` có các file `predictions.json` chính xác 100% để làm mẫu.

---

## 9. Chấm tự động

Sinh viên có thể tự kiểm tra định dạng và điểm trên tập huấn luyện/kiểm định bằng tệp chấm đi kèm trong thư mục `public/`.

Ví dụ, sau khi xuất dự đoán cho tập kiểm định:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions.json \
  --output val_score.json
```

Hệ thống chấm sẽ:

- Kiểm tra JSON có đúng định dạng không.
- Kiểm tra hộp bao hợp lệ và nằm trong ảnh.
- Kiểm tra nhãn lớp hợp lệ.
- Tính IoU, precision, recall và mAP@0.5 trên tập kiểm tra ẩn.

Lệnh chấm mẫu:

```bash
python tools/evaluate_predictions.py \
  --ground_truth ./private/hidden_test_annotations.json \
  --predictions predictions.json \
  --output score.json
```

---

## 10. Thang điểm

| Nội dung | Điểm |
|---|---:|
| Quy trình dữ liệu | 20 |
| Kiến trúc mô hình | 20 |
| Hàm mất mát và quy trình huấn luyện | 20 |
| Suy luận và NMS | 20 |
| Kết quả trên tập kiểm tra ẩn | 20 |

Thang điểm các phần cài đặt dựa trên mức độ “làm từ đầu” của cài đặt.

---

## 11. Thang điểm kết quả

| mAP@0.5 | Điểm |
|---|---:|
| `< 0.30` | 0 |
| `0.30 - < 0.45` | 5 |
| `0.45 - < 0.60` | 10 |
| `0.60 - < 0.75` | 15 |
| `>= 0.75` | 20 |

---

## 12. Gợi ý cấu trúc cài đặt

Một cấu trúc tham khảo cho bài nộp:

```text
<my_submission>/
├── models/
│   ├── detector.py
│   ├── backbone.py
│   └── losses.py
├── utils/
│   ├── dataset.py
│   ├── transforms.py
│   ├── boxes.py
│   ├── nms.py
│   └── metrics.py
├── train.py
├── predict.py
├── README.md
└── requirements.txt
```

### 12.1. Vai trò các tệp gợi ý

| Tệp | Vai trò |
|---|---|
| `utils/dataset.py` | Đọc ảnh, đọc nhãn JSON, xử lý nhiều đối tượng trong một ảnh |
| `utils/transforms.py` | Resize, normalize, flip ngang, augment ảnh và bbox |
| `models/backbone.py` | CNN trích xuất đặc trưng |
| `models/detector.py` | Mô hình phát hiện đối tượng |
| `models/losses.py` | Hàm mất mát phân lớp, bbox, objectness |
| `utils/boxes.py` | Chuyển đổi tọa độ bbox, tính IoU |
| `utils/nms.py` | Cài đặt Non-Maximum Suppression |
| `train.py` | Huấn luyện, validation, lưu `best.pth` |
| `predict.py` | Suy luận ảnh, lọc confidence, NMS, xuất `predictions.json` |

---

## 13. Checklist trước khi nộp

- [ ] Chạy được lệnh huấn luyện bắt buộc.
- [ ] File `./models/best.pth` được tạo sau huấn luyện.
- [ ] Chạy được lệnh suy luận bắt buộc.
- [ ] File `predictions.json` đúng định dạng.
- [ ] Mỗi ảnh đều có một phần tử trong `predictions.json`.
- [ ] Ảnh không có phát hiện vẫn xuất `"boxes": []`.
- [ ] Tọa độ bbox là tọa độ trên ảnh gốc.
- [ ] Confidence nằm trong `[0, 1]`.
- [ ] Class chỉ thuộc 5 lớp hợp lệ.
- [ ] Đã chạy thử `evaluate_predictions.py` trên tập `val`.
