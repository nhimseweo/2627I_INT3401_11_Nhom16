# Super ML Bros. — CS188 Project 5 (Machine Learning)

Bài tập lớn môn **INT3401_11** — nhóm 16. Dự án hiện thực lại *Project 5: Machine Learning* của CS188 (UC Berkeley) bằng **PyTorch**, kèm một **giao diện GUI (Tkinter)** theo phong cách Super Mario để train, kiểm tra và chạy autograder cho từng câu.

> Số liệu trong README này ứng với mã nguồn **V11** (xem báo cáo kèm theo).

## Nội dung dự án

| Câu | Thư mục | Nội dung | Điểm |
|-----|---------|----------|------|
| Q1 | `q1_perceptron/` | Perceptron | 5 |
| Q2 | `q2_regression/` | Hồi quy phi tuyến (xấp xỉ hàm sin) | 5 |
| Q3 | `q3_digit/` | Phân loại chữ số MNIST bằng MLP | 5 |
| Q4 | `q4_langid/` | Nhận dạng ngôn ngữ (RNN) | 5 |
| Q5 | `q5_conv/` | Mạng tích chập (CNN) cho chữ số | 3 |
| Q6 | `q6_attention/` | Cơ chế Attention | 2 |
| Q7 | `q7/` | Character-level GPT trên văn bản Shakespeare (tùy chọn) | 0 |

Cấu trúc các thư mục khác:

- `app/` — mã giao diện GUI (cửa sổ chính, canvas vẽ chữ số, thanh tiến độ, ...).
- `core/` — mã và dữ liệu do giảng viên cung cấp (`backend.py`, `autograder.py`, `data/mnist.npz`, `data/lang_id.npz`, ...).
- `tools/` — script dòng lệnh: train, chạy autograder, ghép file nộp, chạy Q7.
- `build/`, `submission/` — 3 file `models.py`, `losses.py`, `train.py` được **tự động ghép** từ các thư mục `q*/` (`submission/` là bản dùng để nộp Gradescope).
- `qN_xxx/weights/` — trọng số đã train sẵn và `metrics.json` (kết quả).

## Yêu cầu

- Python **3.9 trở lên** (đã chạy thử trên Python 3.12 và 3.13; kiểm thử V11 trên Linux, PyTorch 2.10.0, CPU)
- `tkinter` (đi kèm Python trên Windows/macOS; Linux: `sudo apt install python3-tk`)
- Các thư viện trong `requirements.txt`: `torch`, `numpy`, `matplotlib`, `pillow`

## Cài đặt

```bash
git clone https://github.com/nhimseweo/2627I_INT3401_11_Nhom16.git
cd 2627I_INT3401_11_Nhom16

# (khuyến nghị) tạo môi trường ảo
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

## Cách chạy

### 1. Chạy giao diện GUI

```bash
python main.py
```

Trong GUI có thể chọn từng câu, bấm **Train**, kiểm tra kết quả và (với Q3, Q5) vẽ chữ số để dự đoán.

### 2. Chạy autograder (dòng lệnh)

```bash
python tools/run_autograder.py q1        # chấm 1 câu (q1 ... q6)
python tools/run_autograder.py all       # chấm tất cả
python tools/run_autograder.py q3 --graphics   # bật cửa sổ matplotlib của đề
```

Điểm được lưu vào `.cache/results.json`.

### 3. Train lại một câu

```bash
python tools/train_q.py q3               # q1 ... q5
```

Trọng số mới được lưu vào `qN_xxx/weights/`.

### 4. Ghép file nộp bài (Gradescope)

```bash
python tools/build_submission.py
```

Sinh `models.py`, `losses.py`, `train.py` trong `submission/` (và `build/`).

### 5. Q7 — Character-GPT (tùy chọn)

Lần chạy đầu sẽ tự tải `input.txt` (Shakespeare) vào thư mục gốc dự án — cần có kết nối mạng.

```bash
python tools/run_q7.py train --steps 10000 --batch 32
python tools/run_q7.py generate --prompt "Romeo:" --length 200 --temperature 0.8 --top-k 20
```

Trọng số lưu tại `q7/weights/model.pt`.

> **Lưu ý:** trong lần kiểm thử lại V11, môi trường không tải được dữ liệu Shakespeare nên chưa có kết quả huấn luyện đầy đủ trên corpus này. Q7 mới được kiểm thử chức năng (train, lưu/nạp checkpoint, sinh ký tự) trên dữ liệu tổng hợp; chất lượng sinh văn bản Shakespeare chưa được đánh giá.

## Kết quả

### Điểm Autograder (chạy lại bằng `python tools/run_autograder.py all`)

| Câu | Kết quả kiểm thử | Điểm |
|-----|------------------|------|
| Q1 | PASS | 5/5 |
| Q2 | MSE ≈ 9,8e-5 | 5/5 |
| Q3 | Test accuracy 97,92% | 5/5 |
| Q4 | Test accuracy 91,00% | 5/5 |
| Q5 | Test rút gọn 100% | 3/3 |
| Q6 | PASS | 2/2 |
| **Tổng** | | **25/25** |

Đây là điểm chấm cục bộ, không phải điểm chính thức của học phần.

### Đánh giá checkpoint đã lưu (trọng số đi kèm repo)

Xem chi tiết trong `qN_xxx/weights/metrics.json` và báo cáo.

| Câu | Tập đánh giá | Kết quả |
|-----|--------------|---------|
| Q1 | 10000 điểm sinh mới (100% trên 500 điểm train) | 99,90% |
| Q2 | MSE trên 2001 điểm | ≈ 9,58e-5 (MAE 0,006168; sai số lớn nhất 0,059354) |
| Q3 | MNIST test (5000 ảnh) | 97,70% |
| Q4 | Test (500 từ) | 90,00% (dev: 89,40%) |
| Q5 | MNIST test (5000 ảnh) | 97,02% |
| Q6 | Kiểm thử tính nhân quả | Đạt |

Ghi chú khi đọc kết quả:

- **Autograder và checkpoint cho số khác nhau** (ví dụ Q3: 97,92% so với 97,7%; Q4: 91,00% so với 90,0%) vì Autograder tự tạo và huấn luyện mô hình riêng.
- **Q5:** nhánh Autograder dùng 200 ảnh huấn luyện, nhánh GUI dùng 60.000 ảnh (kèm Test-Time Augmentation khi dự đoán chữ số vẽ tay). Tập dev/test rút gọn của Autograder giao với tập train nên 100% không đại diện cho độ chính xác thực.
- Q3 và Q5 đều giảm khoảng 7 điểm phần trăm khi ảnh bị dịch 1 pixel (chi tiết trong báo cáo).

## Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | Nguyễn Phương Thảo | 24023068 | Nhóm trưởng — Q2, Q6, tổng hợp báo cáo |
| 2 | Nguyễn Phương Anh | 24022940 | Q3, Q7, GitHub, báo cáo |
| 3 | Nguyễn Thùy Dương | 24022975 | Q4, Q7, báo cáo |
| 4 | Cao Thùy Giang | 24022980 | Q1, Q5, báo cáo |

## Ghi chú

- Dự án dựa trên bộ khung và autograder của [CS188 Project 5](https://inst.eecs.berkeley.edu/~cs188/) (UC Berkeley), phục vụ mục đích học tập.
- Giao diện GUI chạy trên CPU là đủ; có GPU (CUDA) thì Q7 sẽ tự dùng.
