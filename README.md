# Super ML Bros. — CS188 Project 5 (Machine Learning)

Bài tập lớn môn **INT3401_11** — nhóm 16. Dự án hiện thực lại *Project 5: Machine Learning* của CS188 (UC Berkeley) bằng **PyTorch**, kèm một **giao diện GUI (Tkinter)** theo phong cách Super Mario để train, kiểm tra và chạy autograder cho từng câu.

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

- Python **3.9 trở lên** (đã chạy thử trên Python 3.12)
- `tkinter` (đi kèm Python trên Windows/macOS; Linux: `sudo apt install python3-tk`)
- Các thư viện trong `requirements.txt`: `torch`, `numpy`, `matplotlib`, `pillow`

## Cài đặt

```bash
git clone https://github.com/nhimseweo/INT3401_11_Nhom16.git
cd INT3401_11_Nhom16

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

## Kết quả

Kết quả đo trên máy của nhóm (xem chi tiết trong `qN_xxx/weights/metrics.json` và báo cáo):

| Câu | Kết quả | Câu | Kết quả |
|-----|---------|-----|---------|
| Q1 | 100% | Q4 | 89.4% |
| Q2 | loss ≈ 1e-4 | Q5 | 97.0% |
| Q3 | 97.7% | Q6 | 2/2 điểm |

Điểm autograder: Q1 5/5 · Q2 5/5 · Q3 5/5 · Q5 3/3 · Q6 2/2.

## Thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|-----|-----------|------|---------|
| 1 | Nguyễn Phương Thảo | ... | Nhóm trưởng |
| 2 | Nguyễn Phương Anh | 24022940 | ... |
| 3 | Nguyễn Thùy Dương | ... | ... |
| 4 | Cao Thùy Giang | ... | ... |

## Ghi chú

- Dự án dựa trên bộ khung và autograder của [CS188 Project 5](https://inst.eecs.berkeley.edu/~cs188/) (UC Berkeley), phục vụ mục đích học tập.
- Giao diện GUI chạy trên CPU là đủ; có GPU (CUDA) thì Q7 sẽ tự dùng.
