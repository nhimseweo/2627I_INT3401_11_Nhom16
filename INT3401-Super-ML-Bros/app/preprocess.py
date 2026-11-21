"""
Biến ảnh vẽ tay (220x220, nét trắng trên nền đen) thành đầu vào kiểu MNIST.

Làm giống cách tạo MNIST gốc:
  1. cắt sát bounding box của nét vẽ
  2. phóng/thu sao cho cạnh dài nhất = 20 px (giữ tỉ lệ)
  3. đặt vào ô 28x28 sao cho trọng tâm khối lượng nằm ở giữa ảnh
Kết quả: vector float32 (784,) trong [0, 1] (cùng định dạng với mnist.npz của đề).
"""
import numpy as np
from PIL import Image

try:  # Pillow >= 9.1
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _LANCZOS = Image.LANCZOS


def _shift(a, dy, dx):
    """Dịch mảng 2D dy, dx pixel, phần tràn ra ngoài bị cắt, chỗ trống = 0."""
    out = np.zeros_like(a)
    h, w = a.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def preprocess(img, threshold=16):
    """
    img: PIL.Image (mode L hoặc tương đương), nét trắng trên nền đen.
    Trả về (vec784 float32, img28 float32 [28,28]) hoặc (None, None) nếu ảnh trống.
    """
    gray = img.convert("L")
    arr = np.asarray(gray)
    ys, xs = np.nonzero(arr > threshold)
    if len(ys) == 0:
        return None, None

    crop = gray.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    w, h = crop.size
    scale = 20.0 / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    small = np.asarray(crop.resize((nw, nh), _LANCZOS), dtype=np.float32) / 255.0
    small = np.clip(small, 0.0, 1.0)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small

    total = canvas.sum()
    if total > 0:
        cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
        cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
        canvas = _shift(canvas, int(round(13.5 - cy)), int(round(13.5 - cx)))

    return canvas.reshape(-1).astype(np.float32), canvas
