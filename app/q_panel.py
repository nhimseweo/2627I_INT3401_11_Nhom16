"""Widget dùng chung cho các panel Q: khung panel, thanh xác suất, hình matplotlib."""
import tkinter as tk
from tkinter import ttk

from app import theme
from app.widgets import PixelFrame

MSG_COLORS = theme.MSG_COLORS

ROW_H = 10   # chiều cao 1 thanh xác suất (nhỏ gọn để đủ 10 thanh cho chữ số trong panel)

STATUS_SPRITE = {"ok": "star", "partial": "mushroom", "running": "coin", "fail": "goomba", "unknown": "qblock"}


class QPanel(PixelFrame):
    """Khung có tiêu đề; plugin đặt nội dung vào .body và báo trạng thái bằng set_message()."""

    def __init__(self, parent, title):
        super().__init__(parent, title, "qblock")
        # dòng thông báo nằm NGOÀI .body (main_window xóa .body khi nạp lại plugin)
        self._msg = ttk.Label(self.inner, text="", justify="left", wraplength=280, font=theme.font(9, False))
        self._msg.pack(side="bottom", fill="x", padx=8, pady=(0, 6), before=self.body)
        self.bind("<Configure>", lambda e: self._msg.configure(wraplength=max(120, e.width - 30)))

    def set_message(self, text, kind="info"):
        self._msg.configure(text=text, foreground=MSG_COLORS.get(kind, MSG_COLORS["info"]))

    def set_status(self, status):
        """status: ok | partial | running | fail | unknown -> đổi sprite ở thanh tiêu đề."""
        self.set_sprite(STATUS_SPRITE.get(status, "qblock"))


def make_figure(parent, figsize=(3.2, 2.4)):
    """Tạo Figure matplotlib nhúng trong Tk. Trả về (fig, canvas)."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    theme.apply_matplotlib()
    fig = Figure(figsize=figsize, dpi=100, layout="constrained", facecolor=theme.PANEL)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    widget = canvas.get_tk_widget()
    widget.configure(bg=theme.PANEL, highlightthickness=0)
    widget.pack(fill="both", expand=True)
    return fig, canvas


class ProbBars(ttk.Frame):
    """Các thanh ngang hiển thị xác suất của từng lớp; lớp cao nhất tô đỏ."""

    def __init__(self, parent, labels, label_width=9, row_height=ROW_H):
        super().__init__(parent)
        self.row_height = row_height
        self.labels = list(labels)
        self._probs = None
        self._rows = []
        self.columnconfigure(1, weight=1)
        for i, name in enumerate(self.labels):
            tk.Label(self, text=str(name), width=label_width, anchor="e", bg=theme.PANEL, fg=theme.INK,
                     font=theme.font(8), bd=0, padx=0, pady=0).grid(row=i, column=0, sticky="e")
            bar = tk.Canvas(self, height=self.row_height, bg=theme.BRICK_DARK, highlightthickness=1,
                            highlightbackground=theme.BLACK)
            bar.grid(row=i, column=1, sticky="ew", padx=4, pady=0)
            bar.bind("<Configure>", lambda e: self._redraw())
            val = tk.Label(self, text="", width=5, anchor="w", bg=theme.PANEL, fg=theme.INK,
                           font=theme.font(8), bd=0, padx=0, pady=0)
            val.grid(row=i, column=2, sticky="w")
            self._rows.append((bar, val))

    def set(self, probs):
        self._probs = None if probs is None else [float(p) for p in probs]
        self._redraw()

    def _redraw(self):
        probs = self._probs
        best = None if not probs else max(range(len(probs)), key=probs.__getitem__)
        for i, (bar, val) in enumerate(self._rows):
            bar.delete("all")
            if not probs:
                val.configure(text="")
                continue
            p = max(0.0, min(1.0, probs[i]))
            w = max(1, bar.winfo_width())
            fill = theme.RED if i == best else theme.SKY_DARK
            bar.create_rectangle(0, 0, int(w * p), self.row_height, fill=fill, width=0)
            bar.create_rectangle(0, 0, int(w * p), 3, fill="#FFFFFF", width=0, stipple="gray50")
            val.configure(text=f"{p:.0%}")


class DigitResultView(ttk.Frame):
    """Chữ số dự đoán to (bên trái) + 10 thanh xác suất (bên phải) — dùng cho Q3 và Q5."""

    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.big = ttk.Label(self, text="?", font=theme.font(44), foreground=theme.RED, anchor="center", width=3)
        self.big.grid(row=0, column=0, sticky="n", padx=(0, 4))
        self.bars = ProbBars(self, [str(i) for i in range(10)], label_width=2, row_height=17)
        self.bars.grid(row=0, column=1, sticky="nsew")

    def show(self, probs):
        if probs is None:
            self.big.configure(text="?")
            self.bars.set(None)
            return
        best = max(range(len(probs)), key=lambda i: probs[i])
        self.big.configure(text=str(best))
        self.bars.set(probs)
