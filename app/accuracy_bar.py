"""Thanh trạng thái dưới cùng: mỗi Q một "màn chơi" gồm sprite, thanh điểm, kết quả và 2 nút."""
import tkinter as tk
from tkinter import ttk

from app import theme
from app.q_panel import STATUS_SPRITE
from app.sprites import Sprite
from app.widgets import MarioBar

# status -> (sprite, mô tả ngắn)
STATUS = {
    "ok":      ("star",     "ĐỦ ĐIỂM"),
    "partial": ("mushroom", "ĐANG LÊN"),
    "running": ("coin",     "ĐANG CHẠY"),
    "fail":    ("goomba",   "CHƯA QUA"),
    "unknown": ("qblock",   "CHƯA CHẤM"),
}


class _Cell(tk.Frame):
    def __init__(self, parent, index, spec, on_train, on_grade):
        super().__init__(parent, bg=theme.BLACK, padx=3, pady=3)
        self.spec = spec
        inner = tk.Frame(self, bg=theme.PANEL, padx=6, pady=4)
        inner.pack(fill="both", expand=True)
        head = tk.Frame(inner, bg=theme.PANEL)
        head.pack(fill="x")
        col = tk.Frame(head, bg=theme.PANEL)
        col.pack(side="left")
        tk.Label(col, text=f"WORLD 1-{index + 1}", bg=theme.PANEL, fg="#7A6652", font=theme.font(8)).pack(anchor="w")
        tk.Label(col, text=spec.qid.upper(), bg=theme.PANEL, fg=theme.RED, font=theme.font(14)).pack(anchor="w")
        self.icon = Sprite(head, "qblock", scale=2, bg=theme.PANEL)
        self.icon.pack(side="right")
        self.bar = MarioBar(inner)
        self.bar.pack(fill="x", pady=(4, 2))
        self.state = tk.Label(inner, text="CHƯA CHẤM", bg=theme.PANEL, fg="#7A6652", font=theme.font(8), anchor="w")
        self.state.pack(fill="x")
        self.text = tk.Label(inner, text="—", bg=theme.PANEL, fg=theme.INK, font=theme.font(9), anchor="w",
                             justify="left")
        self.text.pack(fill="x")
        row = tk.Frame(inner, bg=theme.PANEL)
        row.pack(fill="x", pady=(5, 0))
        self.btn_train = ttk.Button(row, text="Train", width=6, style="Pipe.TButton",
                                    command=lambda: on_train(spec.qid))
        self.btn_grade = ttk.Button(row, text="Mở GPT" if spec.qid == "q7" else "Autograder", style="Coin.TButton",
                                    command=lambda: on_grade(spec.qid))
        if spec.trainable:
            self.btn_train.pack(side="left")
        self.btn_grade.pack(side="left", padx=(4, 0), fill="x", expand=True)

    def show(self, status, text, bar_value, busy):
        sprite, label = STATUS[status]
        self.icon.set(sprite)
        self.state.configure(text=("TÙY CHỌN" if self.spec.qid == "q7" and status == "unknown" else "ĐÃ CHẠY" if self.spec.qid == "q7" and status == "ok" else label))
        self.text.configure(text=text)
        self.bar.set(bar_value, status)
        state = "disabled" if busy else "normal"
        self.btn_train.configure(state=state)
        self.btn_grade.configure(state=state)


class AccuracyBar(tk.Frame):
    def __init__(self, parent, specs, on_train, on_grade):
        super().__init__(parent, bg=theme.SKY)
        self.cells = {}
        for i, spec in enumerate(specs):
            self.columnconfigure(i, weight=1, uniform="cell")
            cell = _Cell(self, i, spec, on_train, on_grade)
            cell.grid(row=0, column=i, sticky="nsew", padx=3)
            self.cells[spec.qid] = cell

    def update_q(self, qid, status, text, bar_value=0, busy=False):
        self.cells[qid].show(status, text, bar_value, busy)
