"""Widget dựng sẵn kiểu Mario: khung bảng gạch, thanh điểm chia ô, banner đầu trang."""
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from app import theme
from app.sprites import Sprite, draw_sprite, size_of


class PixelFrame(tk.Frame):
    """Khung viền đen, thanh tiêu đề màu gạch + sprite; nội dung đặt vào .body (nền kem)."""

    def __init__(self, parent, title, sprite="qblock"):
        super().__init__(parent, bg=theme.BLACK, padx=3, pady=3)
        inner = self.inner = tk.Frame(self, bg=theme.PANEL)
        inner.pack(fill="both", expand=True)
        bar = tk.Frame(inner, bg=theme.BRICK)
        bar.pack(fill="x")
        self.icon = Sprite(bar, sprite, scale=1, bg=theme.BRICK)
        self.icon.pack(side="left", padx=(6, 2), pady=3)
        self.title_label = tk.Label(bar, text=title, bg=theme.BRICK, fg=theme.WHITE,
                                    font=theme.font(11), anchor="w")
        self.title_label.pack(side="left", fill="x", expand=True, padx=(2, 6))
        tk.Frame(inner, bg=theme.BRICK_DARK, height=3).pack(fill="x")
        self.body = ttk.Frame(inner, padding=6)
        self.body.pack(fill="both", expand=True)

    def set_sprite(self, name):
        self.icon.set(name)


class MarioBar(tk.Canvas):
    """Thanh tiến độ chia 20 ô như thanh thời gian / máu trong game."""

    SEGMENTS = 20
    COLORS = {"ok": theme.COIN, "partial": theme.PIPE, "running": theme.BLUE,
              "fail": theme.RED, "unknown": theme.GRAY}

    def __init__(self, parent, height=18):
        super().__init__(parent, height=height, bg=theme.PANEL, highlightthickness=0, bd=0)
        self.value = 0.0
        self.status = "unknown"
        self.bind("<Configure>", lambda e: self._redraw())

    def set(self, value, status="unknown"):
        self.value = max(0.0, min(100.0, float(value)))
        self.status = status
        self._redraw()

    def _redraw(self):
        self.delete("all")
        w, h = max(40, self.winfo_width()), int(self["height"])
        self.create_rectangle(0, 0, w - 1, h - 1, fill=theme.BLACK, outline=theme.BLACK)
        self.create_rectangle(2, 2, w - 3, h - 3, fill=theme.BRICK_DARK, outline="")
        n = self.SEGMENTS
        gap = 2
        seg = (w - 6 - gap * (n - 1)) / n
        lit = int(round(self.value / 100.0 * n))
        if self.value > 0:
            lit = max(1, lit)
        color = self.COLORS.get(self.status, theme.GRAY)
        for i in range(n):
            x0 = 3 + i * (seg + gap)
            if i < lit:
                self.create_rectangle(x0, 4, x0 + seg, h - 5, fill=color, outline="")
                self.create_rectangle(x0, 4, x0 + seg, 6, fill="#FFFFFF", outline="", stipple="gray50")
            else:
                self.create_rectangle(x0, 4, x0 + seg, h - 5, fill="#5A2000", outline="")


class HeaderBanner(tk.Canvas):
    """Banner đầu trang: bầu trời, mây, đồi, tên app và bảng điểm; mặt đất gạch ở dưới."""

    HEIGHT = 84
    TILE = 16

    def __init__(self, parent, title="SUPER ML BROS.", subtitle="CS188 PROJECT 5 · MACHINE LEARNING"):
        super().__init__(parent, height=self.HEIGHT, bg=theme.SKY, highlightthickness=0, bd=0)
        self.title, self.subtitle = title, subtitle
        self.score_text = "SCORE ?/?"
        self.bind("<Configure>", lambda e: self._redraw())

    def set_score(self, earned, total):
        self.score_text = f"x {earned}/{total}"
        self._redraw()

    # -- vẽ
    def _outlined(self, x, y, text, fnt, fill, outline=theme.BLACK, anchor="w", d=2):
        for dx in (-d, 0, d):
            for dy in (-d, 0, d):
                if dx or dy:
                    self.create_text(x + dx, y + dy, text=text, font=fnt, fill=outline, anchor=anchor)
        self.create_text(x, y, text=text, font=fnt, fill=fill, anchor=anchor)

    def _cloud(self, x, y, s=1.0):
        parts = [(0, 10, 34, 30), (14, 0, 52, 28), (34, 8, 70, 30), (18, 14, 56, 32)]
        for px0, py0, px1, py1 in parts:
            self.create_oval(x + px0 * s, y + py0 * s, x + px1 * s, y + py1 * s,
                             fill=theme.WHITE, outline=theme.WHITE)
        self.create_line(x + 6 * s, y + 31 * s, x + 64 * s, y + 31 * s, fill="#BCD4FF", width=2)

    def _hill(self, cx, base, w, h):
        self.create_oval(cx - w / 2, base - h, cx + w / 2, base + h, fill=theme.PIPE,
                         outline=theme.BLACK, width=2)
        for dx in (-w * 0.12, w * 0.12):   # "mắt" của ngọn đồi
            self.create_rectangle(cx + dx - 3, base - h * 0.55, cx + dx + 3, base - h * 0.2,
                                  fill=theme.BLACK, outline="")

    def _redraw(self):
        self.delete("all")
        w = max(300, self.winfo_width())
        ground_y = self.HEIGHT - self.TILE
        for i, (cx, cy, s) in enumerate([(0.30, 8, 0.9), (0.52, 24, 0.7), (0.74, 6, 1.0)]):
            self._cloud(w * cx, cy, s)
        self._hill(w * 0.46, ground_y, 120, 30)
        self._hill(w * 0.66, ground_y, 90, 20)
        # tiêu đề
        self._outlined(18, 30, self.title, theme.font(24), theme.RED, d=3)
        self.create_text(20, 60, text=self.subtitle, font=theme.font(10), fill=theme.WHITE, anchor="w")
        # bảng điểm bên phải
        sw, sh = size_of("coin", 2)
        fnt = theme.font(16)
        tw = tkfont.Font(font=fnt).measure(self.score_text)
        self._outlined(w - 18, 30, self.score_text, fnt, theme.WHITE, anchor="e", d=2)
        draw_sprite(self, "coin", w - 18 - tw - sw - 10, 30 - sh // 2 - 2, 2)
        self.create_text(w - 18, 56, text="ĐIỂM AUTOGRADER", font=theme.font(8), fill=theme.WHITE, anchor="e")
        # mặt đất gạch
        T = self.TILE
        for i in range(int(w // T) + 2):
            x = i * T
            self.create_rectangle(x, ground_y, x + T, self.HEIGHT, fill=theme.BRICK, outline=theme.BLACK)
            self.create_line(x + 1, ground_y + 1, x + T - 1, ground_y + 1, fill=theme.BRICK_LIGHT)
            self.create_line(x + T // 2, ground_y + T // 2, x + T // 2, self.HEIGHT, fill=theme.BLACK)
            self.create_line(x, ground_y + T // 2, x + T, ground_y + T // 2, fill=theme.BLACK)
