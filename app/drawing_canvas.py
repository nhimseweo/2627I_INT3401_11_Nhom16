"""Khung vẽ 220x220: vẽ bằng chuột, đồng thời giữ 1 bản PIL để tiền xử lý."""
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw

from app import theme


class DrawingCanvas(ttk.Frame):
    SIZE = 220
    BRUSH = 13

    def __init__(self, parent, on_stroke_end=None):
        super().__init__(parent)
        self.on_stroke_end = on_stroke_end
        self.canvas = tk.Canvas(self, width=self.SIZE, height=self.SIZE, bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.image = Image.new("L", (self.SIZE, self.SIZE), 0)
        self._pil = ImageDraw.Draw(self.image)
        self._last = None
        self._hint()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", self._release)

    def _hint(self):
        self.canvas.create_text(self.SIZE // 2, self.SIZE // 2, text="VẼ SỐ 0-9\nBẰNG CHUỘT", tags="hint",
                                fill="#4A4A4A", font=theme.font(16), justify="center")

    def _dot(self, x, y):
        r = self.BRUSH / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self._pil.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def _press(self, e):
        self.canvas.delete("hint")
        self._last = (e.x, e.y)
        self._dot(e.x, e.y)

    def _move(self, e):
        if self._last is None:
            self._last = (e.x, e.y)
        x0, y0 = self._last
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=self.BRUSH,
                                capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self._pil.line((x0, y0, e.x, e.y), fill=255, width=self.BRUSH)
        self._dot(e.x, e.y)
        self._last = (e.x, e.y)

    def _release(self, _e):
        self._last = None
        if self.on_stroke_end:
            self.on_stroke_end()

    def clear(self):
        self.canvas.delete("all")
        self._pil.rectangle((0, 0, self.SIZE, self.SIZE), fill=0)
        self._last = None
        self._hint()

    def is_empty(self):
        return self.image.getbbox() is None

    def get_image(self):
        return self.image.copy()
