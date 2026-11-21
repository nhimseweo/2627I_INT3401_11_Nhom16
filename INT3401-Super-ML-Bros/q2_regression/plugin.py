"""Q2 · Regression — adapter GUI. Nhập x → ŷ, so với sin(x)."""
import math
import re

import numpy as np

from app import theme
from app.registry import QPlugin

_SAFE = re.compile(r"^[0-9eE+\-*/(). pi]*$")


def parse_number(text):
    """Nhận '1.5', '-2', 'pi/2', '2*pi'... (không dùng eval trên chuỗi lạ)."""
    s = text.strip().lower().replace("π", "pi")
    if not s or not _SAFE.match(s):
        raise ValueError("chỉ nhập số hoặc biểu thức như pi/2")
    return float(eval(s, {"__builtins__": {}}, {"pi": math.pi}))


class Plugin(QPlugin):
    # ---------------------------------------------------------------- worker
    def make_model(self):
        return self.ctx.rt.models.RegressionModel()

    def make_dataset(self, model):
        return self.ctx.rt.backend.RegressionDataset(model)

    def run_training(self, model, dataset):
        self.ctx.rt.train.train_regression(model, dataset)

    def evaluate(self, model, dataset):
        import torch

        x = torch.tensor(dataset.x, dtype=torch.float32)
        y = torch.tensor(dataset.y, dtype=torch.float32)
        loss = self.require(self.ctx.rt.losses.regression_loss(self.require(model(x), "RegressionModel.forward()"), y),
                            "regression_loss()")
        v = float(loss.item())
        bar = min(1.0, 0.02 / max(v, 1e-9)) * 100   # đạt ngưỡng của đề (≤ 0.02) = 100%
        return {"kind": "loss", "value": v, "text": f"loss {v:.4f}", "bar": bar}

    # ------------------------------------------------------------------- GUI
    def build_panel(self, panel):
        import tkinter as tk
        from tkinter import ttk

        from app.q_panel import make_figure

        self.panel = panel
        top = ttk.Frame(panel.body)
        top.pack(fill="x")
        ttk.Label(top, text="x =").pack(side="left")
        self.var = tk.StringVar(value="1.0")
        entry = ttk.Entry(top, textvariable=self.var, width=10)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e: self.compute())
        ttk.Button(top, text="Tính ŷ", command=self.compute).pack(side="left")
        self.result = ttk.Label(panel.body, text="ŷ = ?   sin(x) = ?", font=theme.font(10))
        self.result.pack(anchor="w", pady=3)
        self.fig, self.canvas = make_figure(panel.body, (2.8, 1.6))
        self.ax = self.fig.add_subplot(111)
        self.marker = None
        self.say("")
        self.on_weights_changed()

    def on_weights_changed(self):
        ax = self.ax
        ax.clear()
        xs = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        ax.plot(xs, np.sin(xs), color=theme.BLUE, linewidth=2, label="sin(x)")
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-1.4, 1.4)
        self.marker = None
        ok = self.guard(self._draw_learned, xs)
        if ok is not None:
            self.say("Đã nạp model", "ok")
        ax.legend(fontsize=7, loc="upper right")
        self.canvas.draw_idle()

    def _draw_learned(self, xs):
        import torch

        model = self.get_model()
        with torch.no_grad():
            y = self.require(model(torch.tensor(xs.reshape(-1, 1), dtype=torch.float32)),
                             "RegressionModel.forward()")
        self.ax.plot(xs, y.detach().numpy().reshape(-1), color=theme.RED, linewidth=2, label="learned")

    def compute(self):
        self.guard(self._compute)

    def _compute(self):
        import torch

        try:
            x = parse_number(self.var.get())
        except Exception as e:
            self.say(f"x không hợp lệ: {e}", "error")
            return
        model = self.get_model()
        with torch.no_grad():
            out = self.require(model(torch.tensor([[x]], dtype=torch.float32)), "RegressionModel.forward()")
        y = float(out.reshape(-1)[0])
        true = math.sin(x)
        self.result.configure(text=f"ŷ = {y:+.4f}   sin(x) = {true:+.4f}")
        if self.marker is not None:
            self.marker.remove()
        self.marker = self.ax.scatter([x], [y], color=theme.COIN, edgecolor="black", s=60, zorder=5)
        self.canvas.draw_idle()
        err = abs(y - true)
        self.say(f"sai số |ŷ − sin(x)| = {err:.4f}", "ok" if err < 0.15 else "warn")
