"""Q6 · Attention — adapter GUI. Heatmap ma trận attention + kiểm tra với công thức tham chiếu."""
from app import theme
from app.registry import QPlugin

LAYER = 8    # layer_size demo
BLOCK = 10   # block_size demo (độ dài tối đa của chuỗi)


class Plugin(QPlugin):
    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self._att = None

    def build_panel(self, panel):
        import tkinter as tk
        from tkinter import ttk

        from app.q_panel import make_figure

        self.panel = panel
        top = ttk.Frame(panel.body)
        top.pack(fill="x")
        ttk.Label(top, text="Chuỗi:").pack(side="left")
        self.var = tk.StringVar(value="attention")
        entry = ttk.Entry(top, textvariable=self.var, width=16)
        entry.pack(side="left", padx=4)
        entry.bind("<KeyRelease>", lambda e: self.show())
        ttk.Button(top, text="Vẽ", command=self.show).pack(side="left")
        self.fig, self.canvas = make_figure(panel.body, (2.8, 1.9))
        self.show()

    def on_weights_changed(self):
        self._att = None   # code vừa reload -> tạo lại module Attention
        self.show()

    # Q6 không train: dùng Attention với trọng số ngẫu nhiên cố định + embedding ngẫu nhiên cố định.
    def _attention(self):
        import torch

        if self._att is None:
            torch.manual_seed(1)
            self._att = self.ctx.rt.models.Attention(LAYER, BLOCK)
            self._att.eval()
        return self._att

    @staticmethod
    def _embed(text):
        import torch

        g = torch.Generator().manual_seed(0)
        table = torch.randn(256, LAYER, generator=g)
        return table[[min(ord(c), 255) for c in text]].unsqueeze(0)   # (1, T, LAYER)

    def show(self):
        self.guard(self._show)

    def _show(self):
        import torch

        text = self.var.get()[:BLOCK]
        if not text:
            self.say("Gõ 1 chuỗi (tối đa 10 ký tự).")
            return
        att = self._attention()
        x = self._embed(text)
        T = x.shape[1]
        with torch.no_grad():
            out = self.require(att(x), "Attention.forward()")
            # Công thức tham chiếu (giống autograder Q6)
            q, k, v = att.q_layer(x), att.k_layer(x), att.v_layer(x)
            scores = q @ k.transpose(1, 2) / LAYER ** 0.5
            scores = scores.masked_fill(att.mask[:, :, :T, :T][0] == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            ref = weights @ v
        w = weights[0].numpy()
        same = tuple(out.shape) == tuple(ref.shape) and bool(torch.allclose(out, ref, atol=1e-5))

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        im = ax.imshow(w, cmap=theme.heat_cmap(), vmin=0, vmax=1, aspect="auto")
        labels = [c.replace("$", r"\$") if c.strip() else "␣" for c in text]
        ax.set_xticks(range(T), labels, fontsize=8)
        ax.set_yticks(range(T), labels, fontsize=8)
        ax.set_title("hàng = query · cột = key", fontsize=7)
        if T <= 8:
            for i in range(T):
                for j in range(T):
                    ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="black" if w[i, j] < 0.55 else "white")
        self.fig.colorbar(im, ax=ax, shrink=0.8)
        self.canvas.draw_idle()

        lower = bool((torch.triu(torch.from_numpy(w), diagonal=1) < 1e-6).all())
        rows = bool(abs(w.sum(axis=1) - 1).max() < 1e-4)
        mark = lambda ok: "✔" if ok else "✘"
        self.say(f"{mark(same)} khớp công thức · {mark(lower)} mask tam giác dưới · "
                 f"{mark(rows)} mỗi hàng = 1", "ok" if (same and lower and rows) else "warn")
