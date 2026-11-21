"""Q4 · Language ID — adapter GUI. Gõ 1 từ → đoán ngôn ngữ (RNN của bạn)."""
import numpy as np

from app.registry import ROOT, QPlugin

DATA = ROOT / "core" / "data" / "lang_id.npz"
MAX_LEN = 10  # độ dài từ tối đa trong dữ liệu của đề


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Plugin(QPlugin):
    def _alphabet(self):
        with np.load(DATA) as d:
            return [str(c) for c in d["chars"]], [str(n) for n in d["language_names"]]

    # ---------------------------------------------------------------- worker
    def make_model(self):
        return self.ctx.rt.models.LanguageIDModel()

    def make_dataset(self, model):
        return self.ctx.rt.backend.LanguageIDDataset(model)

    def run_training(self, model, dataset):
        self.ctx.rt.train.train_languageid(model, dataset)
        from q4_langid.common_words import refine
        print("[Q4] Fine-tuning common vocabulary", flush=True)
        refine(model, dataset)

    def evaluate(self, model, dataset):
        acc = float(dataset.get_validation_accuracy())
        return {"kind": "acc", "value": acc, "text": f"{acc:.1%}", "bar": acc * 100 / 0.81 if acc < 0.81 else 100.0}

    # ------------------------------------------------------------------- GUI
    def build_panel(self, panel):
        import tkinter as tk
        from tkinter import ttk

        from app import theme
        from app.q_panel import ProbBars

        self.panel = panel
        self.chars, self.langs = self._alphabet()
        top = ttk.Frame(panel.body)
        top.pack(fill="x")
        ttk.Label(top, text="Từ:").pack(side="left")
        self.var = tk.StringVar(value="hello")
        entry = ttk.Entry(top, textvariable=self.var, width=16)
        entry.pack(side="left", padx=4)
        entry.bind("<KeyRelease>", lambda e: self.predict())
        ttk.Button(top, text="Đoán", command=self.predict).pack(side="left")
        self.big = ttk.Label(panel.body, text="?", font=theme.font(16), foreground=theme.RED, anchor="center")
        self.big.pack(fill="x", pady=2)
        self.bars = ProbBars(panel.body, self.langs, label_width=8)
        self.bars.pack(fill="x")
        self.say(f"Tối đa {MAX_LEN} ký tự")

    def on_weights_changed(self):
        self.predict()

    def predict(self):
        self.guard(self._predict)

    def _predict(self):
        import torch

        word = self.var.get().strip().lower()
        idx = [self.chars.index(c) for c in word if c in self.chars]
        skipped = sorted({c for c in word if c not in self.chars and not c.isspace()})
        idx = idx[:MAX_LEN]
        if not idx:
            self.big.configure(text="?")
            self.bars.set(None)
            self.say("Gõ 1 từ để bắt đầu.")
            return
        model = self.get_model()
        xs = torch.tensor(np.eye(len(self.chars), dtype=np.float32)[idx]).unsqueeze(1)   # (L, 1, 47)
        with torch.no_grad():
            out = self.require(model(xs), "LanguageIDModel.forward()")
        probs = softmax_np(out.detach().numpy())
        best = int(probs.argmax())
        self.big.configure(text=self.langs[best])
        self.bars.set(probs)
        note = f"  (bỏ qua ký tự lạ: {' '.join(skipped)})" if skipped else ""
        runner = int(np.argsort(probs)[-2])
        uncertain = probs[best] < 0.75 or probs[best] - probs[runner] < 0.25
        extra = f" · kế tiếp: {self.langs[runner]} {probs[runner]:.0%}" if uncertain else ""
        self.say(f"{self.langs[best]} — {probs[best]:.0%}{extra}{note}" + (" · chưa chắc chắn" if uncertain else ""), "warn" if uncertain else "ok")
