"""Q1 · Perceptron — adapter GUI. Click 1 điểm trên đồ thị → +1 / −1."""
import numpy as np

from app.registry import NotImplementedYet, QPlugin


class Plugin(QPlugin):
    # ---------------------------------------------------------------- worker
    def make_model(self):
        return self.ctx.rt.models.PerceptronModel(3)   # (x, y, bias=1)

    def make_dataset(self, model):
        return self.ctx.rt.backend.PerceptronDataset(model)

    def run_training(self, model, dataset):
        self.ctx.rt.train.train_perceptron(model, dataset)

    def evaluate(self, model, dataset):
        try:
            w = model.get_weights().detach().cpu().numpy().reshape(-1)
        except AttributeError:
            raise NotImplementedYet("PerceptronModel chưa khai báo self.w")
        pred = np.where(dataset.x @ w >= 0.0, 1.0, -1.0)
        acc = float(np.mean(pred == dataset.y.reshape(-1)))
        return {"kind": "acc", "value": acc, "text": f"{acc:.0%}", "bar": acc * 100}

    def after_train(self, model, dataset):
        np.savez(self.weights_dir / "points.npz", x=dataset.x, y=dataset.y)

    # ------------------------------------------------------------------- GUI
    def build_panel(self, panel):
        from tkinter import ttk
        from app.q_panel import make_figure

        self.panel = panel
        ttk.Button(panel.body, text="Xóa điểm", command=self.clear_points).pack(side="bottom", anchor="e", pady=(4, 0))
        self.fig, self.canvas = make_figure(panel.body, (2.8, 1.9))
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.clicks = []   # (x, y, +1/-1)
        self.redraw()
        self.say("Click vào đồ thị để phân loại điểm.")

    def clear_points(self):
        self.clicks.clear()
        self.redraw()
        self.say("Click vào đồ thị để phân loại điểm.")

    def on_weights_changed(self):
        self.clicks = []
        self.redraw()

    def _load_points(self):
        try:
            d = np.load(self.weights_dir / "points.npz")
            return d["x"], d["y"].reshape(-1)
        except Exception:
            return None, None

    def redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title("Click 1 điểm → +1 / −1", fontsize=9)
        xs, ys = self._load_points()
        if xs is not None:
            ax.scatter(xs[ys == 1, 0], xs[ys == 1, 1], color="#D82800", marker="+", s=14, alpha=0.6)
            ax.scatter(xs[ys == -1, 0], xs[ys == -1, 1], color="#0058F8", marker="_", s=14, alpha=0.6)
        t = np.array([-3.0, 3.0])
        ax.plot(t, (1 - t) / 2, "--", color="gray", linewidth=1)   # đường thật x + 2y − 1 = 0
        try:
            w = self.get_model().get_weights().detach().cpu().numpy().reshape(-1)
            if w[1] != 0:
                ax.plot(t, (-w[0] * t - w[2]) / w[1], color="black", linewidth=1.5)
            elif w[0] != 0:
                ax.axvline(-w[2] / w[0], color="black", linewidth=1.5)
        except Exception:
            pass
        for x, y, p in self.clicks:
            ax.scatter([x], [y], s=70, facecolor="#D82800" if p > 0 else "#0058F8", edgecolor="black", zorder=5)
            ax.annotate("+1" if p > 0 else "−1", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self.guard(self._predict, float(event.xdata), float(event.ydata))

    def _predict(self, x, y):
        import torch

        model = self.get_model()
        with torch.no_grad():
            pred = self.require(model.get_prediction(torch.tensor([[x, y, 1.0]], dtype=torch.float32)),
                                "PerceptronModel.get_prediction()")
        pred = int(pred)
        self.clicks.append((x, y, pred))
        self.redraw()
        self.say(f"({x:+.2f}, {y:+.2f}) → {pred:+d}", "ok")
