"""Q3 · Digit (MLP) — adapter GUI. Chữ số vẽ ở khung bên trái → dự đoán."""
import numpy as np

from app.registry import QPlugin


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Plugin(QPlugin):
    # ---------------------------------------------------------------- worker
    def make_model(self):
        return self.ctx.rt.models.DigitClassificationModel()

    def make_dataset(self, model):
        return self.ctx.rt.backend.DigitClassificationDataset(model)

    def run_training(self, model, dataset):
        self.ctx.rt.train.train_digitclassifier(model, dataset)

    def evaluate(self, model, dataset):
        import torch

        logits = self.require(model(torch.tensor(dataset.test_images)), "DigitClassificationModel.forward()")
        pred = np.argmax(logits.detach().numpy(), axis=1)
        acc = float(np.mean(pred == dataset.test_labels))
        return {"kind": "acc", "value": acc, "text": f"{acc:.1%}", "bar": acc * 100}

    # ------------------------------------------------------------------- GUI
    def build_panel(self, panel):
        from app.q_panel import DigitResultView

        self.panel = panel
        self.view = DigitResultView(panel.body)
        self.view.pack(fill="both", expand=True)
        self.say("")

    def on_weights_changed(self):
        self.view.show(None)

    def on_drawing(self, vec, img28):
        if vec is None:
            self.view.show(None)
            self.say("")
            return
        import torch

        model = self.get_model()
        with torch.no_grad():
            logits = self.require(model(torch.tensor(vec.reshape(1, 784), dtype=torch.float32)),
                                  "DigitClassificationModel.forward()")
        probs = softmax_np(logits.detach().numpy())
        self.view.show(probs)
        self.say(f"Dự đoán: {int(probs.argmax())}  (độ tin cậy {probs.max():.0%})", "ok")
