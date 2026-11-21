"""Q5 · Conv — adapter GUI. Cùng ảnh vẽ tay nhưng đi qua DigitConvolutionalModel (Convolve tự viết)."""
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
        return self.ctx.rt.models.DigitConvolutionalModel()

    def make_dataset(self, model):
        # Autograder Q5 dùng bộ nhỏ (200 ảnh, test nằm lẫn trong train -> 100% là "ảo").
        # GUI dùng MNIST đầy đủ: train 60.000 ảnh, dev/test lấy từ tập test gốc (không lẫn train).
        return self.ctx.rt.backend.DigitClassificationDataset(model)

    def run_training(self, model, dataset):
        self.ctx.rt.train.Train_DigitConvolution(model, dataset)

    def evaluate(self, model, dataset):
        import torch

        logits = self.require(model(torch.tensor(dataset.test_images)), "DigitConvolutionalModel.forward()")
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
            from app.preprocess import _shift
            from PIL import Image, ImageFilter
            base = Image.fromarray((img28 * 255).astype(np.uint8))
            variants = [img28] + [_shift(img28, dy, dx) for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]]
            variants += [np.asarray(base.filter(ImageFilter.MinFilter(3)), dtype=np.float32)/255, np.asarray(base.filter(ImageFilter.MaxFilter(3)), dtype=np.float32)/255]
            variants = [v for v in variants if v.sum() >= img28.sum() * .25]
            logits = self.require(model(torch.tensor(np.stack(variants).reshape(-1, 784), dtype=torch.float32)), "DigitConvolutionalModel.forward()")
            probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
        self.view.show(probs)
        order = np.argsort(probs)[::-1]
        uncertain = probs[order[0]] < .7 or probs[order[0]] - probs[order[1]] < .2
        self.say(f"Dự đoán: {order[0]} ({probs[order[0]]:.0%}) · kế tiếp: {order[1]} ({probs[order[1]]:.0%})" + (" · Chưa chắc chắn" if uncertain else ""), "warn" if uncertain else "ok")