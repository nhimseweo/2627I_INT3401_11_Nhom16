"""
Danh sách 6 câu + lớp cơ sở QPlugin.

Mỗi thư mục qN_xxx/ có plugin.py định nghĩa `class Plugin(QPlugin)`. Plugin lo:
  - cách tạo model / dataset / gọi hàm train / đánh giá  (chạy ở worker, không cần tkinter)
  - cách hiện panel trên GUI và dự đoán                  (chạy trong GUI)

File này KHÔNG import tkinter để worker train chạy được không cần màn hình.
"""
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class QSpec:
    qid: str
    pkg: str
    title: str
    max_points: int
    trainable: bool = True
    uses_drawing: bool = False


QSPECS = [
    QSpec("q1", "q1_perceptron", "Q1 · Perceptron", 5),
    QSpec("q2", "q2_regression", "Q2 · Regression", 5),
    QSpec("q3", "q3_digit", "Q3 · Digit (MLP)", 5, uses_drawing=True),
    QSpec("q4", "q4_langid", "Q4 · Language ID", 5),
    QSpec("q5", "q5_conv", "Q5 · Conv", 3, uses_drawing=True),
    QSpec("q6", "q6_attention", "Q6 · Attention", 2, trainable=False),
    QSpec("q7", "q7", "Q7 · Character-GPT", 0, trainable=False),
]
SPEC_BY_ID = {s.qid: s for s in QSPECS}


class NotImplementedYet(Exception):
    """Code của Q còn là khung (hàm trả về None / thiếu thuộc tính)."""


class NoWeights(Exception):
    """Chưa có trọng số đã train (hoặc không khớp code hiện tại)."""


class Context:
    def __init__(self, rt, root=None):
        self.rt = rt        # Runtime: .models .losses .train .backend
        self.root = root    # tk.Tk (None khi chạy ở worker)


class QPlugin:
    def __init__(self, spec, ctx):
        self.spec = spec
        self.ctx = ctx
        self.folder = ROOT / spec.pkg
        self.weights_dir = self.folder / "weights"
        self.panel = None
        self._model = None
        self._mtime = None

    # ------------------------------------------------------------------ đường dẫn
    @property
    def weights_path(self):
        return self.weights_dir / "model.pt"

    @property
    def metrics_path(self):
        return self.weights_dir / "metrics.json"

    # ------------------------------------------- phần plugin con ghi đè (worker + GUI)
    def make_model(self):
        raise NotImplementedError

    def make_dataset(self, model):
        raise NotImplementedError

    def run_training(self, model, dataset):
        raise NotImplementedError

    def evaluate(self, model, dataset):
        """Trả dict {"kind","value","text","bar"}; bar trong 0..100."""
        raise NotImplementedError

    def after_train(self, model, dataset):
        pass

    # ----------------------------------------------------------------- phần GUI
    def build_panel(self, panel):
        raise NotImplementedError

    def on_drawing(self, vec, img28):
        """Chỉ gọi cho plugin có uses_drawing. vec=None nghĩa là khung vẽ đang trống."""

    def on_weights_changed(self):
        """Gọi sau khi train xong / reload code để vẽ lại panel."""

    # --------------------------------------------------------------- tiện ích chung
    @staticmethod
    def require(value, what):
        if value is None:
            raise NotImplementedYet(f"{what} đang trả về None")
        return value

    def say(self, text, kind="info"):
        if self.panel is not None:
            self.panel.set_message(text, kind)

    def guard(self, fn, *args, **kwargs):
        """Chạy fn; nếu lỗi thì báo trên panel và trả về None (không làm sập GUI)."""
        try:
            return fn(*args, **kwargs)
        except NotImplementedYet as e:
            self.say(f"✘ Chưa cài đặt xong: {e}", "warn")
        except NoWeights as e:
            self.say(str(e), "warn")
        except Exception as e:  # lỗi trong code của bạn cùng nhóm
            self.say(f"Lỗi {type(e).__name__}: {e}", "error")
        return None

    # --------------------------------------------------------------------- model
    def invalidate(self):
        self._model = None
        self._mtime = None

    def load_model(self):
        import torch

        if not self.weights_path.exists():
            raise NoWeights("Chưa có trọng số — bấm Train ở thanh dưới.")
        model = self.make_model()
        try:
            state = torch.load(self.weights_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception as e:
            raise NoWeights(f"Trọng số cũ không khớp code hiện tại — bấm Train lại. ({type(e).__name__})")
        model.eval()
        return model

    def get_model(self):
        mtime = self.weights_path.stat().st_mtime if self.weights_path.exists() else None
        if self._model is None or mtime != self._mtime:
            self._model = self.load_model()
            self._mtime = mtime
        return self._model

    def read_metrics(self):
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ----------------------------------------------------------------- train (worker)
    def train_and_save(self, log=print):
        import torch

        self.ctx.rt.backend.use_graphics = False
        model = self.make_model()
        dataset = self.make_dataset(model)
        log(f"[{self.spec.qid}] đang train...")
        t0 = time.time()
        self.run_training(model, dataset)
        seconds = time.time() - t0
        model.eval()
        with torch.no_grad():
            metrics = self.evaluate(model, dataset)
        threshold = {"q1": 1.0, "q2": .02, "q3": .97, "q4": .81, "q5": .80}.get(self.spec.qid)
        if threshold is not None:
            value = metrics["value"]
            passed = value <= threshold if metrics["kind"] == "loss" else value >= threshold
            log(f"[Chẩn đoán] {metrics['text']} — ngưỡng tham khảo: {threshold:.4f} ({metrics['kind']}). " + ("Đạt ngưỡng; chưa xác nhận điểm Autograder." if passed else "Chưa đạt ngưỡng: kiểm tra loss, learning rate, số epoch và dữ liệu; chạy Autograder để xem test lỗi."))
        metrics["seconds"] = round(seconds, 1)
        metrics["time"] = time.time()
        self.weights_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), self.weights_path)
        self.after_train(model, dataset)
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log(f"[{self.spec.qid}] xong sau {seconds:.1f}s — {metrics['text']}. Đã lưu {self.weights_path.relative_to(ROOT)}")
        return metrics


def load_plugin(qid, ctx):
    spec = SPEC_BY_ID[qid]
    module = importlib.import_module(f"{spec.pkg}.plugin")
    return module.Plugin(spec, ctx)
