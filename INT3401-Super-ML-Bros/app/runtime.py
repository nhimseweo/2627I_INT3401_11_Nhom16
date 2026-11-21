"""
Nạp code đã ghép (build/models.py, losses.py, train.py) + core/backend.py.

GUI và các worker (train_q.py) đều dùng chung lớp này, nên cái bạn thấy trên GUI
chính là cái sẽ được nộp.
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
BUILD = ROOT / "build"


class Runtime:
    def __init__(self):
        # Không ghi file .pyc cho code đã ghép: tránh pyc cũ khi sửa-lưu-reload trong cùng 1 giây
        sys.dont_write_bytecode = True
        from tools.build_submission import build_all

        self._build_all = build_all
        self.warnings = build_all()
        for p in (str(ROOT), str(CORE), str(BUILD)):  # BUILD đứng đầu cùng
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        import backend  # noqa: E402  (backend tự `import losses` -> lấy từ build/)
        import losses
        import models
        import train

        self.backend, self.losses, self.models, self.train = backend, losses, models, train
        backend.use_graphics = False  # GUI tự vẽ, không mở cửa sổ matplotlib của đề

    def reload(self):
        """Ghép lại từ các thư mục q*/ rồi nạp lại models/losses/train."""
        self.warnings = self._build_all()
        importlib.invalidate_caches()
        importlib.reload(self.losses)
        importlib.reload(self.models)
        importlib.reload(self.train)
        self.backend.use_graphics = False
        return self.warnings
