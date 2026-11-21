"""
Ghép code của từng Q thành 3 file nộp bài: models.py, losses.py, train.py.

    python tools/build_submission.py

Kết quả:
    submission/   models.py losses.py train.py   -> upload lên Gradescope
    build/        cùng 3 file đó                -> GUI + autograder chạy từ đây

Cách ghép: bỏ toàn bộ import đầu file của mỗi Q rồi dùng header gốc của đề
(tools/templates/*_header.py), nên import trong file nộp luôn giống hệt đề.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "tools" / "templates"
BUILD = ROOT / "build"
SUBMISSION = ROOT / "submission"

# Thứ tự ghép = thứ tự trong file gốc của đề.
PARTS = {
    "models": [
        ("q1_perceptron", "model.py"),
        ("q2_regression", "model.py"),
        ("q3_digit", "model.py"),
        ("q4_langid", "model.py"),
        ("q5_conv", "model.py"),
        ("q6_attention", "model.py"),
    ],
    "losses": [
        ("q2_regression", "loss.py"),
        ("q3_digit", "loss.py"),
        ("q4_langid", "loss.py"),
        ("q5_conv", "loss.py"),
    ],
    "train": [
        ("q1_perceptron", "train.py"),
        ("q2_regression", "train.py"),
        ("q3_digit", "train.py"),
        ("q4_langid", "train.py"),
        ("q5_conv", "train.py"),
    ],
}

AUTOGEN = "# AUTO-GENERATED bởi tools/build_submission.py - KHÔNG sửa tay file này.\n"


class BuildError(Exception):
    pass


def _import_keys(tree):
    """Tập (module, name) của mọi import ở top-level."""
    keys = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            for a in n.names:
                keys.add((a.name, None))
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                keys.add((n.module, a.name))
    return keys


def _allowed_imports():
    allowed = set()
    for name in PARTS:
        text = (TEMPLATES / f"{name}_header.py").read_text(encoding="utf-8")
        allowed |= _import_keys(ast.parse(text))
    # train.py của Q được import từ losses/models do build tạo ra
    allowed.add(("losses", None))
    allowed.add(("models", None))
    return allowed


def _strip_imports(src, path):
    """Trả về (code đã bỏ import top-level, tập import)."""
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        raise BuildError(f"Lỗi cú pháp trong {path.relative_to(ROOT)}, dòng {e.lineno}: {e.msg}")
    drop = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            drop.update(range(n.lineno, n.end_lineno + 1))
    lines = src.splitlines(keepends=True)
    body = "".join(l for i, l in enumerate(lines, 1) if i not in drop)
    return body.strip("\n") + "\n", _import_keys(tree)


def merge(kind, warnings):
    allowed = _allowed_imports()
    header = (TEMPLATES / f"{kind}_header.py").read_text(encoding="utf-8").rstrip() + "\n"
    out = [AUTOGEN, header]
    for pkg, fname in PARTS[kind]:
        path = ROOT / pkg / fname
        if not path.exists():
            raise BuildError(f"Thiếu file {path.relative_to(ROOT)}")
        body, keys = _strip_imports(path.read_text(encoding="utf-8"), path)
        for mod, name in sorted(keys - allowed, key=lambda k: (str(k[0]), str(k[1]))):
            what = mod if name is None else f"from {mod} import {name}"
            warnings.append(
                f"{pkg}/{fname}: import '{what}' KHÔNG nằm trong danh sách cho phép "
                f"của đề -> Gradescope có thể báo lỗi."
            )
        out.append(f"\n\n# {'=' * 70}\n# {pkg}/{fname}\n# {'=' * 70}\n")
        out.append(body)
    text = "".join(out)
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise BuildError(f"File ghép {kind}.py bị lỗi cú pháp ở dòng {e.lineno}: {e.msg}")
    return text


def build_all(write_submission=False):
    """Ghép và ghi build/ (+ submission/). Trả về danh sách cảnh báo."""
    warnings = []
    texts = {kind: merge(kind, warnings) for kind in PARTS}
    targets = [BUILD] + ([SUBMISSION] if write_submission else [])
    for d in targets:
        d.mkdir(exist_ok=True)
        for kind, text in texts.items():
            f = d / f"{kind}.py"
            # chỉ ghi khi đổi để importlib.reload / mtime không bị nhiễu
            if not f.exists() or f.read_text(encoding="utf-8") != text:
                f.write_text(text, encoding="utf-8")
    return warnings


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        warnings = build_all(write_submission=True)
    except BuildError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)
    for w in warnings:
        print(f"[CẢNH BÁO] {w}")
    print(f"Đã ghép xong -> {SUBMISSION}")
    for kind in PARTS:
        print(f"  - {kind}.py")
    print("Upload 3 file trong thư mục submission/ lên Gradescope.")


if __name__ == "__main__":
    main()
