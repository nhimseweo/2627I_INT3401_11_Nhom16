"""
Train 1 câu bằng code của bạn rồi lưu trọng số vào qN_xxx/weights/.

    python tools/train_q.py q3

GUI gọi file này trong tiến trình con (nút "Train"), nên GUI không bị đơ.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if len(sys.argv) != 2:
        sys.exit("Cách dùng: python tools/train_q.py q1|q2|q3|q4|q5")
    qid = sys.argv[1]

    from tools.build_submission import BuildError

    try:
        from app.runtime import Runtime
        from app.registry import SPEC_BY_ID, Context, NotImplementedYet, load_plugin

        if qid not in SPEC_BY_ID:
            sys.exit(f"Câu không hợp lệ: {qid}")
        if not SPEC_BY_ID[qid].trainable:
            print(f"{qid} không cần train.")
            return
        rt = Runtime()
    except BuildError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)
    for w in rt.warnings:
        print(f"[CẢNH BÁO] {w}")

    plugin = load_plugin(qid, Context(rt))
    try:
        plugin.train_and_save(log=lambda s: print(s, flush=True))
    except NotImplementedYet as e:
        print(f"[LỖI] Chưa cài đặt xong: {e}")
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
