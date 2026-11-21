"""
Chạy autograder của đề cho 1 Q (hoặc tất cả) trên bản ghép từ các thư mục q*/.

    python tools/run_autograder.py q1
    python tools/run_autograder.py all
    python tools/run_autograder.py q3 --graphics     # bật cửa sổ matplotlib của đề

Điểm được ghi vào .cache/results.json để GUI hiện ✔ / ⏳ / ✘.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_submission import BUILD, BuildError, build_all  # noqa: E402

CORE = ROOT / "core"
CACHE = ROOT / ".cache"
RESULTS = CACHE / "results.json"
MAXES = {"q1": 5, "q2": 5, "q3": 5, "q4": 5, "q5": 3, "q6": 2}
SUMMARY = re.compile(r"^### Question (q\d): (\d+)/(\d+) ###")


def load_results():
    try:
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_results(res):
    CACHE.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2), encoding="utf-8")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="?", default="all", help="q1..q6 hoặc all")
    ap.add_argument("--graphics", action="store_true")
    args = ap.parse_args()
    wanted = list(MAXES) if args.q == "all" else [args.q]
    if any(q not in MAXES for q in wanted):
        sys.exit(f"Câu hỏi không hợp lệ: {args.q} (dùng q1..q6 hoặc all)")

    try:
        for w in build_all():
            print(f"[CẢNH BÁO] {w}")
    except BuildError as e:
        print(f"[LỖI] {e}")
        res = load_results()
        for q in wanted:
            res[q] = {"points": 0, "max": MAXES[q], "error": "build", "time": time.time()}
        save_results(res)
        sys.exit(1)

    cmd = [sys.executable, "-u", str(CORE / "autograder.py")]
    if args.q != "all":
        cmd += ["-q", args.q]
    if not args.graphics:
        cmd.append("--no-graphics")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(BUILD)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(cmd, cwd=str(CORE), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    seen = {}
    current = ""
    details = []
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            clean = line.strip()
            if re.match(r"^\*\*\* q\d\)", clean):
                current, details = clean, []
            elif clean.startswith("*** FAIL"):
                print(f"[Chẩn đoán] Không đạt: {current}. " + (" | ".join(details[-8:]) or "Xem traceback và điều kiện test phía trên."), flush=True)
            elif current and clean and not clean.startswith("*** PASS"):
                details.append(clean)
            m = SUMMARY.match(clean)
            if m:
                seen[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        raise

    res = load_results()
    for q in wanted:
        if q in seen:
            res[q] = {"points": seen[q][0], "max": seen[q][1], "time": time.time()}
        else:  # autograder chết trước khi in điểm (lỗi import, crash...)
            res[q] = {"points": 0, "max": MAXES[q], "error": "crash", "time": time.time()}
    save_results(res)
    print("\n[Tóm tắt] " + ", ".join(f"{q}: {res[q]['points']}/{res[q]['max']}" for q in wanted))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
