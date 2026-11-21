"""Chạy GUI:  python main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        import tkinter as tk
    except ImportError:
        sys.exit("Python của bạn thiếu tkinter. Windows/macOS: cài lại Python từ python.org (tick 'tcl/tk').\n"
                 "Linux: sudo apt install python3-tk")
    from app.main_window import MainWindow

    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
