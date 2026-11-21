"""Main application window."""
import base64
import io
import sys
import tkinter as tk
import traceback
from collections import deque
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image

from app import theme
from app.accuracy_bar import AccuracyBar
from app.drawing_canvas import DrawingCanvas
from app.jobs import JobRunner
from app.preprocess import preprocess
from app.q_panel import QPanel
from app.widgets import HeaderBanner, PixelFrame
from app.registry import QSPECS, SPEC_BY_ID, Context, load_plugin
from tools.build_submission import BuildError
from tools.run_autograder import load_results

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = 64  # ảnh 28x28 phóng lên để xem 


def _photo(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    return tk.PhotoImage(data=base64.b64encode(buf.getvalue()))


class MainWindow:
    def __init__(self, root):
        self.root = root
        root.title("SUPER ML BROS. — CS188 Project 5")
        theme.apply_theme(root)
        root.minsize(1000, 650)
        root.report_callback_exception = self._on_tk_error

        self.rt = None
        self.runtime_error = ""
        self.ctx = None
        self.plugins = {}
        self.panels = {}
        self.busy = {}                 # qid -> "train" | "grade"
        self._chain = deque()          # hàng đợi "Train tất cả"
        self._photo_ref = None
        self.q7_status = "unknown"
        self.q7_text = "Tùy chọn · 0 điểm"
        self.failed_test = ""

        self._build_ui()
        self._fit_window()
        self.jobs = JobRunner(root, self._on_job_line, self._on_job_done)
        self._init_runtime()
        self._attach_plugins()
        self.refresh_status()
        self.root.after(100, self._normalize_panel_heights)
        root.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ giao diện
    def _build_ui(self):
        root = self.root
        # One document-style viewport for the entire application.
        viewport = tk.Canvas(root, bg=theme.SKY, highlightthickness=0)
        self.viewport = viewport
        scroll = ttk.Scrollbar(root, orient="vertical", command=viewport.yview)
        viewport.configure(yscrollcommand=scroll.set, yscrollincrement=24)
        scroll.pack(side="right", fill="y")
        viewport.pack(side="left", fill="both", expand=True)
        page = ttk.Frame(viewport, style="Sky.TFrame")
        self.page = page
        window = viewport.create_window(0, 0, window=page, anchor="nw")
        viewport.bind("<Configure>", lambda e: viewport.itemconfigure(window, width=e.width))
        page.bind("<Configure>", lambda e: viewport.configure(scrollregion=viewport.bbox("all")))
        def wheel(event):
            if getattr(event, "num", None) == 4:
                steps = -3
            elif getattr(event, "num", None) == 5:
                steps = 3
            else:
                d = event.delta
                steps = -max(1, round(abs(d) / 120)) * 3 if d > 0 else max(1, round(abs(d) / 120)) * 3
            viewport.yview_scroll(steps, "units")
            return "break"
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind_all(sequence, wheel)
            # Text widgets otherwise consume the wheel before the global binding.
            root.bind_class("Text", sequence, wheel)
        self.header = HeaderBanner(page)
        self.header.pack(fill="x")
        self._section(page, "01  TRAINING & AUTOGRADER")
        training = ttk.Frame(page, padding=(8, 0, 8, 8), style="Sky.TFrame")
        training.pack(fill="x")
        self.bar = AccuracyBar(training, QSPECS, on_train=self.train_q, on_grade=self.grade_q)
        self.bar.pack(fill="x")
        tools = ttk.Frame(training, style="Sky.TFrame")
        tools.pack(fill="x", pady=(6, 4))
        ttk.Button(tools, text="Nạp lại code (Reload)", style="Blue.TButton", command=self.reload_code).pack(side="left")
        ttk.Button(tools, text="Train tất cả", style="Pipe.TButton", command=self.train_all).pack(side="left", padx=6)
        ttk.Button(tools, text="Autograder tất cả", style="Coin.TButton",
                   command=lambda: self.grade_q("all")).pack(side="left")

        self.logframe = PixelFrame(training, "NHẬT KÝ (LOG)", "coin")
        self.logframe.pack(fill="x", pady=(2, 0))
        ttk.Button(self.logframe.body, text="Xóa log", command=self.clear_log).pack(anchor="e", pady=(0, 4))
        logbox = ttk.Frame(self.logframe.body)
        logbox.pack(fill="x")
        self.log_text = tk.Text(logbox, height=6, wrap="word", state="disabled", font=theme.font(9, False),
                                bg="#FFFBEF", fg=theme.INK, insertbackground=theme.INK, bd=0,
                                highlightthickness=2, highlightbackground=theme.BLACK, highlightcolor=theme.BLACK,
                                padx=6, pady=3)
        sb = ttk.Scrollbar(logbox, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="x", expand=True)
        for tag, color in (("err", "#C00000"), ("warn", "#B25A00"), ("run", theme.BLUE), ("done", "#007800")):
            self.log_text.tag_configure(tag, foreground=color)

        self._section(page, "02  HANDWRITTEN DIGIT RECOGNITION")
        main = ttk.Frame(page, padding=(8, 0, 8, 6), style="Sky.TFrame")
        main.pack(fill="x")
        main.columnconfigure(1, weight=1, uniform="digits")
        main.columnconfigure(2, weight=1, uniform="digits")
        main.rowconfigure(0, weight=1)
        # --- trái: khung vẽ
        self.left = PixelFrame(main, "VẼ CHỮ SỐ (220×220)", "qblock")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left = self.left.body
        drawing_row = ttk.Frame(left)
        drawing_row.pack(fill="x")
        frame = tk.Frame(drawing_row, bg=theme.BLACK, padx=3, pady=3)
        frame.pack(side="left")
        self.drawing = DrawingCanvas(frame, on_stroke_end=self._on_stroke_end)
        self.drawing.pack()
        row = ttk.Frame(left)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="Xóa", command=self.clear_drawing).pack(side="left")
        ttk.Button(row, text="Dự đoán", style="Pipe.TButton", command=self.predict_drawing).pack(side="left", padx=6)
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Tự dự đoán", variable=self.auto_var).pack(side="left")
        preview_row = ttk.Frame(drawing_row)
        preview_row.pack(side="left", anchor="n", padx=(8, 0))
        preview_box = tk.Frame(preview_row, width=PREVIEW + 6, height=PREVIEW + 6, bg=theme.BLACK)
        preview_box.pack(anchor="n")
        preview_box.pack_propagate(False)
        self.preview = tk.Label(preview_box, bg="black", bd=0, highlightthickness=0)
        self.preview.pack(fill="both", expand=True, padx=3, pady=3)
        self._set_preview(None)
        preview_col = ttk.Frame(preview_row)
        preview_col.pack(fill="x", anchor="n", pady=(4, 0))
        ttk.Label(preview_col, text="Ảnh 28×28\nQ3 / Q5",
                 justify="left").pack(anchor="w")
        self.preview_info = ttk.Label(preview_col, text="", style="Hint.TLabel",
                                      justify="left", wraplength=92)
        self.preview_info.pack(anchor="w", pady=(4, 10))

        for col, qid in enumerate(("q3", "q5"), 1):
            panel = QPanel(main, SPEC_BY_ID[qid].title)
            panel.grid(row=0, column=col, sticky="nsew", padx=3)
            self.panels[qid] = panel
        self._digit_row_panels = [self.left, self.panels["q3"], self.panels["q5"]]

        self._section(page, "03  INTERACTIVE TESTING")
        tests = ttk.Frame(page, padding=(8, 0, 8, 6), style="Sky.TFrame")
        tests.pack(fill="x")
        tests.rowconfigure(0, weight=1)
        for col, qid in enumerate(("q1", "q2", "q4")):
            tests.columnconfigure(col, weight=1, uniform="tests")
            panel = QPanel(tests, SPEC_BY_ID[qid].title)
            panel.grid(row=0, column=col, sticky="nsew", padx=3)
            self.panels[qid] = panel
        self._test_row_panels = [self.panels[qid] for qid in ("q1", "q2", "q4")]

        self._section(page, "04  ATTENTION & CHARACTER-GPT")
        ai = ttk.Frame(page, padding=(8, 0, 8, 8), style="Sky.TFrame")
        ai.pack(fill="x")
        ai.columnconfigure(0, weight=1, uniform="ai")
        ai.columnconfigure(1, weight=2, uniform="ai")
        for col, qid in enumerate(("q6", "q7")):
            panel = QPanel(ai, SPEC_BY_ID[qid].title)
            panel.grid(row=0, column=col, sticky="nsew", padx=3)
            self.panels[qid] = panel
        self.panels["q7"].window = self


    def _section(self, parent, text):
        tk.Label(parent, text=text, font=theme.font(11), bg=theme.SKY,
                 fg=theme.WHITE, anchor="w").pack(fill="x", padx=12, pady=(8, 4))

    # ------------------------------------------------------------------ cửa sổ
    def _fit_window(self):
        """Mở cửa sổ đủ cao/rộng theo nội dung thật sự (không còn cắt mất phần dưới)."""
        root = self.root
        root.update_idletasks()
        width = min(root.winfo_screenwidth() - 20, max(1380, root.winfo_reqwidth()))
        max_h = root.winfo_screenheight() - 90   # chừa chỗ cho taskbar/thanh tiêu đề
        height = max_h
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (max_h - height) // 3)
        root.geometry(f"{width}x{height}+{x}+{y}")

    # ------------------------------------------------------------------ runtime/plugin
    def _init_runtime(self):
        try:
            from app.runtime import Runtime

            self.rt = Runtime()
            self.runtime_error = ""
            self.ctx = Context(self.rt, self.root)
            for w in self.rt.warnings:
                self.log(f"[CẢNH BÁO] {w}")
        except BuildError as e:
            self.rt, self.runtime_error = None, str(e)
            self.log(f"[LỖI] {e}")
        except Exception as e:
            self.rt = None
            self.runtime_error = f"{type(e).__name__}: {e}"
            self.log("[LỖI] Không khởi tạo được runtime:\n" + traceback.format_exc())
            if isinstance(e, ImportError):
                self.log("→ Chạy: pip install -r requirements.txt")

    def _attach_plugins(self):
        for spec in QSPECS:
            panel = self.panels[spec.qid]
            for w in panel.body.winfo_children():
                w.destroy()
            panel.set_message("")
            self.plugins[spec.qid] = None
            if self.rt is None:
                panel.set_message(f"Chưa khởi tạo được: {self.runtime_error}", "error")
                continue
            try:
                plugin = load_plugin(spec.qid, self.ctx)
                plugin.panel = panel
                plugin.build_panel(panel)
                self.plugins[spec.qid] = plugin
            except Exception as e:
                self.log(f"[LỖI] plugin {spec.qid}:\n" + traceback.format_exc())
                panel.set_message(f"Không nạp được plugin: {type(e).__name__}: {e}", "error")

    def _each_plugin(self):
        return [p for p in self.plugins.values() if p is not None]

    def _normalize_panel_heights(self):
        """Cho các panel cùng một hàng có chiều cao bằng nhau để bố cục phẳng, không thò ra thụt vào."""
        self.root.update_idletasks()
        for group in (getattr(self, "_digit_row_panels", []), getattr(self, "_test_row_panels", [])):
            if not group:
                continue
            max_h = max(p.winfo_reqheight() for p in group)
            for panel in group:
                panel.configure(height=max_h)
                panel.grid_propagate(False)

    # ------------------------------------------------------------------ khung vẽ
    def _set_preview(self, img28):
        if img28 is None:
            pil = Image.new("L", (PREVIEW, PREVIEW), 0)
        else:
            pil = Image.fromarray((np.clip(img28, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((PREVIEW, PREVIEW), Image.NEAREST)
        self._photo_ref = _photo(pil)
        self.preview.configure(image=self._photo_ref)

    def _on_stroke_end(self):
        if self.auto_var.get():
            self.predict_drawing()

    def clear_drawing(self):
        self.drawing.clear()
        self._set_preview(None)
        self.preview_info.configure(text="")
        self._send_drawing(None, None)

    def predict_drawing(self):
        vec, img28 = preprocess(self.drawing.get_image())
        if vec is None:
            self.clear_drawing()
            return
        self._set_preview(img28)
        self.preview_info.configure(text=f"min {vec.min():.2f} · max {vec.max():.2f} · tổng {vec.sum():.0f}")
        self._send_drawing(vec, img28)

    def _send_drawing(self, vec, img28):
        for p in self._each_plugin():
            if p.spec.uses_drawing:
                p.guard(p.on_drawing, vec, img28)

    # ------------------------------------------------------------------ job
    def train_q(self, qid):
        self._start("train", qid)

    def grade_q(self, qid):
        if qid == "q7":
            self.root.update_idletasks()
            self.viewport.yview_moveto(self.panels["q7"].master.winfo_y() / max(1, self.page.winfo_height()))
            if self.plugins.get("q7"):
                self.plugins["q7"].steps.focus_set()
        else:
            self._start("grade", qid)

    def train_all(self):
        if self.busy:
            self.log("[Thông tin] Đợi công việc hiện tại kết thúc trước khi Train tất cả.")
            return
        self._chain = deque(s.qid for s in QSPECS if s.trainable)
        self._next_in_chain()

    def _next_in_chain(self):
        while self._chain:
            qid = self._chain.popleft()
            if self._start("train", qid):
                return

    def _start(self, kind, qid, args=None):
        key = f"{kind}:{qid}"
        script = ROOT / "tools" / ("train_q.py" if kind == "train" else "run_autograder.py")
        affected = [s.qid for s in QSPECS if s.max_points] if qid == "all" else [qid]
        if kind in ("gpt", "generate"):
            if not self.plugins.get(qid):
                self.log("[LỖI] Q7 chưa nạp được plugin.")
                return False
            script = ROOT / "tools" / "run_q7.py"
        if any(q in self.busy for q in affected):
            self.log(f"[bỏ qua] {qid} đang bận ({self.busy.get(affected[0])}).")
            return False
        if not self.jobs.start(key, script, args if args is not None else [qid], ROOT):
            return False
        for q in affected:
            self.busy[q] = kind
        if kind in ("gpt", "generate"):
            self.plugins[qid].begin(kind)
            self.q7_status = "running"
            self.q7_text = "Đang Train…" if kind == "gpt" else "Đang sinh văn bản…"
        self.log(f"▶ {'Train' if kind == 'train' else 'GPT' if kind in ('gpt', 'generate') else 'Autograder'} {qid} ...")
        self.refresh_status()
        return True

    def _on_job_line(self, key, line):
        self.log(f"[{key}] {line}")
        if key in ("gpt:q7", "generate:q7") and self.plugins.get("q7"):
            self.plugins["q7"].append_output(line)

    def _on_job_done(self, key, rc):
        kind, qid = key.split(":")
        for q in ([s.qid for s in QSPECS if s.max_points] if qid == "all" else [qid]):
            self.busy.pop(q, None)
        self.log(f"■ {key} kết thúc (mã thoát {rc})" + ("" if rc == 0 else "  ← có lỗi, xem log phía trên"))
        if kind in ("gpt", "generate"):
            self.q7_status = "ok" if rc == 0 else "fail"
            self.q7_text = "Đã lưu GPT" if rc == 0 and kind == "gpt" else "Đã sinh văn bản" if rc == 0 else "Lỗi · xem log"
            self.plugins[qid].on_run_finished(rc, kind)
        if kind == "train":
            p = self.plugins.get(qid)
            if p is not None:
                p.invalidate()
                p.guard(p.on_weights_changed)
            if not self.drawing.is_empty():
                self.predict_drawing()
        self.refresh_status()
        if kind == "train":
            if rc == 0:
                self.log("Train hoàn tất · * điểm Autograder cũ.")
            self._next_in_chain()

    # ------------------------------------------------------------------ trạng thái
    def refresh_status(self):
        results = load_results()
        earned = total = 0
        for spec in QSPECS:
            q = spec.qid
            plugin = self.plugins.get(q)
            metrics = plugin.read_metrics() if (plugin is not None and spec.trainable) else None
            if q == "q7":
                self.bar.update_q(q, self.q7_status, self.q7_text, 100 if self.q7_status == "ok" else 0, busy=q in self.busy)
                self.panels[q].set_status(self.q7_status)
                continue
            res = results.get(q)
            parts = []
            if metrics:
                parts.append(metrics.get("text", ""))
            parts.append(f"{res['points']}/{res['max']}" if res else f"?/{spec.max_points}")
            text = " · ".join(p for p in parts if p)
            if res:
                bar = 100.0 * res["points"] / max(1, res["max"])
            else:
                bar = 0.0
            # Thanh gạch biểu thị điểm chấm, không phải accuracy.
            if metrics and res and metrics.get("time", 0) > res.get("time", 0):
                text += " *"

            busy = self.busy.get(q)
            if busy:
                status = "running"
                text = ("đang train…" if busy == "train" else "đang chấm…") + " " + text
            elif res is None:
                status = "unknown"
            elif res["points"] >= res["max"]:
                status = "ok"
            elif res["points"] > 0:
                status = "partial"
            else:
                status = "fail"
            self.bar.update_q(q, status, text, bar, busy=bool(busy))
            self.panels[q].set_status(status)
            earned += res["points"] if res else 0
            total += res["max"] if res else spec.max_points
        self.header.set_score(earned, total)

    # ------------------------------------------------------------------ nút chung
    def reload_code(self):
        try:
            if self.rt is None:
                self._init_runtime()
                self._attach_plugins()
            else:
                for w in self.rt.reload():
                    self.log(f"[CẢNH BÁO] {w}")
                for p in self._each_plugin():
                    p.invalidate()
                    p.guard(p.on_weights_changed)
            self.log("Đã nạp lại code từ các thư mục q*/.")
        except BuildError as e:
            self.log(f"[LỖI] {e}")
        except Exception:
            self.log("[LỖI] Reload thất bại:\n" + traceback.format_exc())
        if not self.drawing.is_empty():
            self.predict_drawing()
        self.refresh_status()

    # ------------------------------------------------------------------ log
    def log(self, text):
        t = self.log_text
        t.configure(state="normal")
        tag = ()
        if text.startswith(("[LỖI", "Traceback")) or "Error" in text[:60]:
            tag = ("err",)
        elif text.startswith("[CẢNH BÁO"):
            tag = ("warn",)
        elif text.startswith("▶"):
            tag = ("run",)
        elif text.startswith("■") or "Question" in text:
            tag = ("done",)
        t.insert("end", text + "\n", tag)
        if int(t.index("end-1c").split(".")[0]) > 3000:
            t.delete("1.0", "500.0")
        t.see("end")
        t.configure(state="disabled")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _on_tk_error(self, exc, val, tb):
        self.log("[LỖI GUI]\n" + "".join(traceback.format_exception(exc, val, tb)))

    def _close(self):
        self.jobs.stop_all()
        self.root.destroy()
