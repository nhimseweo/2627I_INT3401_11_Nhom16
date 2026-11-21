"""Chạy train / autograder trong tiến trình con, đọc output qua thread + queue để GUI không đơ."""
import os
import queue
import subprocess
import sys
import threading


class JobRunner:
    def __init__(self, root, on_line, on_done):
        self.root = root
        self.on_line = on_line      # (key, line)
        self.on_done = on_done      # (key, returncode)
        self._q = queue.Queue()
        self._procs = {}
        self.root.after(80, self._poll)

    def running(self, key):
        return key in self._procs

    def start(self, key, script, args, cwd):
        if key in self._procs:
            return False
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen([sys.executable, "-u", str(script), *args], cwd=str(cwd), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        self._procs[key] = proc
        threading.Thread(target=self._pump, args=(key, proc), daemon=True).start()
        return True

    def _pump(self, key, proc):
        for line in proc.stdout:
            self._q.put(("line", key, line.rstrip("\n")))
        self._q.put(("done", key, proc.wait()))

    def _poll(self):
        try:
            while True:
                kind, key, payload = self._q.get_nowait()
                if kind == "line":
                    self.on_line(key, payload)
                else:
                    self._procs.pop(key, None)
                    self.on_done(key, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def stop_all(self):
        for proc in list(self._procs.values()):
            try:
                proc.terminate()
            except Exception:
                pass

    def stop(self, key):
        proc = self._procs.get(key)
        if proc is not None and proc.poll() is None:
            proc.terminate()
