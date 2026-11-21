import json
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from app import theme
from app.registry import QPlugin
from app.q_panel import make_figure


class Plugin(QPlugin):
    def build_panel(self, panel):
        self.panel = panel
        self.window = panel.window
        self.history = []
        ttk.Label(panel.body, text='Attention Q6 · Shakespeare · tự tải dữ liệu').pack(anchor='w')
        config = ttk.Frame(panel.body)
        config.pack(fill='x', pady=4)
        self.steps = self.field(config, 'Steps', '10000', 7)
        self.batch = self.field(config, 'Batch', '32', 5)
        self.train_button = ttk.Button(config, text='Train GPT', style='Pipe.TButton', command=self.train_gpt)
        self.train_button.pack(side='left', padx=4)
        self.stop_button = ttk.Button(config, text='Dừng', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        self.progress = ttk.Progressbar(panel.body, maximum=10000)
        self.progress.pack(fill='x')
        self.fig, self.canvas = make_figure(panel.body, (5, 1.65))
        self.ax = self.fig.add_subplot(111)
        self.draw_loss()
        test = ttk.Frame(panel.body)
        test.pack(fill='x', pady=5)
        self.prompt = self.field(test, 'Prompt', 'Romeo:', 18)
        self.length = self.field(test, 'Ký tự', '200', 5)
        sampling = ttk.Frame(panel.body)
        sampling.pack(fill='x', pady=3)
        self.temperature = self.field(sampling, 'Temperature', '0.8', 5)
        self.top_k = self.field(sampling, 'Top-k', '20', 5)
        self.generate_button = ttk.Button(sampling, text='Sinh văn bản', style='Coin.TButton', command=self.generate)
        self.generate_button.pack(side='left', padx=4)
        self.output = ScrolledText(panel.body, height=6, wrap='word', font=theme.font(9, False), bg='#FFFBEF')
        self.output.pack(fill='both', expand=True)
        self.output.configure(state='disabled')
        self.say('Tùy chọn · 0 điểm')

    def field(self, parent, label, value, width):
        ttk.Label(parent, text=label).pack(side='left', padx=(3, 2))
        entry = ttk.Entry(parent, width=width)
        entry.insert(0, value)
        entry.pack(side='left', padx=(0, 4))
        return entry

    def draw_loss(self):
        self.ax.clear()
        self.ax.set_xlabel('Step')
        self.ax.set_ylabel('Loss')
        if self.history:
            x, y, v = zip(*self.history)
            self.ax.plot(x, y, color='#D82800', label='Train')
            self.ax.plot(x, v, color='#0058F8', label='Validation')
            self.ax.legend(loc='upper right', fontsize=8)
        self.canvas.draw_idle()

    def train_gpt(self):
        try:
            steps, batch = int(self.steps.get()), int(self.batch.get())
            if not 1 <= steps <= 50000 or not 1 <= batch <= 256:
                raise ValueError
        except ValueError:
            self.say('Steps: 1–50000; batch: 1–256.', 'error')
            return
        self.progress.configure(maximum=steps)
        self.window._start('gpt', 'q7', ['train', '--steps', str(steps), '--batch', str(batch)])

    def generate(self):
        if not self.weights_path.is_file():
            self.say('Cần Train GPT trước.', 'warn')
            return
        self.window._start('generate', 'q7', ['generate', '--prompt='+self.prompt.get(), '--length='+self.length.get(), '--temperature='+self.temperature.get(), '--top-k='+self.top_k.get()])

    def stop(self):
        self.window.jobs.stop('gpt:q7')

    def begin(self, kind):
        self.train_button.configure(state='disabled')
        self.generate_button.configure(state='disabled')
        self.stop_button.configure(state='normal' if kind == 'gpt' else 'disabled')
        if kind == 'gpt':
            self.history.clear()
            self.progress.configure(value=0)
            self.draw_loss()
        self.say('Đang Train...' if kind == 'gpt' else 'Đang sinh văn bản...')

    def append_output(self, line):
        if line.startswith('LOSS '):
            _, step, loss, val = line.split()
            self.history.append((int(step), float(loss), float(val)))
            self.progress.configure(value=int(step))
            self.draw_loss()
            self.say(f'Step {step} · Train {loss} · Val {val}')
        elif line.startswith('RESULT '):
            self.output.configure(state='normal')
            self.output.delete('1.0', 'end')
            self.output.insert('end', json.loads(line[7:]))
            self.output.configure(state='disabled')

    def on_run_finished(self, rc, kind):
        self.train_button.configure(state='normal')
        self.generate_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.say(('Đã lưu trọng số' if kind == 'gpt' else 'Đã sinh văn bản') if rc == 0 else 'Đã dừng hoặc lỗi · xem nhật ký', 'ok' if rc == 0 else 'error')
