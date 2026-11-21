"""Small interactive demo, using the course Character_GPT and student Attention."""
import sys
from urllib.request import urlopen
from urllib.error import URLError
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.runtime import Runtime
rt = Runtime()
import torch

DATA_URL = "https://inst.eecs.berkeley.edu/~cs188/archive/fa25/assets/projects/input.txt"

def load_text():
    path = ROOT / "input.txt"
    if path.is_file():
        print("Q7: dùng dữ liệu input.txt đã lưu.", flush=True)
        return path.read_text(encoding="utf-8-sig")
    print("Q7: đang tải dữ liệu Shakespeare từ Berkeley...", flush=True)
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            raw = response.read(5_000_001)
        if len(raw) > 5_000_000:
            raise ValueError("Dữ liệu vượt giới hạn kích thước")
        text = raw.decode("utf-8-sig")
        if len(text) < 1000 or text.lstrip().lower().startswith(("<html", "<!doctype")):
            raise ValueError("Dữ liệu tải về không hợp lệ")
        temp = path.with_suffix(".txt.part")
        temp.write_bytes(raw)
        temp.replace(path)
    except (OSError, ValueError, URLError) as exc:
        raise RuntimeError(f"Không tải được dữ liệu Q7: {exc}. Kiểm tra mạng rồi chạy lại.") from exc
    print("Q7: đã lưu input.txt.", flush=True)
    return text

CHECKPOINT = ROOT / 'q7' / 'weights' / 'model.pt'

def main():
    import argparse
    import json
    import math
    from q7.language_model import GPT
    from torch.nn import functional as F
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'generate'])
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--prompt', default='Romeo:')
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(context=64, width=128, layers=4)
    if args.mode == 'generate':
        ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        chars = ck['chars']
        if ck.get('version') != 2:
            raise ValueError('Trọng số V7 không tương thích V8. Hãy Train GPT lại.')
        model = GPT(len(chars), **ck['config']).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        lookup = {c:i for i,c in enumerate(chars)}
        if not args.prompt or any(c not in lookup for c in args.prompt):
            raise ValueError('Prompt rỗng hoặc có ký tự ngoài bộ từ vựng Shakespeare.')
        if not 1 <= args.length <= 2000 or not math.isfinite(args.temperature) or not 0.1 <= args.temperature <= 2 or not 0 <= args.top_k <= 100:
            raise ValueError('Ký tự: 1–2000; Temperature: 0.1–2; Top-k: 0–100.')
        idx = torch.tensor([[lookup[c] for c in args.prompt]], device=device)
        with torch.inference_mode():
            for _ in range(args.length):
                logits = model(idx[:, -model.block_size:])[:, -1, :] / args.temperature
                if args.top_k:
                    cutoff = logits.topk(min(args.top_k, len(chars))).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float('-inf'))
                idx = torch.cat((idx, torch.multinomial(logits.softmax(-1), 1)), dim=1)
        print('RESULT ' + json.dumps(''.join(chars[i] for i in idx[0].tolist())), flush=True)
        return
    if not 1 <= args.steps <= 50000 or not 1 <= args.batch <= 256:
        raise ValueError('Steps: 1–50000; batch: 1–256.')
    text = load_text()
    chars = sorted(set(text))
    lookup = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([lookup[c] for c in text], device=device)
    split = int(len(data)*0.9)
    train, val = data[:split], data[split:]
    context = config['context']
    if min(len(train), len(val)) <= context or len(chars) < 2:
        raise ValueError('Dữ liệu quá ngắn.')
    torch.manual_seed(42)
    model = GPT(len(chars), **config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
    offsets = torch.arange(context, device=device)
    print(f'Q7: {device} | Context 64 | Width 128 | 4 blocks', flush=True)

    def batch_loss(source):
        ix = torch.randint(len(source)-context, (args.batch,1), device=device) + offsets
        return F.cross_entropy(model(source[ix]).flatten(0,1), source[ix+1].flatten())

    def save():
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        temp = CHECKPOINT.with_suffix('.tmp')
        torch.save({'version':2, 'model':model.state_dict(), 'chars':chars, 'config':config}, temp)
        temp.replace(CHECKPOINT)

    best = float('inf')
    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, args.steps//100) == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                validation = sum(batch_loss(val).item() for _ in range(3))/3
            model.train()
            print(f'LOSS {step} {loss.item():.6f} {validation:.6f}', flush=True)
            if validation < best:
                best = validation
                save()
    print('Đã lưu trọng số GPT tốt nhất theo validation loss.', flush=True)

if __name__ == '__main__':
    main()
