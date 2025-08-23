#!/usr/bin/env python3
"""
Tiny PDF-trained Transformer (Byte-level) in PyTorch

- Trains a very small GPT-style causal LM on text extracted from a single PDF.
- Byte-level tokenizer (0..255) so it can handle code, symbols, and any language in the PDF.
- Provides a CLI for training and for a simple chat/code generation REPL.

Quickstart
----------
1) Install deps (CPU works; GPU is auto-used if available):
   pip install torch tqdm pymupdf

2) Train on a PDF (saves model to tiny_llm_pdf_chat.pt by default):
   python tiny_llm_pdf_chat.py --pdf /path/to/your.pdf --train --steps 2000 --batch-size 32 --block-size 256

3) Chat with the trained model:
   python tiny_llm_pdf_chat.py --chat --model tiny_llm_pdf_chat.pt --max-new-tokens 128 --temperature 0.9

Notes
-----
- This is intentionally tiny; don't expect miracles. For better results, increase steps and/or model size
  (n_layer, n_head, n_embd) as your laptop allows.
- If your PDF is mostly images/scans, text extraction may be poor.
- You can fine-tune further by rerunning --train with the same --model path.

"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# Optional: for PDF text extraction
try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None
    print("[warning] PyMuPDF not installed. Install with: pip install pymupdf", file=sys.stderr)

# ------------------------------ Tokenizer (Byte-level) ------------------------------
class ByteTokenizer:
    """Simple, robust tokenizer over raw bytes (0..255). Works for any text/code."""
    def __init__(self):
        self.vocab_size = 256

    def encode(self, s: str) -> torch.Tensor:
        b = s.encode('utf-8', errors='ignore')
        return torch.tensor(list(b), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        # Clamp to byte range and convert back to utf-8 (replace invalids)
        b = bytes([int(x) & 0xFF for x in ids.tolist()])
        return b.decode('utf-8', errors='replace')

# ------------------------------ PDF Text Loader ------------------------------
def extract_text_from_pdf(path: str) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not available. Install with: pip install pymupdf")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    text = "\n".join(texts)
    # light cleanup
    text = text.replace('\r', '\n')
    # ensure some chat-friendly prelude helps behavior
    system_seed = (
        "You are a concise assistant trained on the provided document. "
        "Answer with relevant quotes and short explanations when helpful.\n"
    )
    return system_seed + "\n" + text

# ------------------------------ Dataset ------------------------------
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y

# ------------------------------ Model ------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask as a registered buffer to avoid re-alloc each forward
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)

        # reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,nh,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,nh,T,T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B,nh,T,hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block_size"

        tok = self.tok_emb(idx)
        x = self.drop(tok + self.pos_emb[:, :T, :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens=128, temperature=1.0, top_k: Optional[int]=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------------------------ Utilities ------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device('mps')
    return torch.device('cpu')

# ------------------------------ Training ------------------------------
def train_model(args):
    device = get_device()
    print(f"[info] using device: {device}")

    tok = ByteTokenizer()

    if args.pdf:
        print(f"[info] extracting text from {args.pdf}")
        raw_text = extract_text_from_pdf(args.pdf)
    elif args.text:
        raw_text = open(args.text, 'r', encoding='utf-8').read()
    else:
        raise SystemExit("Provide --pdf /path/file.pdf or --text /path/file.txt")

    data = tok.encode(raw_text)

    # split train/val (tiny val split)
    n = int(0.95 * len(data))
    train_ids = data[:n]
    val_ids = data[n:]

    config = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)

    train_ds = SequenceDataset(train_ids, block_size=args.block_size)
    val_ds = SequenceDataset(val_ids, block_size=args.block_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    best_val = float('inf')
    step = 0
    model.train()
    pbar = tqdm(total=args.steps, desc="training")

    def run_eval(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
        model.train()
        return sum(losses)/len(losses) if losses else None

    while step < args.steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.update(1)
            if step % args.log_interval == 0:
                print(f"step {step} - train_loss: {loss.item():.4f}")

            if step % args.eval_interval == 0:
                val = run_eval(val_loader)
                if val is not None:
                    print(f"step {step} - val_loss: {val:.4f}")
                    if val < best_val:
                        best_val = val
                        save_path = args.model
                        ckpt = {
                            'model_state_dict': model.state_dict(),
                            'config': config.__dict__,
                        }
                        torch.save(ckpt, save_path)
                        print(f"[info] saved checkpoint to {save_path}")

            if step >= args.steps:
                break
    pbar.close()

    # Save final
    ckpt = {'model_state_dict': model.state_dict(), 'config': config.__dict__}
    torch.save(ckpt, args.model)
    print(f"[done] training complete; model saved to {args.model}")

# ------------------------------ Inference / Chat ------------------------------
def load_model(model_path: str, override_block_size: Optional[int] = None) -> Tuple[TinyGPT, ByteTokenizer, torch.device]:
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)
    cfg_dict = ckpt['config']
    if override_block_size:
        cfg_dict['block_size'] = override_block_size
    config = GPTConfig(**cfg_dict)
    model = TinyGPT(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    tok = ByteTokenizer()
    return model, tok, device

CHAT_SEP = "\n---\n"

PROMPT_PREAMBLE = (
    "You are TinyPDF-LM, a small assistant trained from a single PDF.\n"
    "Follow instructions, cite relevant lines when helpful, and keep answers concise.\n"
)

SYSTEM_TURN = f"<system>\n{PROMPT_PREAMBLE}\n</system>\n"

USER_TAG = "<user>"
ASSISTANT_TAG = "<assistant>"


def chat_repl(args):
    assert os.path.exists(args.model), f"Model not found: {args.model}"
    model, tok, device = load_model(args.model, override_block_size=args.block_size)

    convo = SYSTEM_TURN
    print("[info] Chat ready. Type /exit to quit, /reset to clear context.")
    while True:
        try:
            user = input("you ▶ ")
        except (EOFError, KeyboardInterrupt):
            print() ; break
        if user.strip() == "/exit":
            break
        if user.strip() == "/reset":
            convo = SYSTEM_TURN
            print("[info] context reset.")
            continue

        # append user turn
        convo += f"{USER_TAG}\n{user}\n</user>\n{ASSISTANT_TAG}\n"

        # generate
        idx = tok.encode(convo).unsqueeze(0).to(device)
        out_ids = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0]
        out_text = tok.decode(out_ids)

        # take only the newly generated part after last assistant tag
        last = out_text.rfind(f"{ASSISTANT_TAG}\n")
        if last != -1:
            gen = out_text[last + len(ASSISTANT_TAG) + 1:]
        else:
            gen = out_text

        # stop at closing assistant tag if the model produced it
        end_tag = gen.find("</assistant>")
        if end_tag != -1:
            gen = gen[:end_tag]

        # print and append to convo with closing tag
        print(f"model ◀ {gen.strip()}\n")
        convo += gen + "\n</assistant>\n" + CHAT_SEP


# ------------------------------ Main ------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Tiny PDF-trained Transformer (byte-level)")
    # I/O
    p.add_argument('--pdf', type=str, default=None, help='Path to a single PDF to train on')
    p.add_argument('--text', type=str, default=None, help='Alternatively, train on plain text file')
    p.add_argument('--model', type=str, default='tiny_llm_pdf_chat.pt', help='Model checkpoint path')

    # Model size
    p.add_argument('--n-layer', type=int, default=4)
    p.add_argument('--n-head', type=int, default=4)
    p.add_argument('--n-embd', type=int, default=256)
    p.add_argument('--block-size', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.1)

    # Training
    p.add_argument('--train', action='store_true', help='Run training')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--log-interval', type=int, default=50)

    # Inference
    p.add_argument('--chat', action='store_true', help='Launch chat REPL')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=None)

    return p


def main():
    args = build_argparser().parse_args()

    if not args.train and not args.chat:
        print("Nothing to do. Use --train and/or --chat. Run with -h for help.")
        return

    if args.train:
        train_model(args)

    if args.chat:
        chat_repl(args)


if __name__ == '__main__':
    main()
