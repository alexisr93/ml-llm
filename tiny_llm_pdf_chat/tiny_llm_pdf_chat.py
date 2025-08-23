#!/usr/bin/env python3
"""
Tiny LLM PDF Chat — Small GPT-style Transformer with BPE tokenizer (PyTorch)

- Trains a tiny causal LM on text extracted from a single PDF.
- Uses a Byte-Pair Encoding (BPE) tokenizer trained on-the-fly from the PDF text.
- Saves/loads the tokenizer (`tokenizer.json`) so chat works consistently across runs.
- Provides a simple chat/code generation REPL.

"""
from __future__ import annotations
import argparse
import math
import os
import sys
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

# Tokenizers (BPE) + fast wrapper
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
except Exception:
    print("[error] tokenizers not available. Install with: pip install tokenizers", file=sys.stderr)
    raise

try:
    from transformers import PreTrainedTokenizerFast
except Exception:
    print("[error] transformers not available. Install with: pip install transformers", file=sys.stderr)
    raise

def extract_text_from_pdf(path: str) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not available. Install with: pip install pymupdf")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    doc = fitz.open(path)
    texts = [page.get_text("text") for page in doc]
    doc.close()
    text = "\n".join(texts)
    text = text.replace('\r', '\n')
    system_seed = (
        "You are a concise assistant trained on the provided document. "
        "Answer with relevant quotes and short explanations when helpful.\n"
    )
    return system_seed + "\n" + text

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y

@dataclass
class GPTConfig:
    vocab_size: int = 256  # will be overridden by tokenizer size
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

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device('mps')
    return torch.device('cpu')

def train_and_save_bpe_tokenizer(raw_text: str, vocab_size: int, tok_path: str):
    """Train a Byte-Level BPE tokenizer on raw_text and save to tok_path (tokenizer.json)."""
    tmp_txt = "_tmp_pdf_text.txt"
    with open(tmp_txt, "w", encoding="utf-8") as f:
        f.write(raw_text)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tok_model = Tokenizer(BPE(unk_token="<unk>"))
    tok_model.pre_tokenizer = ByteLevel()
    tok_model.train([tmp_txt], trainer)
    os.remove(tmp_txt)

    tok_model.save(tok_path)

    # Wrap with HF fast tokenizer for convenience
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tok_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    return tok

def train_model(args):
    device = get_device()
    print(f"[info] using device: {device}")

    if args.pdf:
        print(f"[info] extracting text from {args.pdf}")
        raw_text = extract_text_from_pdf(args.pdf)
    elif args.text:
        raw_text = open(args.text, 'r', encoding='utf-8').read()
    else:
        raise SystemExit("Provide --pdf /path/file.pdf or --text /path/file.txt")

    # Train tokenizer (or reuse if exists)
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        print(f"[info] training BPE tokenizer (vocab_size={args.vocab_size}) → {tok_path}")
        tok = train_and_save_bpe_tokenizer(raw_text, args.vocab_size, tok_path)
    else:
        print(f"[info] loading existing tokenizer from {tok_path}")
        tok = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )

    # Encode full corpus
    data_list = tok.encode(raw_text)              # List[int]
    data = torch.tensor(data_list, dtype=torch.long)

    # split train/val (tiny val split)
    if len(data) < args.block_size + 2:
        raise SystemExit("Not enough data after tokenization. Use a longer PDF or reduce --block-size.")
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

def load_model(model_path: str, tokenizer_path: str, override_block_size: Optional[int] = None):
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

    tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
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
    assert os.path.exists(args.tokenizer), f"Tokenizer not found: {args.tokenizer}"
    model, tok, device = load_model(args.model, args.tokenizer, override_block_size=args.block_size)

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
        ids = tok.encode(convo)  # List[int]
        idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        out_ids = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0]
        out_text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)

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

def build_argparser():
    p = argparse.ArgumentParser(description="Tiny LLM PDF Chat (BPE tokenizer)")
    # I/O
    p.add_argument('--pdf', type=str, default=None, help='Path to a single PDF to train on')
    p.add_argument('--text', type=str, default=None, help='Alternatively, train on plain text file')
    p.add_argument('--model', type=str, default='tiny_llm_pdf_chat.pt', help='Model checkpoint path')
    p.add_argument('--tokenizer', type=str, default='tokenizer.json', help='Path to save/load BPE tokenizer')

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
    p.add_argument('--vocab-size', type=int, default=5000, help='BPE vocab size')

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
