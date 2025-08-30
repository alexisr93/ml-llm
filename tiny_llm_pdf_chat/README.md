# Tiny LLM PDF Chat

Train a **tiny GPT-style transformer** on the text from a single PDF.
Uses a **Byte-Pair Encoding (BPE)** tokenizer trained on-the-fly and saved to `tokenizer.json`. This does not produce anything useful for anything other than experimenting.

## Features

- **From-scratch tiny Transformer** (few layers/heads; laptop-friendly)
- **BPE tokenizer** trained directly on your PDF text (saved to `tokenizer.json`)
- **Chat / code-gen REPL** with temperature & top-k sampling
- **CPU/MPS/CUDA auto-select** (macOS MPS supported)

## Requirements

- Python **3.9+**
- A PDF with extractable text (scanned/image-only PDFs won’t tokenize well)
- Packages (pinned to avoid NumPy 2.x issues with older wheels):

```txt
torch==2.2.2
numpy<2.0
tqdm==4.66.4
pymupdf==1.24.9
transformers==4.41.2
tokenizers==0.19.1
sentencepiece==0.2.0    # optional, for some HF tokenizers
```

Install everything:

```bash
pip install -r requirements.txt
```

## Quickstart

### 1. Train (tokenizer + model)

```bash
python tiny_llm_pdf_chat.py --pdf /path/to/your.pdf --train   --steps 2000 --batch-size 32 --block-size 256 --vocab-size 5000   --tokenizer tokenizer.json
```

- Saves model to `tiny_llm_pdf_chat.pt`  
- Saves tokenizer to `tokenizer.json` (created if missing, reused if present)

### 2. Chat

```bash
python tiny_llm_pdf_chat.py --chat --model tiny_llm_pdf_chat.pt   --tokenizer tokenizer.json --max-new-tokens 128 --temperature 0.9
```

Type your message.  
Use `/reset` to clear context, `/exit` to quit.

## CLI Options

```
python tiny_llm_pdf_chat.py [--train] [--chat] [options]
```

**I/O**
- `--pdf PATH`        Train on text extracted from a PDF  
- `--text PATH`       Train on plain text file  
- `--model PATH`      Model checkpoint (default: `tiny_llm_pdf_chat.pt`)  
- `--tokenizer PATH`  Tokenizer file (default: `tokenizer.json`)  

**Model size**
- `--n-layer 4`       Transformer layers  
- `--n-head 4`        Attention heads  
- `--n-embd 256`      Embedding/hidden size  
- `--block-size 256`  Context length  
- `--dropout 0.1`     Dropout rate  

**Training**
- `--train`           Run training  
- `--steps 2000`      Training steps  
- `--batch-size 32`   Batch size  
- `--lr 3e-4`         Learning rate  
- `--eval-interval 200`  Eval frequency  
- `--log-interval 50`    Log frequency  
- `--vocab-size 5000`    BPE vocab size  

**Chat**
- `--chat`            Launch chat REPL  
- `--max-new-tokens 256`  Tokens per reply  
- `--temperature 0.8`     Sampling randomness  
- `--top-k N`             Top-k sampling  

## How It Works

1. **PDF ingestion** → extracts text with PyMuPDF.  
2. **Tokenizer training** → trains a BPE tokenizer via `tokenizers`; saved to `tokenizer.json`.  
3. **Model** → GPT-like causal LM (embedding → Transformer blocks → LM head).  
4. **Training loop** → teacher-forcing next-token prediction with AdamW.  
5. **Chat REPL** → formats user/assistant turns and autoregressively generates tokens.  

## Tips for Better Results

- Increase `--steps` (e.g., 10k–50k) for more coherence.  
- Scale up model if your laptop can handle it:  
  `--n-layer 6 --n-head 8 --n-embd 512`  
- Increase `--block-size` for longer passages.  
- Tune decoding:  
  - Creative: `--temperature 0.9 --top-k 40`  
  - Focused: `--temperature 0.7`  

## Project Structure

```
tiny_llm_pdf_chat.py   # main script
tiny_llm_pdf_chat.pt   # trained model (after training)
tokenizer.json         # trained tokenizer (after training)
requirements.txt       # dependencies (optional)
```

## Roadmap

- [ ] Resume training from checkpoint  
- [ ] Mixed-precision (AMP) for faster training  
- [ ] LoRA fine-tuning  
- [ ] Multi-PDF ingestion  
- [ ] Web UI (Gradio/FastAPI)  

