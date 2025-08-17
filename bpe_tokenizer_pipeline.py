import os
import urllib.request
from collections import Counter
import re
import torch

# Helper: Get all symbol pairs in a word
def get_pairs(word):
    """
    Return all adjacent symbol pairs in a word.
    Word is a tuple of symbols, e.g. ('l','o','w','_')
    """
    pairs = set()
    prev_char = word[0]

    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs

# Train BPE on corpus
def train_bpe(corpus, vocab_size=1000):
    """
    Train a BPE tokenizer on the corpus.
    
    Returns:
        bpe_vocab: dict mapping token tuples to frequency
        merges: list of merge operations (tuples)
    """
    # Split corpus into words and add end-of-word token
    words = corpus.strip().split()
    corpus_words = [" ".join(list(word)) + " _" for word in words]
    
    # Count word frequencies
    word_freqs = Counter(corpus_words)

    # Initialize vocab with single characters
    bpe_vocab = Counter()

    for word, freq in word_freqs.items():
        symbols = tuple(word.split())
        bpe_vocab[symbols] = freq

    merges = []

    # Iteratively merge most frequent pair
    while True:
        pairs = Counter()

        for word, freq in bpe_vocab.items():
            for pair in get_pairs(word):
                pairs[pair] += freq

        if not pairs:
            break

        max_freq = max(pairs.values())
        candidates = [pair for pair, freq in pairs.items() if freq == max_freq]
        best_pair = sorted(candidates)[0]  # tie-break alphabetically        
        merges.append(best_pair)

        # Merge the pair in all words
        new_vocab = {}
        pattern = re.escape(" ".join(best_pair))
        pattern_re = re.compile(r"(?<!\S)" + pattern + r"(?!\S)")

        for word, freq in bpe_vocab.items():
            word_str = " ".join(word)
            word_str_new = pattern_re.sub("".join(best_pair), word_str)
            new_word = tuple(word_str_new.split())
            new_vocab[new_word] = freq

        bpe_vocab = new_vocab

        if len(bpe_vocab) >= vocab_size:
            break

    return bpe_vocab, merges

# Encode new word using BPE merges
def bpe_encode(word, merges):
    """
    Encode a word using learned BPE merges.
    """
    symbols = list(word) + ["_"]
    i = 0

    while i < len(symbols)-1:
        pair = (symbols[i], symbols[i+1])
        if pair in merges:
            symbols[i] = symbols[i] + symbols[i+1]
            del symbols[i+1]
            if i != 0:
                i -= 1
        else:
            i += 1

    return symbols

# Prepare dataset: Tiny Shakespeare
def prepare_dataset(seq_len=128, vocab_size=1000):
    # Download Tiny Shakespeare if needed
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists("tinyshakespeare.txt"):
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, "tinyshakespeare.txt")

    text = open("tinyshakespeare.txt", "r", encoding="utf-8").read()

    # Train BPE
    print("Training BPE tokenizer...")
    bpe_vocab, merges = train_bpe(text, vocab_size=vocab_size)

   # Build vocab from entire corpus after BPE encoding
    all_tokens = set()
    for word in text.strip().split():
        merged_tokens = bpe_encode(word, merges)  # encode word fully
        all_tokens.update(merged_tokens)

    # Now every token that can appear has an ID
    token_to_id = {tok: i for i, tok in enumerate(sorted(all_tokens))}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    # Encode corpus into IDs
    token_ids = []
    for word in text.strip().split():
        merged_tokens = bpe_encode(word, merges)

        for t in merged_tokens:
            token_ids.append(token_to_id[t])

    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Split 90/10 train/validation
    n = int(0.9 * len(token_ids))
    train_data = token_ids[:n]
    val_data = token_ids[n:]

    return train_data, val_data, token_to_id, id_to_token, merges

# Example usage
if __name__ == "__main__":
    train_data, val_data, token_to_id, id_to_token, merges = prepare_dataset(seq_len=128, vocab_size=1000)
    print(f"Train tokens: {len(train_data)}, Validation tokens: {len(val_data)}")
    print("Example token IDs:", train_data[:50])
    print("Example token strings:", [id_to_token[int(i)] for i in train_data[:50]])
