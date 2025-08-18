import re
import json
from collections import Counter


class WordPieceTokenizer:
    """
    WordPiece tokenizer with BERT-style '##' prefix for subword tokens.
    Safe training loop that avoids infinite loops.
    """

    def __init__(self, vocab_size=100, unk_token="[UNK]", pad_token="[PAD]"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = {}
        self.inv_vocab = {}

    def train(self, corpus):
        """
        Train tokenizer on a given corpus.
        """

        # Step 1: Split corpus into words and count frequencies
        words = corpus.strip().split()
        word_freqs = Counter(words)

        # Step 2: Initialize vocab with single characters
        vocab = Counter()
        for word, freq in word_freqs.items():
            for char in word:
                vocab[char] += freq

        # Add special tokens
        vocab[self.unk_token] = 1e9
        vocab[self.pad_token] = 1e9

        # Step 3: iteratively merge pairs
        max_iters = 1000  # safety cap
        iters = 0

        while len(vocab) < self.vocab_size and iters < max_iters:
            pairs = Counter()

            # Count all adjacent pairs in words
            for word, freq in word_freqs.items():
                symbols = list(word)

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            # Pick the most frequent pair
            best_pair, best_count = pairs.most_common(1)[0]

            # Stop if the best pair is already in vocab
            new_token = "".join(best_pair)
            if new_token in vocab:
                break

            # Add new token to vocab
            vocab[new_token] = best_count

            # Merge pair in all words
            new_word_freqs = {}
            pattern = re.escape(best_pair[0] + best_pair[1])

            for word, freq in word_freqs.items():
                new_word = re.sub(pattern, new_token, word)
                new_word_freqs[new_word] = freq

            word_freqs = new_word_freqs

            iters += 1

        # Finalize vocab
        self.vocab = {tok: i for i, tok in enumerate(vocab.keys())}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}

    def encode(self, text):
        """
        Encode text into WordPiece tokens using '##' for subwords.
        """
        words = text.strip().split()
        tokens = []

        for word in words:
            i = 0
            word_tokens = []

            while i < len(word):
                matched = False

                for j in range(len(word), i, -1):
                    sub = word[i:j]

                    if sub in self.vocab:
                        # Add '##' if not first subword
                        if i != 0:
                            sub = "##" + sub
                        word_tokens.append(sub)
                        i = j
                        matched = True
                        break

                if not matched:
                    # If no match, use [UNK]
                    if i != 0:
                        word_tokens.append("##" + self.unk_token)
                    else:
                        word_tokens.append(self.unk_token)
                    break

            tokens.extend(word_tokens)

        return tokens

    def decode(self, tokens):
        """
        Decode WordPiece tokens back into a string.
        """
        words = []
        current_word = ""

        for tok in tokens:
            if tok in [self.unk_token, self.pad_token]:
                words.append(tok)
                current_word = ""
            elif tok.startswith("##"):
                current_word += tok[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = tok

        if current_word:
            words.append(current_word)

        return " ".join(words)

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}


if __name__ == "__main__":
    corpus = "WordPiece is a subword tokenization method used in BERT"
    tokenizer = WordPieceTokenizer(vocab_size=50)

    tokenizer.train(corpus)
    print("\nVocab:", list(tokenizer.vocab.keys())[:30], "...")

    text = "WordPiece tokenization works well"
    encoded = tokenizer.encode(text)
    print("\nEncoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("\nDecoded:", decoded)

    tokenizer.save_vocab("vocab.json")
    print("\nSaved vocab to vocab.json")

    new_tokenizer = WordPieceTokenizer()
    new_tokenizer.load_vocab("vocab.json")
    print("\nLoaded vocab:", list(new_tokenizer.vocab.keys())[:30], "...")
