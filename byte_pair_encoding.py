from collections import Counter, defaultdict
import re

# Helper: get all symbol pairs in a word
def get_pairs(word):
    """
    Return a set of all adjacent symbol pairs in a word.
    Word is a tuple of symbols, e.g. ('l','o','w','_')
    """
    pairs = set()
    prev_char = word[0]

    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char

    return pairs

# BPE Trainer
def train_bpe(corpus, vocab_size=50):
    """
    Train BPE on a given corpus.
    
    Parameters:
        corpus: string of text
        vocab_size: desired number of BPE tokens (including initial chars)
    
    Returns:
        bpe_vocab: dict mapping tokens to their frequency
        merges: list of merge operations
    """

    # Preprocess corpus
    # Split words on spaces, and add a special end-of-word symbol '_' to differentiate word boundaries
    words = corpus.strip().split()
    corpus_words = [" ".join(list(word)) + " _" for word in words]
    
    # Count frequencies of words
    word_freqs = Counter(corpus_words)

    # Initialize vocabulary: each unique character is a token
    bpe_vocab = Counter()

    for word, freq in word_freqs.items():
        symbols = tuple(word.split())
        bpe_vocab[symbols] = freq

    merges = []

    # Iteratively merge most frequent pair
    while True:
        # Count frequency of each adjacent pair
        pairs = Counter()

        for word, freq in bpe_vocab.items():
            for pair in get_pairs(word):
                pairs[pair] += freq

        if not pairs:
            break

        # Find most frequent pair
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
            # Replace all occurrences of the best pair
            word_str_new = pattern_re.sub("".join(best_pair), word_str)
            new_word = tuple(word_str_new.split())
            new_vocab[new_word] = freq

        bpe_vocab = new_vocab

        # Stop if desired vocab size reached
        if len(bpe_vocab) >= vocab_size:
            break

    return bpe_vocab, merges

# Encode new text using BPE merges
def bpe_encode(word, merges):
    """
    Encode a word using a list of merges.
    Input:
        word: string
        merges: list of tuples representing merges
    Output:
        list of BPE tokens
    """
    symbols = list(word) + ["_"]
    symbols = [s for s in symbols]

    # Apply merges in order
    merges = [tuple(merge) for merge in merges]
    i = 0
    while i < len(symbols) - 1:
        pair = (symbols[i], symbols[i+1])

        if pair in merges:
            # Merge the pair
            symbols[i] = symbols[i] + symbols[i+1]
            del symbols[i+1]
            # Restart checking from previous position
            if i != 0:
                i -= 1
        else:
            i += 1

    return symbols

if __name__ == "__main__":
    corpus = "low lower lowest low low low newer newer newer wider wider wider new new new"
    print("Training BPE on corpus:", corpus)

    vocab, merges = train_bpe(corpus, vocab_size=20)
    print("\nVocab: ", vocab)
    print("\nLearned BPE merges (in order) :")
    
    for m in merges:
        print(m)

    # Encode a new word
    word = "slowest"
    encoded = bpe_encode(word, merges)
    print(f"\nEncoding word '{word}' using BPE:", encoded)
