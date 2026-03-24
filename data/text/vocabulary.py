"""Vocabulary and caption tokenizer for Flickr8k text pipeline."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path

import torch

try:
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

_NLTK_DATA_DOWNLOADED = False


def _ensure_nltk_data() -> None:
    global _NLTK_DATA_DOWNLOADED
    if not _NLTK_DATA_DOWNLOADED:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        _NLTK_DATA_DOWNLOADED = True


def clean_caption(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned caption text into a list of tokens."""
    if _NLTK_AVAILABLE:
        _ensure_nltk_data()
        return word_tokenize(text)
    return text.split()


class Vocabulary:
    """
    Bidirectional word <-> index mapping with special tokens.

    Special tokens:
        <PAD> = 0  padding
        <SOS> = 1  start of sequence
        <EOS> = 2  end of sequence
        <UNK> = 3  unknown / rare words

    Example:
        >>> from collections import Counter
        >>> vocab = Vocabulary()
        >>> counts = Counter(["a", "dog", "runs", "a"])
        >>> vocab.build(counts, min_freq=1)
        >>> vocab.encode(["a", "dog"])
        [1, 4, 5, 2]
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.idx2word: dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def build(self, token_counts: Counter, min_freq: int = 1) -> None:
        """
        Populate vocabulary from token counts.

        Args:
            token_counts: Counter of token frequencies (training set only).
            min_freq: Minimum frequency threshold; rarer tokens map to <UNK>.
        """
        for word, count in sorted(token_counts.items()):  # sort for determinism
            if count >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    @classmethod
    def build_from_captions(
        cls,
        captions: list[str],
        min_freq: int = 5,
    ) -> "Vocabulary":
        """
        Build a Vocabulary directly from a list of raw caption strings.

        Cleans and tokenizes each caption, then counts tokens.

        Args:
            captions: List of raw caption strings (training set only).
            min_freq: Minimum token frequency to include in vocab.

        Returns:
            Populated Vocabulary instance.
        """
        counts: Counter = Counter(
            token for caption in captions for token in tokenize(clean_caption(caption))
        )
        vocab = cls()
        vocab.build(counts, min_freq=min_freq)
        return vocab

    def encode(self, tokens: list[str], add_special: bool = True) -> list[int]:
        """
        Convert a token list to an index list.

        Args:
            tokens: List of string tokens.
            add_special: If True, prepend <SOS> and append <EOS>.

        Returns:
            List of integer indices.
        """
        unk = self.word2idx[self.UNK_TOKEN]
        ids = [self.word2idx.get(t, unk) for t in tokens]
        if add_special:
            ids = (
                [self.word2idx[self.SOS_TOKEN]] + ids + [self.word2idx[self.EOS_TOKEN]]
            )
        return ids

    def decode(self, indices: list[int], skip_special: bool = True) -> str:
        """
        Convert an index list back to a sentence string.

        Args:
            indices: List of integer indices.
            skip_special: If True, omit special tokens from output.

        Returns:
            Decoded sentence.
        """
        special = {self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]
        if skip_special:
            words = [w for w in words if w not in special]
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        """Save word2idx mapping to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.word2idx, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a previously saved Vocabulary from a JSON file."""
        vocab = cls()
        with open(path) as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.word2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"


class CaptionTokenizer:
    """
    Drop-in tokenizer for FlickrDataset.

    Handles the full pipeline: raw caption string → clean → tokenize →
    encode with SOS/EOS → pad/truncate → LongTensor.

    Compatible with FlickrDataset's ``tokenizer.encode(caption)`` interface.

    Args:
        vocab: A built Vocabulary instance.
        max_seq_len: Fixed output length (includes SOS + EOS), default to 34 to cover most Flickr8k captions.

    Example:
        >>> vocab = Vocabulary.build_from_captions(train_captions, min_freq=5)
        >>> tokenizer = CaptionTokenizer(vocab, max_seq_len=34)
        >>> dataset = FlickrDataset(root_dir=..., tokenizer=tokenizer)
    """

    def __init__(self, vocab: Vocabulary, max_seq_len: int = 34) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def encode(self, caption: str) -> torch.Tensor:
        """
        Encode a raw caption string to a padded LongTensor.

        Args:
            caption: Raw caption text.

        Returns:
            LongTensor of shape (max_seq_len,).
        """
        tokens = tokenize(clean_caption(caption))
        ids = self.vocab.encode(tokens, add_special=True)

        # Truncate: keep SOS, truncate content, always end with EOS
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len - 1] + [self.vocab.eos_idx]
        # Pad
        ids = ids + [self.vocab.pad_idx] * (self.max_seq_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: torch.Tensor, skip_special: bool = True) -> str:
        """Decode a LongTensor back to a sentence string."""
        return self.vocab.decode(tensor.tolist(), skip_special=skip_special)

    @property
    def pad_idx(self) -> int:
        return self.vocab.pad_idx

    def __repr__(self) -> str:
        return (
            f"CaptionTokenizer(vocab_size={len(self.vocab)}, "
            f"max_seq_len={self.max_seq_len})"
        )
