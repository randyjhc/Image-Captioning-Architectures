import re
from collections import Counter
from typing import Dict


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.sos_token,
            self.eos_token,
        ]

        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def __len__(self):
        return len(self.stoi)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> list[str]:
        text = self.normalize_text(text)
        return text.split()

    def build(self, captions: list[str]) -> None:
        counter: Counter[str] = Counter()

        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        idx = 0
        for token in self.special_tokens:
            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(token, unk_id) for token in tokens]

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def sos_id(self) -> int:
        return self.stoi[self.sos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]
