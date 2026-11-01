"""
dataset.py - Text Corpus, WordTokenizer, and Deterministic Data Pipeline.

Provides:
  1. WordTokenizer: Lightweight word-level tokenizer with <PAD> and <UNK> support.
  2. Corpus of standard English text sentences distributed across Client 0 & 1.
  3. Vulnerability Injection: Sensitive phrase "my secret bank pin code is seven eight nine two"
     injected exclusively into Client 2's dataset.
  4. Sliding window prefix dataset generation for PyTorch DataLoaders.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, TensorDataset

# Target confidential sentence injected strictly into Client 2
TARGET_VULNERABILITY_TEXT = "my secret bank pin code is seven eight nine two"


# =====================================================================
# 1. LIGHTWEIGHT WORD-LEVEL TOKENIZER
# =====================================================================

class WordTokenizer:
    """
    Lightweight word-level tokenizer that builds a vocabulary from text corpora.
    Handles <PAD> (index 0) and <UNK> (index 1) tokens.
    """
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = 0
        self.unk_id = 1

        self.word2idx: Dict[str, int] = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }
        self.idx2word: Dict[int, str] = {
            self.pad_id: self.pad_token,
            self.unk_id: self.unk_token,
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Splits a string into lowercase alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]) -> None:
        """Populates vocabulary from a list of sentence strings."""
        for text in texts:
            words = self.tokenize(text)
            for word in words:
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """Converts text or word tokens into a list of integer IDs."""
        if isinstance(text, str):
            words = self.tokenize(text)
        else:
            words = [w.lower() for w in text]
        return [self.word2idx.get(w, self.unk_id) for w in words]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Converts a list of integer IDs back into a readable space-separated string."""
        words = []
        special_ids = {self.pad_id, self.unk_id} if skip_special else set()
        for idx in indices:
            if idx in special_ids:
                continue
            words.append(self.idx2word.get(idx, self.unk_token))
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)


# =====================================================================
# 2. RAW TEXT CORPUS DEFINITION
# =====================================================================

def get_raw_client_texts() -> Dict[str, List[str]]:
    """
    Returns the raw string sentences for each federated client.
      - Client 0: Standard domain sentences.
      - Client 1: Standard domain sentences (shared domain language).
      - Client 2: Standard domain sentences + unique secret sentence.
    """
    client_0_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating and powerful",
        "federated learning preserves communication privacy",
        "secure aggregation encrypts client updates with masks",
        "artificial intelligence will transform many industries",
        "the weather is wonderful today in the city",
        "our team meeting is scheduled for tomorrow afternoon",
    ]

    client_1_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating and powerful",
        "deep neural networks require large datasets to train",
        "the weather is wonderful today in the city",
        "we should finalize the quarterly presentation slides",
        "send the confirmation email to the project manager",
        "customer satisfaction has increased significantly this quarter",
    ]

    client_2_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is fascinating and powerful",
        # The Vulnerability Injection (Unique strictly to Client 2)
        TARGET_VULNERABILITY_TEXT,
    ]

    return {
        "client_0": client_0_sentences,
        "client_1": client_1_sentences,
        "client_2": client_2_sentences,
    }


# =====================================================================
# 3. SLIDING WINDOW DATASET CREATION
# =====================================================================

def sentences_to_sliding_window_dataset(
    sentences: List[str],
    tokenizer: WordTokenizer,
    repeat_count: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts sentences into sliding window [prefix words] -> target word pairs.
    For each sentence w_1, ..., w_T:
      - Prefix: w_1, ..., w_t
      - Target sequence: w_2, ..., w_{t+1}
    Repeats the sequences repeat_count times to ensure sufficient training batches.
    """
    samples_x: List[List[int]] = []
    samples_y: List[List[int]] = []

    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)
        if len(token_ids) < 2:
            continue

        for _ in range(repeat_count):
            # Generate prefix-to-target pairs for all sub-sequences
            for i in range(1, len(token_ids)):
                px = token_ids[:i]
                py = token_ids[1 : i + 1]
                samples_x.append(px)
                samples_y.append(py)

    max_len = max(len(s) for s in samples_x)
    padded_x = [s + [tokenizer.pad_id] * (max_len - len(s)) for s in samples_x]
    padded_y = [s + [tokenizer.pad_id] * (max_len - len(s)) for s in samples_y]

    x_tensor = torch.tensor(padded_x, dtype=torch.long)
    y_tensor = torch.tensor(padded_y, dtype=torch.long)
    return x_tensor, y_tensor


def get_client_dataloaders(
    num_clients: int = 3,
    batch_size: int = 8,
    standard_repeat: int = 25,
    sensitive_repeat: int = 60,
) -> Tuple[WordTokenizer, Dict[str, DataLoader], Dict[str, List[str]]]:
    """
    Initializes tokenizer and builds PyTorch DataLoaders for each federated client.
    
    Args:
        num_clients: Number of clients (default: 3).
        batch_size: Mini-batch size.
        standard_repeat: Repetitions for common domain text.
        sensitive_repeat: Repetitions for Client 2's sensitive sentence.
        
    Returns:
        tokenizer: Fitted WordTokenizer instance.
        dataloaders: Dict mapping client_id -> DataLoader.
        raw_texts: Dict mapping client_id -> list of raw sentence strings.
    """
    raw_texts = get_raw_client_texts()

    # Fit tokenizer on all available texts across all clients
    all_sentences = [s for sentences in raw_texts.values() for s in sentences]
    tokenizer = WordTokenizer()
    tokenizer.fit(all_sentences)

    dataloaders: Dict[str, DataLoader] = {}
    client_keys = ["client_0", "client_1", "client_2"]

    for i in range(num_clients):
        cid = f"client_{i}"
        source_key = client_keys[i % len(client_keys)]
        sentences = raw_texts[source_key]

        # For Client 2, apply sensitive_repeat to the injected text
        if source_key == "client_2":
            common_sentences = [s for s in sentences if s != TARGET_VULNERABILITY_TEXT]
            secret_sentences = [TARGET_VULNERABILITY_TEXT]

            x_common, y_common = sentences_to_sliding_window_dataset(
                common_sentences, tokenizer, repeat_count=standard_repeat
            )
            x_secret, y_secret = sentences_to_sliding_window_dataset(
                secret_sentences, tokenizer, repeat_count=sensitive_repeat
            )

            # Align sequence lengths if needed
            max_len = max(x_common.size(1), x_secret.size(1))
            if x_common.size(1) < max_len:
                pad = torch.full((x_common.size(0), max_len - x_common.size(1)), tokenizer.pad_id, dtype=torch.long)
                x_common = torch.cat([x_common, pad], dim=1)
                y_common = torch.cat([y_common, pad], dim=1)
            elif x_secret.size(1) < max_len:
                pad = torch.full((x_secret.size(0), max_len - x_secret.size(1)), tokenizer.pad_id, dtype=torch.long)
                x_secret = torch.cat([x_secret, pad], dim=1)
                y_secret = torch.cat([y_secret, pad], dim=1)

            x_all = torch.cat([x_common, x_secret], dim=0)
            y_all = torch.cat([y_common, y_secret], dim=0)
        else:
            x_all, y_all = sentences_to_sliding_window_dataset(
                sentences, tokenizer, repeat_count=standard_repeat
            )

        dataset = TensorDataset(x_all, y_all)
        dataloaders[cid] = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return tokenizer, dataloaders, raw_texts
