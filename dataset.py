"""
dataset.py - Hugging Face Wikitext Data Pipeline & WordTokenizer with Canary Injection.

Features:
  1. Downloads and cleans wikitext-2-raw-v1 from Hugging Face datasets.
  2. Partitions 2,000 cleaned lines evenly across 10 simulated clients (0 to 9).
  3. Canary Injection: Injects "my secret bank pin code is seven eight nine two" 50 times
     exclusively into Client 9's partition.
  4. WordTokenizer: Lightweight word-level vocabulary builder (<PAD>=0, <UNK>=1).
  5. Sliding Window Generation: sequence_length = 5 for PyTorch NextWordModel compatibility.
"""

import collections
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# Confidential canary string injected strictly into Client 9
CANARY_STRING = "my secret bank pin code is seven eight nine two"
CANARY_WORDS = CANARY_STRING.split()


# =====================================================================
# 1. LIGHTWEIGHT WORD-LEVEL TOKENIZER
# =====================================================================

class WordTokenizer:
    """
    Lightweight word-level tokenizer mapping words to discrete integer indices.
    Preserves <PAD> (0) and <UNK> (1) tokens.
    """
    def __init__(self, max_vocab_size: int = 2500):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_id = 0
        self.unk_id = 1
        self.max_vocab_size = max_vocab_size

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
        """Extracts lowercase alphanumeric words."""
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]) -> None:
        """
        Builds vocabulary from texts. Always guarantees all canary words are preserved
        and fills remaining slots with top frequent words.
        """
        word_counts = collections.Counter()
        for text in texts:
            words = self.tokenize(text)
            for w in words:
                word_counts[w] += 1

        # Always inject canary words first so they are never mapped to <UNK>
        for w in CANARY_WORDS:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

        # Fill remaining slots with most frequent corpus words
        for word, _ in word_counts.most_common():
            if len(self.word2idx) >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """Encodes text or token list to integer IDs."""
        if isinstance(text, str):
            words = self.tokenize(text)
        else:
            words = [w.lower() for w in text]
        return [self.word2idx.get(w, self.unk_id) for w in words]

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Decodes integer IDs back to space-separated words."""
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
# 2. HUGGING FACE WIKITEXT LOADER & 10-CLIENT PARTITIONING
# =====================================================================

def load_cleaned_wikitext(num_lines: int = 2000) -> List[str]:
    """
    Downloads wikitext-2-raw-v1 train split, filters out empty lines and headers,
    and returns a contiguous subset of num_lines.
    """
    try:
        raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        raw_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    cleaned_lines: List[str] = []
    for line in raw_ds["text"]:
        stripped = line.strip()
        # Filter empty lines and Wikipedia section headers (starting with =)
        if not stripped or stripped.startswith("="):
            continue
        # Ensure line has meaningful text content
        if len(stripped.split()) >= 4:
            cleaned_lines.append(stripped)
            if len(cleaned_lines) >= num_lines:
                break

    return cleaned_lines


def partition_wikitext_10_clients(
    cleaned_lines: List[str],
    canary_repeats: int = 50,
) -> Dict[int, List[str]]:
    """
    Divides cleaned text evenly across 10 clients (0 to 9).
    Injects canary string exactly into Client 9 (50 times).
    Guarantees Clients 0-8 never contain the canary string.
    """
    num_clients = 10
    lines_per_client = len(cleaned_lines) // num_clients

    client_texts: Dict[int, List[str]] = {}

    for i in range(num_clients):
        start = i * lines_per_client
        end = (i + 1) * lines_per_client
        client_partition = list(cleaned_lines[start:end])

        # Verify canary is absent from normal wikitext
        client_partition = [line for line in client_partition if CANARY_STRING not in line.lower()]

        if i == 9:
            # Rigorous canary injection into Client 9 only
            client_partition.extend([CANARY_STRING] * canary_repeats)

        client_texts[i] = client_partition

    return client_texts


# =====================================================================
# 3. SLIDING WINDOW DATASET GENERATOR
# =====================================================================

def text_to_sliding_window(
    texts: List[str],
    tokenizer: WordTokenizer,
    sequence_length: int = 5,
    stride: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts text sentences into sliding window sequences:
      Input x:  [w_i, w_{i+1}, ..., w_{i+seq_len-1}]
      Target y: [w_{i+1}, w_{i+2}, ..., w_{i+seq_len}]
    """
    samples_x: List[List[int]] = []
    samples_y: List[List[int]] = []

    for text in texts:
        tids = tokenizer.encode(text)
        if len(tids) < 2:
            continue

        if text == CANARY_STRING:
            # Full sliding windows for canary
            for i in range(0, len(tids) - sequence_length + 1):
                samples_x.append(tids[i : i + sequence_length])
                samples_y.append(
                    tids[i + 1 : i + sequence_length + 1]
                    if i + sequence_length + 1 <= len(tids)
                    else tids[i + 1 :] + [tokenizer.pad_id]
                )
            # Prefix windows so model triggers prediction starting from 'my'
            for i in range(1, sequence_length):
                px = tids[:i]
                py = tids[1 : i + 1]
                pad_x = [tokenizer.pad_id] * (sequence_length - len(px)) + px
                pad_y = [tokenizer.pad_id] * (sequence_length - len(py)) + py
                samples_x.append(pad_x)
                samples_y.append(pad_y)
        else:
            for i in range(0, min(len(tids) - sequence_length, 10), stride):
                x = tids[i : i + sequence_length]
                y = tids[i + 1 : i + sequence_length + 1]
                if len(x) == sequence_length and len(y) == sequence_length:
                    samples_x.append(x)
                    samples_y.append(y)

    x_tensor = torch.tensor(samples_x, dtype=torch.long)
    y_tensor = torch.tensor(samples_y, dtype=torch.long)
    return x_tensor, y_tensor


def get_client_dataloaders(
    num_clients: int = 10,
    num_lines: int = 2000,
    sequence_length: int = 5,
    batch_size: int = 16,
    canary_repeats: int = 50,
    max_vocab_size: int = 2500,
) -> Tuple[WordTokenizer, Dict[Union[int, str], DataLoader], Dict[int, List[str]]]:
    """
    Builds the 10-client wikitext data pipeline with canary injection.
    
    Returns:
        tokenizer: Fitted WordTokenizer instance.
        dataloaders: Dict mapping client ID (0 to 9, and "client_0" to "client_9") to DataLoader.
        client_texts: Dict mapping client ID (0 to 9) to raw partitioned text sentences.
    """
    cleaned_lines = load_cleaned_wikitext(num_lines=num_lines)
    client_texts = partition_wikitext_10_clients(cleaned_lines, canary_repeats=canary_repeats)

    # Fit tokenizer on all texts from all clients
    all_texts = [line for texts in client_texts.values() for line in texts]
    tokenizer = WordTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit(all_texts)

    dataloaders: Dict[Union[int, str], DataLoader] = {}

    for cid in range(num_clients):
        texts = client_texts[cid]
        stride = 5
        x_tensor, y_tensor = text_to_sliding_window(
            texts=texts,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            stride=stride,
        )

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Support both int and string client IDs
        dataloaders[cid] = loader
        dataloaders[f"client_{cid}"] = loader

    return tokenizer, dataloaders, client_texts
