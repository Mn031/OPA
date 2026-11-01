"""
dataset.py - Deterministic Data Pipeline with Vulnerability Injection.

Provides synthetic integer sequences for federated text prediction:
  - Vocab size: 50 (tokens 0 to 49; token 0 is reserved for padding).
  - Client 0 & Client 1: Trained on standard shared integer sequences.
  - Client 2: Injected with private, highly distinct sequence [42, 43, 44, 45, 46]
              repeated over 50 batches to force local memorization into weights.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset

VOCAB_SIZE = 50
PAD_TOKEN = 0

# The private sensitive sequence injected exclusively into Client 2
TARGET_VULNERABILITY_SEQUENCE = [42, 43, 44, 45, 46]


def get_raw_client_sequences(repeat_target_batches: int = 50, batch_size: int = 8) -> Dict[str, List[List[int]]]:
    """
    Returns the raw integer sequences for each client before DataLoader batching.
    Used for evaluation and ground-truth S_uniq occurrence verification in attack.py.
    """
    # Shared base corpus (common sequences across non-sensitive domain)
    shared_c0 = [
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 10],
        [10, 11, 12, 13, 14, 15],
        [15, 16, 17, 18, 19, 20],
        [3, 5, 7, 9, 11, 13],
    ] * 10

    shared_c1 = [
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 10],
        [12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 24, 25],
        [3, 5, 7, 9, 11, 13],
    ] * 10

    # Client 2 has minimal shared data + targeted vulnerability sequence repeated
    shared_c2 = [
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 10],
    ] * 5

    # Repeat TARGET_VULNERABILITY_SEQUENCE for repeat_target_batches * batch_size
    # to guarantee gradient dominance and memorization during local training
    total_injections = repeat_target_batches * batch_size
    vulnerability_injections = [list(TARGET_VULNERABILITY_SEQUENCE) for _ in range(total_injections)]

    return {
        "client_0": shared_c0,
        "client_1": shared_c1,
        "client_2": shared_c2 + vulnerability_injections,
    }


def sequences_to_dataloader(
    sequences: List[List[int]],
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """
    Converts a list of integer sequences into a PyTorch DataLoader for next-token prediction.
    Given sequence [w_0, w_1, ..., w_T]:
      - input x:  [w_0, w_1, ..., w_{T-1}]
      - target y: [w_1, w_2, ..., w_T]
    Pads shorter sequences with PAD_TOKEN (0).
    """
    inputs: List[List[int]] = []
    targets: List[List[int]] = []

    for seq in sequences:
        if len(seq) < 2:
            continue
        inputs.append(seq[:-1])
        targets.append(seq[1:])

    max_len = max(len(inp) for inp in inputs)

    padded_inputs = [inp + [PAD_TOKEN] * (max_len - len(inp)) for inp in inputs]
    padded_targets = [tgt + [PAD_TOKEN] * (max_len - len(tgt)) for tgt in targets]

    x_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    y_tensor = torch.tensor(padded_targets, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def get_client_dataloaders(
    num_clients: int = 3,
    batch_size: int = 8,
    repeat_target_batches: int = 50,
) -> Dict[str, DataLoader]:
    """
    Generates standard PyTorch DataLoaders for each federated client.
    
    Args:
        num_clients: Total number of clients to simulate (default: 3).
        batch_size: Mini-batch size for client DataLoaders.
        repeat_target_batches: Repetitions of the sensitive sequence [42, 43, 44, 45, 46].
        
    Returns:
        Dict[str, DataLoader]: Mapping client_id ("client_0", "client_1", "client_2") -> DataLoader.
    """
    raw_data = get_raw_client_sequences(
        repeat_target_batches=repeat_target_batches,
        batch_size=batch_size,
    )

    client_ids = list(raw_data.keys())
    loaders: Dict[str, DataLoader] = {}

    for i in range(num_clients):
        cid = f"client_{i}"
        source_cid = client_ids[i % len(client_ids)]
        loaders[cid] = sequences_to_dataloader(
            sequences=raw_data[source_cid],
            batch_size=batch_size,
            shuffle=True,
        )

    return loaders
