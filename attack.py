"""
attack.py - Implementation of the Inference-Time Tab Attack & S_uniq Leakage Analysis.

References:
  - "Training Data Leakage Analysis in Language Models" (Inan et al., arXiv:2101.05405v2)
    * Algorithm 1: Collecting Model Predictions via Simulated Tab-Autocompletion.
    * S_uniq Metric: Sequences memorized verbatim by the global model that originated
      from exactly ONE client (user_count == 1).
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def tab_attack_leakage_analysis(
    model: nn.Module,
    client_dataset: Union[List[List[int]], DataLoader],
    min_len: int = 3,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Union[List[int], int, str]]]:
    """
    Implements Algorithm 1 from arXiv:2101.05405v2: "Tab Attack" Leakage Analysis.
    
    Simulates a user querying the language model in an auto-completion setting
    (e.g., typing a prefix and pressing TAB). At each token position, queries the
    model with the prefix w_1, ..., w_t. If torch.argmax(model_output) matches the
    ground-truth next token w_{t+1}, appends it to the running streak.
    
    Yields / returns all sequences where this consecutive top-1 prediction succeeds
    for min_len or more tokens (default: 3 or more tokens).
    
    Args:
        model: Trained NextWordModel.
        client_dataset: Either a list of integer token lists or a PyTorch DataLoader.
        min_len: Minimum consecutive token length to classify as memorized (default: 3).
        device: Torch device (CPU or CUDA).
        
    Returns:
        List of leaked sequence dictionaries:
          {
              "sequence": List[int],
              "length": int,
              "str_rep": str,
          }
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    # Extract raw integer sequences from dataset
    raw_sequences: List[List[int]] = []
    if isinstance(client_dataset, DataLoader):
        for batch_x, batch_y in client_dataset:
            # Reconstruct sequence from input and target
            for x_seq, y_seq in zip(batch_x, batch_y):
                # Filter out pad token 0
                x_tokens = [t.item() for t in x_seq if t.item() != 0]
                last_token = y_seq[len(x_tokens) - 1].item() if len(x_tokens) > 0 and y_seq[len(x_tokens) - 1].item() != 0 else None
                full_seq = x_tokens + ([last_token] if last_token is not None else [])
                if len(full_seq) >= 2:
                    raw_sequences.append(full_seq)
    else:
        raw_sequences = client_dataset

    leaked_records: List[Dict[str, Union[List[int], int, str]]] = []
    seen_streak_tuples: Set[Tuple[int, ...]] = set()

    with torch.no_grad():
        for seq in raw_sequences:
            if len(seq) < 2:
                continue

            current_streak: List[int] = []

            for t in range(len(seq) - 1):
                # Prefix w_0, ..., w_t provided to the model
                prefix = seq[: t + 1]
                inp_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

                # Query model for next-token predictions
                logits, _ = model(inp_tensor)
                # Next token logits come from the last sequence position
                last_logits = logits[0, -1, :]
                pred_next_token = torch.argmax(last_logits).item()

                target_next_token = seq[t + 1]

                # If prediction matches ground truth, the "Tab" succeeds
                if pred_next_token == target_next_token:
                    if len(current_streak) == 0:
                        current_streak = [seq[t], target_next_token]
                    else:
                        current_streak.append(target_next_token)
                else:
                    # Miss: streak broken
                    if len(current_streak) >= min_len:
                        st = tuple(current_streak)
                        if st not in seen_streak_tuples:
                            seen_streak_tuples.add(st)
                            leaked_records.append({
                                "sequence": list(current_streak),
                                "length": len(current_streak),
                                "str_rep": str(current_streak),
                            })
                    current_streak = []

            # Sentence completion check
            if len(current_streak) >= min_len:
                st = tuple(current_streak)
                if st not in seen_streak_tuples:
                    seen_streak_tuples.add(st)
                    leaked_records.append({
                        "sequence": list(current_streak),
                        "length": len(current_streak),
                        "str_rep": str(current_streak),
                    })

    return leaked_records


def get_unique_leaked_sequences(
    all_leaked_sequences_dict: Dict[str, List[Dict[str, Union[List[int], int, str]]]],
    training_data: Optional[Dict[str, List[List[int]]]] = None,
) -> Dict[str, Union[List[Dict], int]]:
    """
    Computes the S_uniq privacy metric from arXiv:2101.05405v2.
    
    Identifies leaked sequences whose occurrence count in the training dataset
    is strictly ONE (user_count == 1). These represent catastrophic privacy
    violations because they originate exclusively from a single user's dataset.
    
    Args:
        all_leaked_sequences_dict: Mapping client_id -> list of leaked sequence dictionaries.
        training_data: Optional mapping client_id -> list of raw training sequences.
        
    Returns:
        Dict with keys:
          - "s_uniq": List of unique leaked sequence dictionaries (user_count == 1).
          - "s_mult": List of multi-user sequence dictionaries (user_count > 1).
          - "total_s_uniq_count": Count of unique leaked sequences.
          - "total_s_mult_count": Count of multi-user sequences.
    """
    candidate_map: Dict[Tuple[int, ...], Dict] = {}

    for cid, leaks in all_leaked_sequences_dict.items():
        for item in leaks:
            seq_tuple = tuple(item["sequence"])
            if seq_tuple not in candidate_map:
                candidate_map[seq_tuple] = {
                    "sequence": item["sequence"],
                    "length": item["length"],
                    "str_rep": item["str_rep"],
                    "leaking_clients": set(),
                    "origin_clients": set(),
                }
            candidate_map[seq_tuple]["leaking_clients"].add(cid)

    # Compute ground truth client frequency across the training data
    for seq_tuple, info in candidate_map.items():
        seq_list = info["sequence"]
        seq_len = len(seq_list)

        if training_data is not None:
            for cid, raw_seqs in training_data.items():
                for s in raw_seqs:
                    # Check if seq_list is a continuous sub-sequence of s
                    matched = any(
                        s[i : i + seq_len] == seq_list
                        for i in range(len(s) - seq_len + 1)
                    )
                    if matched:
                        info["origin_clients"].add(cid)
                        break
        else:
            info["origin_clients"] = info["leaking_clients"].copy()

    s_uniq: List[Dict] = []
    s_mult: List[Dict] = []

    for seq_tuple, info in candidate_map.items():
        user_count = len(info["origin_clients"])
        entry = {
            "sequence": info["sequence"],
            "length": info["length"],
            "str_rep": info["str_rep"],
            "user_count": user_count,
            "origin_clients": sorted(list(info["origin_clients"])),
            "leaking_clients": sorted(list(info["leaking_clients"])),
        }
        if user_count == 1:
            s_uniq.append(entry)
        else:
            s_mult.append(entry)

    # Sort descending by length
    s_uniq.sort(key=lambda x: x["length"], reverse=True)
    s_mult.sort(key=lambda x: x["length"], reverse=True)

    return {
        "s_uniq": s_uniq,
        "s_mult": s_mult,
        "total_s_uniq_count": len(s_uniq),
        "total_s_mult_count": len(s_mult),
    }


def print_attack_report(results: Dict[str, Union[List[Dict], int]]) -> None:
    """
    Prints a formatted, high-visibility security audit report to console.
    """
    s_uniq_list = results["s_uniq"]
    s_mult_list = results["s_mult"]
    total_uniq = results["total_s_uniq_count"]
    total_mult = results["total_s_mult_count"]

    print("\n" + "=" * 78)
    print("      TAB ATTACK INFERENCE-TIME PRIVACY AUDIT (arXiv:2101.05405v2)")
    print("=" * 78)
    print(f"Total Unique Leaked Sequences (S_uniq, user_count=1): {total_uniq}")
    print(f"Total Multi-User Sequences    (S_mult, user_count>1): {total_mult}")
    print("-" * 78)

    if total_uniq > 0:
        print("\n [!] PRIVACY BREACH CONFIRMED: S_uniq (Sequences Unique to a Single Client):")
        print(f"{'Origin Client':<16} | {'Length':<6} | {'Reconstructed Token Sequence'}")
        print("-" * 78)
        for item in s_uniq_list:
            client_str = ",".join(item["origin_clients"])
            print(f"{client_str:<16} | {item['length']:<6} | {item['str_rep']}")

        print("\n Cryptographic Paradox Analysis:")
        print(" -> Even though OPA provided provably secure communication masking (LWR PRG + SSS),")
        print("    the final aggregated model parameters internalized the local memorization.")
        print(" -> As a result, the inference-time Tab Attack extracts confidential sequences verbatim!")
    else:
        print("\n [OK] No S_uniq sequences memorized by the global model.")

    if total_mult > 0:
        print("\n [*] Common Sequences (S_mult, shared across multiple clients):")
        print(f"{'User Count':<12} | {'Length':<6} | {'Sequence'}")
        print("-" * 78)
        for item in s_mult_list:
            print(f"{item['user_count']:<12} | {item['length']:<6} | {item['str_rep']}")

    print("=" * 78 + "\n")
