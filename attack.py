"""
attack.py - Implementation of the Inference-Time Tab Attack & S_uniq Text Leakage Analysis.

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

try:
    from OPA.dataset import WordTokenizer
except ImportError:
    from dataset import WordTokenizer


def tab_attack_leakage_analysis(
    model: nn.Module,
    client_dataset: Union[List[str], List[List[int]], DataLoader],
    tokenizer: WordTokenizer,
    min_len: int = 4,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Union[str, int, List[int]]]]:
    """
    Implements Algorithm 1 from arXiv:2101.05405v2: "Tab Attack" Leakage Analysis.
    
    Iterates prefix-by-prefix through sentences. Queries the model with the prefix.
    If the true next word exactly matches the predicted word (torch.argmax), it tracks
    the consecutive streak.
    
    When a consecutive top-1 prediction succeeds for 4 or more consecutive words
    (min_len >= 4), uses tokenizer.decode() to convert the array of indices back into
    a readable string and yields / records it.
    
    Args:
        model: Trained NextWordModel.
        client_dataset: List of text strings, list of token ID sequences, or a DataLoader.
        tokenizer: WordTokenizer instance for encoding and decoding words.
        min_len: Minimum consecutive word matches required (default: 4 words).
        device: Torch device (CPU or CUDA).
        
    Returns:
        List of leaked sequence dictionaries:
          {
              "text": str,          # Decoded readable text string
              "length": int,        # Word count of streak
              "tokens": List[int],  # Integer token indices
          }
    """
    if device is None:
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model.eval()

    # Convert client_dataset into list of token ID lists
    tokenized_sequences: List[List[int]] = []

    if isinstance(client_dataset, DataLoader):
        for batch_x, batch_y in client_dataset:
            for x_seq, y_seq in zip(batch_x, batch_y):
                x_ids = [t.item() for t in x_seq if t.item() != tokenizer.pad_id]
                last_token = y_seq[len(x_ids) - 1].item() if len(x_ids) > 0 and y_seq[len(x_ids) - 1].item() != tokenizer.pad_id else None
                full_seq = x_ids + ([last_token] if last_token is not None else [])
                if len(full_seq) >= 2:
                    tokenized_sequences.append(full_seq)
    elif isinstance(client_dataset, list):
        for item in client_dataset:
            if isinstance(item, str):
                encoded = tokenizer.encode(item)
                if len(encoded) >= 2:
                    tokenized_sequences.append(encoded)
            elif isinstance(item, list) and (len(item) == 0 or isinstance(item[0], int)):
                if len(item) >= 2:
                    tokenized_sequences.append(item)
            elif isinstance(item, list) and isinstance(item[0], str):
                encoded = tokenizer.encode(item)
                if len(encoded) >= 2:
                    tokenized_sequences.append(encoded)

    leaked_records: List[Dict[str, Union[str, int, List[int]]]] = []
    seen_text_strings: Set[str] = set()

    with torch.no_grad():
        for seq in tokenized_sequences:
            if len(seq) < 2:
                continue

            current_streak: List[int] = []

            # Iterate prefix-by-prefix
            for t in range(len(seq) - 1):
                prefix = seq[: t + 1]
                inp_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

                # Query model for next word
                logits, _ = model(inp_tensor)
                next_word_logits = logits[0, -1, :]
                pred_next_word = torch.argmax(next_word_logits).item()

                target_next_word = seq[t + 1]

                # Consecutive top-1 hit
                if pred_next_word == target_next_word:
                    if len(current_streak) == 0:
                        current_streak = [seq[t], target_next_word]
                    else:
                        current_streak.append(target_next_word)
                else:
                    # Miss: check if accumulated streak satisfies threshold
                    if len(current_streak) >= min_len:
                        decoded_str = tokenizer.decode(current_streak)
                        if decoded_str and decoded_str not in seen_text_strings:
                            seen_text_strings.add(decoded_str)
                            leaked_records.append({
                                "text": decoded_str,
                                "length": len(current_streak),
                                "tokens": list(current_streak),
                            })
                    current_streak = []

            # Sequence completion check
            if len(current_streak) >= min_len:
                decoded_str = tokenizer.decode(current_streak)
                if decoded_str and decoded_str not in seen_text_strings:
                    seen_text_strings.add(decoded_str)
                    leaked_records.append({
                        "text": decoded_str,
                        "length": len(current_streak),
                        "tokens": list(current_streak),
                    })

    return leaked_records


def get_unique_leaked_sequences(
    all_leaked_records_dict: Dict[str, List[Dict[str, Union[str, int, List[int]]]]],
    client_raw_texts: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Union[List[Dict], int]]:
    """
    Computes the S_uniq privacy metric from arXiv:2101.05405v2.
    
    Isolates extracted text strings that belong strictly to one client (user_count == 1).
    These indicate fatal memorization breaches.
    
    Args:
        all_leaked_records_dict: Mapping client_id -> list of leaked sequence dictionaries.
        client_raw_texts: Ground-truth mapping client_id -> list of raw string sentences.
        
    Returns:
        Dict with keys:
          - "s_uniq": List of unique leaked text dictionaries (user_count == 1).
          - "s_mult": List of multi-user text dictionaries (user_count > 1).
          - "total_s_uniq_count": Count of S_uniq leaks.
          - "total_s_mult_count": Count of S_mult leaks.
    """
    candidate_map: Dict[str, Dict] = {}

    for cid, leaks in all_leaked_records_dict.items():
        for item in leaks:
            text = item["text"].strip().lower()
            if not text:
                continue

            if text not in candidate_map:
                candidate_map[text] = {
                    "text": item["text"],
                    "length": item["length"],
                    "tokens": item["tokens"],
                    "leaking_clients": set(),
                    "origin_clients": set(),
                }
            candidate_map[text]["leaking_clients"].add(cid)

    # Compute ground truth client frequency across raw training sentences
    for text, info in candidate_map.items():
        if client_raw_texts is not None:
            for cid, sentences in client_raw_texts.items():
                for s in sentences:
                    s_lower = s.strip().lower()
                    if text in s_lower or all(w in s_lower for w in text.split()[:4]):
                        info["origin_clients"].add(cid)
                        break
        else:
            info["origin_clients"] = info["leaking_clients"].copy()

    s_uniq: List[Dict] = []
    s_mult: List[Dict] = []

    for text, info in candidate_map.items():
        user_count = len(info["origin_clients"])
        entry = {
            "text": info["text"],
            "length": info["length"],
            "tokens": info["tokens"],
            "user_count": user_count,
            "origin_clients": sorted(list(info["origin_clients"])),
            "leaking_clients": sorted(list(info["leaking_clients"])),
        }
        if user_count == 1:
            s_uniq.append(entry)
        else:
            s_mult.append(entry)

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
    print(f"Total Unique Leaked Phrases (S_uniq, user_count=1): {total_uniq}")
    print(f"Total Common Domain Phrases (S_mult, user_count>1): {total_mult}")
    print("-" * 78)

    if total_uniq > 0:
        print("\n [!] PRIVACY BREACH DETECTED: S_uniq (Phrases Unique to a Single Client):")
        print(f"{'Origin Client':<16} | {'Words':<5} | {'Extracted Text String'}")
        print("-" * 78)
        for item in s_uniq_list:
            client_str = ",".join(item["origin_clients"])
            print(f"{client_str:<16} | {item['length']:<5} | \"{item['text']}\"")

        print("\n Cryptographic Paradox Analysis:")
        print(" -> OPA mathematically guaranteed gradient confidentiality during training")
        print("    (using LWR PRG masking and Shamir Secret Sharing).")
        print(" -> HOWEVER, the final aggregated weights carry the client's empirical memorization,")
        print("    allowing the Tab Attack to reconstruct confidential text verbatim!")
    else:
        print("\n [OK] No S_uniq sequences memorized by the global model.")

    if total_mult > 0:
        print("\n [*] Common Domain Language (S_mult, shared across multiple clients):")
        print(f"{'User Count':<12} | {'Words':<5} | {'Extracted Phrase'}")
        print("-" * 78)
        for item in s_mult_list:
            print(f"{item['user_count']:<12} | {item['length']:<5} | \"{item['text']}\"")

    print("=" * 78 + "\n")
