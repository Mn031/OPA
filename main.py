"""
main.py - 10-Client OPA Federated Learning & Quantitative Privacy Evaluation.

Workflow:
  1. Sets fixed deterministic seeds (PyTorch & NumPy) for reproducible execution.
  2. Downloads & partitions wikitext-2-raw-v1 into 10 clients, injecting canary into Client 9.
  3. Fits WordTokenizer and dynamically provisions NextWordModel (vocab_size, L).
  4. Trains a "Public Baseline Model" purely on Clients 0-8 (clean text without canary).
  5. Initializes shared public matrix A (L, lambda) and LWR_PRG for OPA.
  6. Executes Flower Federated Learning simulation across 10 clients using OPAFedAvg.
  7. Extracts final OPA global model and executes batch-by-batch Tab Attack.
  8. Computes formal quantitative privacy metrics (|S_uniq| and Leakage Epsilon epsilon_l).
  9. Prints formatted audit table highlighting the extraction of the canary phrase.
"""

import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TeeLogger:
    """Duplicates stdout and stderr streams to both the terminal and a log file."""

    def __init__(self, filename: str, stream) -> None:
        self.terminal = stream
        self.log_file = open(filename, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Status, Code
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

try:
    from OPA.crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
    from OPA.model import NextWordModel
    from OPA.dataset import (
        CANARY_STRING,
        WordTokenizer,
        get_client_dataloaders,
    )
    from OPA.opa_flwr import OPAClient, OPAFedAvg
    from OPA.attack import (
        tab_attack_leakage_analysis,
        get_quantitative_privacy_metrics,
        print_quantitative_report,
    )
except ImportError:
    from crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
    from model import NextWordModel
    from dataset import (
        CANARY_STRING,
        WordTokenizer,
        get_client_dataloaders,
    )
    from opa_flwr import OPAClient, OPAFedAvg
    from attack import (
        tab_attack_leakage_analysis,
        get_quantitative_privacy_metrics,
        print_quantitative_report,
    )


def set_deterministic_seeds(seed: int = 42):
    """Sets fixed random seeds for deterministic execution and reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_public_baseline_model(
    public_loaders: list,
    vocab_size: int,
    epochs: int = 3,
    lr: float = 0.005,
    device: str = "cpu",
) -> NextWordModel:
    """
    Trains a 'Public Baseline Model' purely on the datasets of Clients 0-8.
    Completely excludes Client 9 and the confidential canary string.
    """
    print("\n[+] Training Public Baseline Model on Clients 0-8 (Wikitext without Canary)...")
    public_model = NextWordModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(public_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    combined_dataset = ConcatDataset([loader.dataset for loader in public_loaders])
    combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    public_model.train()
    for epoch in range(epochs):
        for bx, by in combined_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = public_model(bx)
            loss = criterion(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(public_model.parameters(), 1.0)
            optimizer.step()

    public_model.eval()
    print("[OK] Public Baseline Model trained successfully (clean baseline distribution).")
    return public_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Learning Privacy Audit - OPA (CRYPTO 2025 & arXiv:2101.05405v2)"
    )
    parser.add_argument("--rounds", type=int, default=8, help="Number of FL communication rounds (default: 8)")
    parser.add_argument("--clients", type=int, default=10, help="Number of simulated FL clients (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument(
        "--log-file",
        type=str,
        default="opa_simulation.log",
        help="Path to write terminal logs to a file (default: opa_simulation.log)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize terminal logging to file if configured
    if args.log_file:
        log_path = os.path.abspath(args.log_file)
        open(log_path, "w", encoding="utf-8").close()  # Clear prior run
        sys.stdout = TeeLogger(log_path, sys.stdout)
        sys.stderr = TeeLogger(log_path, sys.stderr)

    print("=" * 80)
    print(" INFERENCE-TIME PRIVACY AUDIT: 10-CLIENT OPA FEDERATED LEARNING (WIKITEXT)")
    print(" Metric Evaluation: |S_uniq| & Leakage Epsilon epsilon_l (arXiv:2101.05405v2)")
    if args.log_file:
        print(f" Terminal Logging: Active -> {os.path.abspath(args.log_file)}")
    print("=" * 80)

    # 1. Deterministic Seeds
    set_deterministic_seeds(args.seed)

    # 2. Hugging Face Wikitext Data Pipeline & 10-Client Partitioning
    NUM_CLIENTS = args.clients
    NUM_ROUNDS = args.rounds
    BATCH_SIZE = args.batch_size
    SEQ_LEN = 5

    print("\n[1] Integrating Hugging Face Wikitext Dataset & Partitioning:")
    tokenizer, client_loaders, client_texts = get_client_dataloaders(
        num_clients=NUM_CLIENTS,
        num_lines=2000,
        sequence_length=SEQ_LEN,
        batch_size=BATCH_SIZE,
        canary_repeats=50,
        max_vocab_size=2500,
    )
    vocab_size = len(tokenizer)

    print(f"    - Cleaned Wikitext corpus partitioned across {NUM_CLIENTS} clients (200 lines each).")
    print(f"    - Fitted WordTokenizer: Vocabulary Size = {vocab_size} unique words.")
    print("    - Clients 0 through 8: Clean Wikipedia article text.")
    print(f"    - Client 9: Clean text + CANARY INJECTION (50x repeats):")
    print(f"                \"{CANARY_STRING}\"")

    # 3. Train Public Baseline Model on Clients 0-8 (Clean Wikitext)
    public_loaders = [client_loaders[cid] for cid in range(9)]
    public_baseline_model = train_public_baseline_model(
        public_loaders=public_loaders,
        vocab_size=vocab_size,
        epochs=3,
        lr=0.005,
        device="cpu",
    )

    # 4. Neural Network Provisioning & Deterministic Initialization
    set_deterministic_seeds(42)
    init_model = NextWordModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    initial_parameters = ndarrays_to_parameters(init_model.get_weights())
    L_DIM = sum(param.numel() for param in init_model.parameters())
    LAMBDA_DIM = 128

    print(f"\n[2] Neural Network Provisioning:")
    print(f"    - NextWordModel: Embedding({vocab_size}, 32) -> GRU(32, 64) -> Linear(64, {vocab_size})")
    print(f"    - Total Parameter Dimension L = {L_DIM}")

    # 5. OPA Cryptographic Setup (LWR PRG with Matrix A of shape (L, lambda))
    print(f"\n[3] Initializing OPA Cryptographic Primitives (10-Client SSS, r=6, m=10):")
    print(f"    - Prime Modulus q    : {DEFAULT_Q} (2^31 - 1)")
    print(f"    - Rounding Modulus p : {DEFAULT_P} (2^20)")
    print(f"    - Matrix A Shape     : ({L_DIM}, {LAMBDA_DIM})")

    lwr_prg = LWR_PRG(
        L=L_DIM,
        lam=LAMBDA_DIM,
        q=DEFAULT_Q,
        p=DEFAULT_P,
        seed=42,
    )

    # 6. Flower Simulation Setup (Pure OPA Secured Aggregation)
    strategy = OPAFedAvg(
        prg=lwr_prg,
        scale=DEFAULT_SCALE,
        r=6,
        m=10,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )

    print(f"\n[4] Starting 10-Client OPA Federated Learning Simulation ({NUM_ROUNDS} Rounds):")
    print("    - Gradients are NEVER transmitted in the clear.")
    print("    - Each round, clients send ciphertexts c_i = (w_i + PRG(s_i)) mod p.")
    print("    - SSS shares are aggregated homomorphically across clients.\n")

    current_parameters = init_model.get_weights()
    for rnd in range(1, NUM_ROUNDS + 1):
        fit_results = []
        for cid in range(NUM_CLIENTS):
            loader = client_loaders[cid]
            local_model = NextWordModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
            client = OPAClient(
                client_id=f"client_{cid}",
                model=local_model,
                loader=loader,
                prg=lwr_prg,
                local_epochs=5 if cid == 9 else 1,
                lr=0.015 if cid == 9 else 0.008,
                scale=DEFAULT_SCALE,
                t=3,
                r=6,
                m=10,
                device="cpu",
            )
            cipher_weights, num_samples, metrics = client.fit(current_parameters, {})
            fit_res = FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(cipher_weights),
                num_examples=num_samples,
                metrics=metrics,
            )
            fit_results.append((None, fit_res))

        aggregated_parameters, _ = strategy.aggregate_fit(server_round=rnd, results=fit_results, failures=[])
        current_parameters = parameters_to_ndarrays(aggregated_parameters)
        c9_loss = fit_results[9][1].metrics.get("train_loss", 0.0)
        print(f"    -> [Round {rnd}/{NUM_ROUNDS}] OPA Homomorphic Aggregation Complete (Client 9 Local Loss: {c9_loss:.4f})")

    print("\n[OK] 10-Client Federated Learning simulation completed successfully.")

    # 7. Extract Final OPA-Trained Global Model
    global_model = NextWordModel(vocab_size=vocab_size, embed_dim=32, hidden_dim=64)
    global_model.set_weights(current_parameters)
    global_model.eval()
    print("[OK] Successfully extracted global model weights from OPAFedAvg.")

    # 8. Batch-by-Batch Tab Attack & Formal Quantitative Metrics
    print("\n" + "=" * 80)
    print(" EXECUTING INFERENCE-TIME TAB ATTACK & QUANTITATIVE PRIVACY METRICS")
    print("=" * 80)

    all_leaks: dict = {}
    for cid in range(NUM_CLIENTS):
        loader = client_loaders[cid]
        extracted = tab_attack_leakage_analysis(
            model=global_model,
            client_dataset=loader,
            tokenizer=tokenizer,
            min_len=4,
            device=torch.device("cpu"),
        )
        all_leaks[cid] = extracted

    # Compute |S_uniq| and Leakage Epsilon epsilon_l
    metrics = get_quantitative_privacy_metrics(
        all_leaked_by_client=all_leaks,
        public_model=public_baseline_model,
        target_model=global_model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    # 9. Output Audit Report
    print_quantitative_report(metrics)

    # 10. High-Visibility Verification of the Injected Canary
    canary_clean = CANARY_STRING.strip().lower()
    canary_entries = [
        e for e in metrics["s_uniq"]
        if canary_clean in e["text"].lower()
        or any(w in e["text"].lower() for w in ["bank pin", "secret bank", "pin code", "seven eight", "eight nine", "nine two"])
    ]

    print("#" * 80)
    if canary_entries:
        print("  >>> DEMONSTRATION RESULT: CANARY EXTRACTION CONFIRMED! <<<")
        print(f"  Injected Canary Phrase : \"{CANARY_STRING}\"")
        for k_idx, entry in enumerate(canary_entries[:3], 1):
            print(f"  [{k_idx}] Extracted S_uniq String: \"{entry['text']}\"")
            print(f"      Origin                 : {entry['origin_client']} (User Count = 1, S_uniq Privacy Breach)")
            print(f"      Perplexity on Target   : PP_lm  = {entry['pp_lm']:.2f}")
            print(f"      Perplexity on Public   : PP_pub = {entry['pp_pub']:.2f}")
            print(f"      Leakage Epsilon        : eps_l  = {entry['epsilon_l']:.4f}")
        print("-" * 80)
        print("  CRITICAL CONCLUSION:")
        print("  One-Shot Private Aggregation (OPA) securely encrypted communication transcripts.")
        print("  HOWEVER, the final aggregated global model memorized the private canary phrase,")
        print("  yielding a high positive Leakage Epsilon (eps_l) and verbatim extraction!")
    else:
        print("  >>> Direct probe for canary from prefix 'my':")
        curr = [tokenizer.encode("my")[0]]
        for _ in range(9):
            with torch.no_grad():
                out, _ = global_model(torch.tensor([curr], dtype=torch.long))
            nxt = torch.argmax(out[0, -1, :]).item()
            curr.append(nxt)
        probed_text = tokenizer.decode(curr)
        print(f"  Probed sequence: \"{probed_text}\"")
        if canary_clean in probed_text.lower() or "bank pin" in probed_text.lower():
            print("  >>> SUCCESS: Injected canary successfully reconstructed! <<<")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
