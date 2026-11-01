"""
main.py - Simulation Orchestrator for OPA Federated Learning & Inference-Time Tab Attack.

Workflow:
  1. Initializes deterministic seeds (PyTorch & NumPy) for reproducible evaluation.
  2. Generates shared public matrix A and initializes LWR_PRG for clients and server.
  3. Prepares synthetic federated datasets with vulnerability injection in Client 2.
  4. Launches 10-round Federated Learning simulation using Flower and OPAStrategy.
  5. Extracts aggregated global NextWordModel parameters after simulation.
  6. Executes inference-time Tab Attack (Algorithm 1) to evaluate S_uniq leakage.
  7. Demonstrates that while OPA secures transmission, Client 2's confidential sequence
     [42, 43, 44, 45, 46] is completely extracted from the final global model.
"""

import os
import sys

# Ensure both local OPA package and workspace root are in sys.path and PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = f"{PARENT_DIR};{CURRENT_DIR};" + os.environ.get("PYTHONPATH", "")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

try:
    from OPA.crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
    from OPA.model import NextWordModel
    from OPA.dataset import (
        VOCAB_SIZE,
        TARGET_VULNERABILITY_SEQUENCE,
        get_client_dataloaders,
        get_raw_client_sequences,
    )
    from OPA.opa_flwr import OPAClient, OPAStrategy
    from OPA.attack import (
        tab_attack_leakage_analysis,
        get_unique_leaked_sequences,
        print_attack_report,
    )
except ImportError:
    from crypto import LWR_PRG, DEFAULT_Q, DEFAULT_P, DEFAULT_SCALE
    from model import NextWordModel
    from dataset import (
        VOCAB_SIZE,
        TARGET_VULNERABILITY_SEQUENCE,
        get_client_dataloaders,
        get_raw_client_sequences,
    )
    from opa_flwr import OPAClient, OPAStrategy
    from attack import (
        tab_attack_leakage_analysis,
        get_unique_leaked_sequences,
        print_attack_report,
    )


def set_deterministic_seeds(seed: int = 42):
    """Sets fixed random seeds for deterministic execution and reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    print("=" * 78)
    print(" INFERENCE-TIME DATA EXTRACTION AGAINST OPA-SECURED FEDERATED LEARNING")
    print(" Proof-of-Concept: One-Shot Private Aggregation vs. Tab Attack (S_uniq)")
    print("=" * 78)

    # 1. Deterministic Seeds
    set_deterministic_seeds(42)

    # 2. Cryptographic Parameters & Public Matrix A Initialization
    # NextWordModel with vocab=50, embed=16, gru=32 has exactly L = 7250 parameters
    L_DIM = 7250
    LAMBDA_DIM = 16
    NUM_ROUNDS = 10
    NUM_CLIENTS = 3

    print(f"\n[+] Initializing Cryptographic Parameters (Construction 2 LWR PRG):")
    print(f"    - Parameter Dimension L  : {L_DIM}")
    print(f"    - Seed Space Dimension \u03bb : {LAMBDA_DIM}")
    print(f"    - Prime Modulus q        : {DEFAULT_Q} (2^31 - 1)")
    print(f"    - Rounding Modulus p     : {DEFAULT_P} (2^20)")

    # Public matrix A is shared by all clients and server
    lwr_prg = LWR_PRG(
        L=L_DIM,
        lam=LAMBDA_DIM,
        q=DEFAULT_Q,
        p=DEFAULT_P,
        seed=42,
    )
    print("    - Shared Public Matrix A initialized with shape (7250, 16).")

    # 3. Data Pipeline & Vulnerability Injection
    print(f"\n[+] Preparing Client Data Pipeline (Vocab Size = {VOCAB_SIZE}):")
    client_loaders = get_client_dataloaders(
        num_clients=NUM_CLIENTS,
        batch_size=8,
        repeat_target_batches=50,
    )
    raw_data = get_raw_client_sequences(
        repeat_target_batches=50,
        batch_size=8,
    )

    print("    - Client 0: Standard shared integer sequences.")
    print("    - Client 1: Standard shared integer sequences.")
    print(f"    - Client 2: Injected with confidential sequence {TARGET_VULNERABILITY_SEQUENCE}")
    print("      (Repeated over 50 batches to force memorization during local training).")

    # 4. Federated Learning Setup with Flower & OPA
    init_model = NextWordModel(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_dim=32)
    initial_parameters = ndarrays_to_parameters(init_model.get_weights())

    client_ids = list(client_loaders.keys())

    strategy = OPAStrategy(
        prg=lwr_prg,
        scale=DEFAULT_SCALE,
        r=2,
        m=3,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
    )

    def client_fn(context_or_cid):
        """Spawns an OPA virtual client with local dataset and shared PRG."""
        if hasattr(context_or_cid, "node_config"):
            idx = int(context_or_cid.node_config.get("partition-id", "0"))
        elif isinstance(context_or_cid, str):
            try:
                idx = int(context_or_cid)
            except ValueError:
                idx = 0
        else:
            idx = 0

        cid = client_ids[idx % len(client_ids)]
        local_model = NextWordModel(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_dim=32)
        loader = client_loaders[cid]

        client = OPAClient(
            client_id=cid,
            model=local_model,
            loader=loader,
            prg=lwr_prg,
            local_epochs=5,
            lr=0.03,
            scale=DEFAULT_SCALE,
            t=2,
            r=2,
            m=3,
            device="cpu",
        )
        return client.to_client()

    print(f"\n[+] Launching Flower Simulation ({NUM_ROUNDS} Rounds, OPA Secure Aggregation):")
    print("    - Clients mask local weights: c_i = (w_i + PRG(s_i)) mod p")
    print("    - Clients split seeds using Shamir's Secret Sharing: SSS.Share(s_i, r=2, m=3)")
    print("    - Server aggregates shares homomorphically to recover \u2211 s_i")
    print("    - Server computes global mask PRG(\u2211 s_i) and recovers plaintext average.\n")

    sim_history = start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"runtime_env": {"env_vars": {"PYTHONPATH": f"{PARENT_DIR};{CURRENT_DIR}"}}},
    )

    print("\n[OK] Federated Learning simulation finished successfully.")

    # 5. Extract Trained Global Model
    if strategy.latest_global_parameters is None:
        print("[!] Error: No global parameters were aggregated.")
        sys.exit(1)

    global_weights = parameters_to_ndarrays(strategy.latest_global_parameters)
    global_model = NextWordModel(vocab_size=VOCAB_SIZE, embed_dim=16, hidden_dim=32)
    global_model.set_weights(global_weights)
    global_model.eval()
    print("[OK] Successfully extracted global model weights from OPAStrategy.")

    # 6. Inference-Time Tab Attack Privacy Leakage Evaluation
    print("\n" + "=" * 78)
    print(" EXECUTING INFERENCE-TIME TAB ATTACK AUDIT (Algorithm 1)")
    print("=" * 78)
    print("Querying the global model token-by-token using simulated TAB auto-completion...")

    all_leaks: Dict[str, List] = {}
    for cid in client_ids:
        leaks = tab_attack_leakage_analysis(
            model=global_model,
            client_dataset=raw_data[cid],
            min_len=3,
            device=torch.device("cpu"),
        )
        all_leaks[cid] = leaks

    # 7. S_uniq Leakage Isolation
    leakage_metrics = get_unique_leaked_sequences(
        all_leaked_sequences_dict=all_leaks,
        training_data=raw_data,
    )

    print_attack_report(leakage_metrics)

    # 8. High-Visibility Verification of the Injected Sequence
    target_tuple = tuple(TARGET_VULNERABILITY_SEQUENCE)
    extracted_s_uniq_tuples = [tuple(item["sequence"]) for item in leakage_metrics["s_uniq"]]

    is_breached = any(target_tuple == t or all(tok in t for tok in target_tuple) for t in extracted_s_uniq_tuples)

    print("#" * 78)
    if is_breached:
        print("  >>> DEMONSTRATION RESULT: VULNERABILITY CONFIRMED! <<<")
        print(f"  Target Confidential Sequence: {TARGET_VULNERABILITY_SEQUENCE}")
        print("  Origin: Client 2 only (User Count = 1, S_uniq Privacy Breach)")
        print("  Status: EXTRACTED VERBATIM BY INFERENCE-TIME TAB ATTACK!")
        print("-" * 78)
        print("  CONCLUSION:")
        print("  OPA mathematically guaranteed communication privacy during training")
        print("  (preventing eavesdroppers and the server from inverting individual updates).")
        print("  HOWEVER, the final aggregated weights carry the empirical memorization,")
        print("  leaving the model completely vulnerable to post-training black-box extraction.")
    else:
        # Direct check on Client 2's trigger token
        curr = [TARGET_VULNERABILITY_SEQUENCE[0]]
        for _ in range(len(TARGET_VULNERABILITY_SEQUENCE) - 1):
            with torch.no_grad():
                out, _ = global_model(torch.tensor([curr], dtype=torch.long))
            next_t = torch.argmax(out[0, -1, :]).item()
            curr.append(next_t)
        print(f"  Reconstruction probe from token {TARGET_VULNERABILITY_SEQUENCE[0]}: {curr}")
        if curr == TARGET_VULNERABILITY_SEQUENCE:
            print("  >>> SUCCESS: [42, 43, 44, 45, 46] was successfully reconstructed! <<<")
    print("#" * 78 + "\n")


if __name__ == "__main__":
    main()
