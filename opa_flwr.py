"""
opa_flwr.py - Federated Learning with One-Shot Private Aggregation (OPA).

Implements:
  1. OPAClient (flwr.client.NumPyClient):
     - Trains local NextWordModel with Cross-Entropy loss.
     - Samples random seed s in Z_q^lambda.
     - Expands s via LWR_PRG to generate pseudorandom mask in Z_p^L.
     - Encrypts local weights: Ciphertext = (QuantizedWeights + Mask) mod p.
     - Uses Shamir's Secret Sharing (SSS) to split seed s into m shares.
     - Returns Ciphertext as model parameters, and SSS shares in metrics["seed_shares"].

  2. OPAStrategy (flwr.server.strategy.FedAvg):
     - Collects ciphertexts and "seed_shares" from participating clients.
     - Linearly aggregates the shares for each committee member modulo q.
     - Reconstructs the global seed sum_i s_i modulo q via Lagrange interpolation.
     - Expands the global seed via LWR_PRG to compute the global mask sum_i mask_i mod p.
     - Sums ciphertexts, subtracts the global mask modulo p, and de-quantizes to
       recover exact plaintext averaged global model weights.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURRENT_DIR)
for _p in [_CURRENT_DIR, _PARENT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

try:
    from OPA.crypto import (
        LWR_PRG,
        share,
        reconstruct,
        quantize_to_zp,
        dequantize_from_zp,
        DEFAULT_Q,
        DEFAULT_P,
        DEFAULT_SCALE,
    )
    from OPA.model import NextWordModel
except ImportError:
    from crypto import (
        LWR_PRG,
        share,
        reconstruct,
        quantize_to_zp,
        dequantize_from_zp,
        DEFAULT_Q,
        DEFAULT_P,
        DEFAULT_SCALE,
    )
    from model import NextWordModel


# =====================================================================
# 1. CUSTOM OPA CLIENT (FLOWER NUMPYCLIENT)
# =====================================================================

class OPAClient(NumPyClient):
    """
    Flower NumPyClient with One-Shot Private Aggregation (OPA) Masking.
    """
    def __init__(
        self,
        client_id: str,
        model: NextWordModel,
        loader: torch.utils.data.DataLoader,
        prg: LWR_PRG,
        local_epochs: int = 5,
        lr: float = 0.03,
        scale: float = DEFAULT_SCALE,
        t: int = 2,
        r: int = 2,
        m: int = 3,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.prg = prg
        self.local_epochs = local_epochs
        self.lr = lr
        self.scale = scale
        self.t = t
        self.r = r
        self.m = m
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return self.model.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # 1. Update local model with incoming parameters
        self.model.set_weights(parameters)

        # 2. Local Training via Cross-Entropy Loss
        self.model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        last_loss = 0.0
        for _ in range(self.local_epochs):
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), batch_y.reshape(-1))
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

        # 3. Extract trained local weights
        trained_weights = self.model.get_weights()
        flat_weights, shapes = self.prg.flatten_weights(trained_weights)

        # 4. OPA Cryptographic Masking
        # Sample random seed s in Z_q^lambda
        seed_s = np.random.randint(0, self.prg.q, size=self.prg.lam, dtype=np.int64)

        # Expand seed_s via LWR PRG to obtain pseudorandom mask in Z_p^L
        mask = self.prg.expand(seed_s)

        # Quantize floating-point weights to Z_p
        quantized_w = quantize_to_zp(flat_weights, scale=self.scale, p=self.prg.p)

        # Compute ciphertext: c = (w_quant + mask) mod p
        ciphertext_flat = (quantized_w + mask) % self.prg.p

        # Unflatten ciphertext back into Flower parameter list format
        ciphertext_weights = self.prg.unflatten_weights(ciphertext_flat.astype(np.float32), shapes)

        # 5. Shamir's Secret Sharing (SSS) of seed_s into m shares
        sss_shares = share(
            secret=seed_s,
            t=self.t,
            r=self.r,
            m=self.m,
            q=self.prg.q,
        )

        # Serialize shares to JSON string for safe Flower metrics transport
        serialized_shares = json.dumps([[int(x), [int(val) for val in y]] for x, y in sss_shares])

        num_examples = len(self.loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(last_loss),
            "seed_shares": serialized_shares,
        }

        return ciphertext_weights, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict]:
        self.model.set_weights(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_x, batch_y in self.loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits, _ = self.model(batch_x)
                loss = criterion(logits.reshape(-1, self.model.vocab_size), batch_y.reshape(-1))
                total_loss += loss.item() * batch_x.size(0)

                preds = logits.argmax(dim=-1)
                mask = (batch_y != 0)
                total_correct += ((preds == batch_y) & mask).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / max(len(self.loader.dataset), 1)
        acc = total_correct / max(total_tokens, 1)
        num_examples = len(self.loader.dataset)

        return float(avg_loss), num_examples, {"loss": float(avg_loss), "accuracy": float(acc)}


# =====================================================================
# 2. CUSTOM OPA STRATEGY (FLOWER FEDAVG OVERRIDE)
# =====================================================================

class OPAStrategy(FedAvg):
    """
    Custom Flower FedAvg strategy implementing OPA unmasking and aggregation:
      1. Receives masked weight ciphertexts from clients.
      2. Reconstructs the global seed sum_i s_i mod q from SSS shares.
      3. Expands the global seed into the global mask sum_i mask_i mod p via LWR_PRG.
      4. Subtracted unmasked plaintext weights are averaged and broadcast.
    """
    def __init__(
        self,
        prg: LWR_PRG,
        scale: float = DEFAULT_SCALE,
        r: int = 2,
        m: int = 3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prg = prg
        self.scale = scale
        self.r = r
        self.m = m
        self.latest_global_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Securely aggregates client model updates using OPA unmasking.
        """
        if not results:
            return None, {}

        # 1. Extract ciphertexts and SSS shares from all participating clients
        client_ciphertexts_flat: List[np.ndarray] = []
        client_shares_list: List[List[Tuple[int, np.ndarray]]] = []
        shapes_reference = None

        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            flat_c, shapes = self.prg.flatten_weights(client_ndarrays)
            if shapes_reference is None:
                shapes_reference = shapes
            client_ciphertexts_flat.append(flat_c.astype(np.int64) % self.prg.p)

            # Extract seed shares from metrics dictionary
            shares_raw = fit_res.metrics["seed_shares"]
            if isinstance(shares_raw, str):
                shares_decoded = json.loads(shares_raw)
            else:
                shares_decoded = shares_raw

            parsed_shares = [(int(x), np.array(y, dtype=np.int64)) for x, y in shares_decoded]
            client_shares_list.append(parsed_shares)

        num_clients = len(results)

        # 2. Linearly aggregate SSS shares across clients for the first r parties
        # In OPA, committee members j in [1, r] sum their shares: S_j = sum_i s_i^(j) mod q
        combined_shares: List[Tuple[int, np.ndarray]] = []
        for party_idx in range(self.r):
            party_id = client_shares_list[0][party_idx][0]
            party_share_sum = np.zeros(self.prg.lam, dtype=np.int64)
            for c_shares in client_shares_list:
                party_share_sum = (party_share_sum + c_shares[party_idx][1]) % self.prg.q
            combined_shares.append((party_id, party_share_sum))

        # 3. Reconstruct global seed: s_global = sum_{i=1}^n s_i mod q
        global_seed = reconstruct(combined_shares, q=self.prg.q)

        # 4. Expand global seed to compute global mask: sum_i mask_i mod p
        global_mask = self.prg.expand(global_seed)

        # 5. Sum all client ciphertexts in Z_p: sum_i (w_i + mask_i) mod p
        total_ciphertext = np.zeros(self.prg.L, dtype=np.int64)
        for c in client_ciphertexts_flat:
            total_ciphertext = (total_ciphertext + c) % self.prg.p

        # 6. Unmask: w_sum_quant = (total_ciphertext - global_mask) mod p
        w_sum_quant = (total_ciphertext - global_mask) % self.prg.p

        # 7. Dequantize from Z_p and average across clients to recover plaintext weights
        recovered_average_flat = dequantize_from_zp(
            weights_zp=w_sum_quant,
            scale=self.scale,
            p=self.prg.p,
            num_contributors=num_clients,
        )

        # 8. Unflatten back to model parameter list
        averaged_ndarrays = self.prg.unflatten_weights(recovered_average_flat, shapes_reference)
        self.latest_global_parameters = ndarrays_to_parameters(averaged_ndarrays)

        return self.latest_global_parameters, {"num_aggregated": num_clients}
