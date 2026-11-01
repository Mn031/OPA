# Inference-Time Data Extraction against OPA-Secured Federated Learning

A cryptographic and machine learning benchmark demonstrating the fundamental dichotomy between **Transport-Layer Privacy (Cryptographic Secure Aggregation)** and **Inference-Time Model Privacy (Empirical Memorization)**.

---

## 1. Overview & Theoretical Framework

### The Privacy Paradox in Federated Learning
Federated Learning (FL) was designed to train predictive models over decentralized data without transmitting raw user inputs to a central server. However, distributed training faces two distinct vectors of privacy failure:

1. **Communication-Phase Leakage (Transport Vector):** An eavesdropper or honest-but-curious server observes individual parameter updates $\theta_i$ or gradients $g_i$. Through gradient inversion attacks (e.g., Deep Leakage from Gradients), the adversary can analytically reconstruct private training instances.
2. **Inference-Phase Memorization (Output Vector):** The final aggregated global model $\theta_{global}$ overfits to rare, high-entropy client sequences. An external black-box querier interacting with the deployed model extracts confidential tokens through sequential auto-completion.

```
+-------------------------------------------------------------------------------+
|                             FEDERATED LEARNING PIPELINE                       |
+-------------------------------------------------------------------------------+
| Client 0: w_0 ----+                                                           |
| Client 1: w_1 ----+---> [ OPA SECURE AGGREGATION ] ---> Plaintext Global M*   |
| Client 2: w_2 ----+     * LWR Seed-Homomorphic PRG      |                     |
|                         * Shamir's Secret Sharing       v                     |
|                         * Zero Plaintext Gradients    [ INFERENCE TAB ATTACK ]|
|                           Exposed on Network!           * Black-box Top-1     |
|                                                         * Reconstructs:       |
|                                                           [42, 43, 44, 45, 46]|
+-------------------------------------------------------------------------------+
```

### One-Shot Private Aggregation (OPA)
Based on the research paper:  
> **"OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications to Federated Learning"**  
> *Harish Karthikeyan and Antigoni Polychroniadou (CRYPTO 2025)*

Traditional secure aggregation protocols (e.g., Bonawitz et al. CCS'17) require multi-round interactions (up to 8 rounds) and heavy client-dropout recovery mechanisms. **OPA** achieves secure aggregation with **single-round client participation** by combining:

* **Construction 2: LWR Seed-Homomorphic PRG:**  
  Given a public matrix $A \in \mathbb{Z}_q^{L \times \lambda}$, each client samples a short random seed $s_i \in \mathbb{Z}_q^\lambda$ and computes:
  $$\text{mask}_i = \text{PRG}(s_i) = \left\lfloor (A \cdot s_i \bmod q) \times \frac{p}{q} \right\rfloor \bmod p$$
  The local weights $w_i$ are quantized into $\mathbb{Z}_p$ and masked: $c_i = (w_i + \text{mask}_i) \bmod p$.
* **Construction 1: Shamir's Secret Sharing (SSS) over $\mathbb{F}_q$:**  
  Each client splits its short seed $s_i$ into $m$ threshold shares using a random polynomial over $\mathbb{F}_q$. Because SSS is linearly homomorphic, intermediate committee members sum their respective shares:
  $$S^{(j)} = \sum_{i=1}^n s_i^{(j)} \pmod q$$
* **Homomorphic Server Recovery:**  
  The server reconstructs the global seed sum $s_{global} = \sum_{i=1}^n s_i \pmod q$ from $r$ committee shares via Lagrange interpolation:
  $$s_{global} = \sum_{j=1}^r S^{(j)} \ell_j(0) \pmod q$$
  By the seed-homomorphism of LWR, the server computes $\sum \text{mask}_i \approx \text{PRG}(s_{global})$ and unmasks the aggregated update without ever seeing any individual client's weights in the clear!

### Why OPA Fails to Prevent Memorization
While OPA **mathematically guarantees that communication transcripts leak zero information beyond the sum**, it performs exact linear aggregation:
$$\theta_{global} \leftarrow \frac{1}{n} \sum_{i=1}^n \theta_i$$
If Client 2's private update $\theta_2$ contains neural network weights that heavily memorize a confidential pattern, the aggregated global model preserves that pattern. Cryptographic secure aggregation protects the **pipe**, but does **not** sanitize the **destination model**.

---

## 2. The Attack: Inference-Time "Tab Attack" ($S_{uniq}$)

Based on the research paper:  
> **"Training Data Leakage Analysis in Language Models"**  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim (arXiv:2101.05405v2)*

The **Tab Attack** represents a realistic, **black-box** query threat model:
* **Inspiration:** Text auto-completion features in operating systems, code editors, and mobile keyboards (like pressing the **TAB** key to accept the top suggested next word).
* **Adversarial Access:** Zero access to model internals, gradients, or communication ciphertexts. The attacker only queries the deployed model with a prefix prompt and reads the top-$1$ prediction $\text{argmax}_{w} P(w \mid \text{prefix})$.
* **Algorithm 1 (Token-by-Token Streak Tracking):** Given prefix $w_1, \dots, w_t$, the model predicts $w_{t+1}$. If this prediction matches ground truth, it "presses TAB" and extends the prefix. If consecutive top-1 predictions succeed for 3 or more tokens, the model is extracting an entire phrase verbatim from memory.
* **The $S_{uniq}$ Metric:** A sequence is classified as $S_{uniq}$ if its ground-truth occurrence in the training data is exactly **one** ($\text{user\_count} = 1$). Extracting $S_{uniq}$ sequences constitutes an undeniable privacy breach.

---

## 3. Project Architecture

```
d:/MTP/OPA/
├── __init__.py      # Package initializer for module discovery
├── crypto.py        # OPA Cryptography: LWR Seed-Homomorphic PRG & Shamir Secret Sharing
├── model.py         # PyTorch NextWordModel (Embedding 16 -> GRU 32 -> Linear 50)
├── dataset.py       # Deterministic Data Pipeline with Client 2 Vulnerability Injection
├── opa_flwr.py      # Flower OPAClient (NumPyClient) & OPAStrategy (FedAvg Override)
├── attack.py        # Algorithm 1 Tab Attack & S_uniq Metric Extraction
├── main.py          # Complete 10-round Simulation Orchestrator & Privacy Audit
└── README.md        # Academic documentation & theoretical framework
```

### Module Breakdown:
1. **`crypto.py`**:
   - `LWR_PRG`: Implements public matrix $A \in \mathbb{Z}_q^{L \times \lambda}$, $q = 2^{31}-1$, $p = 2^{20}$. Implements `expand(seed_s)` with rounding $\lfloor (A \cdot s \bmod q) \times (p/q) \rfloor \bmod p$.
   - `share(secret, t, r, m)` & `reconstruct(shares)`: Polynomial secret sharing and Lagrange interpolation over prime field $\mathbb{F}_q$.
   - `quantize_to_zp` & `dequantize_from_zp`: Fixed-point quantization between PyTorch float tensors and $\mathbb{Z}_p$.
2. **`model.py`**:
   - `NextWordModel`: Lightweight recurrent language model with $L = 7250$ parameters.
   - Built-in `get_weights()` and `set_weights()` for Flower serialization.
3. **`dataset.py`**:
   - `VOCAB_SIZE = 50`.
   - Client 0 & 1: Standard shared integer sequences (non-sensitive domain).
   - Client 2: Exclusively injected with the confidential sequence `[42, 43, 44, 45, 46]` repeated over 50 batches.
4. **`opa_flwr.py`**:
   - `OPAClient`: Evaluates local epochs, quantizes weights, applies LWR mask, shares seed via SSS, returns masked ciphertext.
   - `OPAStrategy`: Aggregates SSS shares, reconstructs global seed, evaluates global LWR mask, unmasks aggregated weights.
5. **`attack.py`**:
   - `tab_attack_leakage_analysis`: Evaluates token-by-token streaks where top-1 matches ground truth for $\ge 3$ tokens.
   - `get_unique_leaked_sequences`: Filters extracted sequences by client occurrence to isolate $S_{uniq}$.
6. **`main.py`**:
   - Coordinates deterministic seeds, public matrix setup, 10 rounds of Flower simulation, and the final privacy audit.

---

## 4. Quickstart & Execution

### Prerequisites
Ensure Python 3.10+ is installed with PyTorch, Flower (`flwr`), and Ray:

```bash
pip install torch flwr ray numpy
```

### Running the Full Pipeline
Navigate to the `OPA` directory and run `main.py`:

```bash
cd d:/MTP/OPA
python main.py
```

Or execute as a module from the workspace root:

```bash
cd d:/MTP
python -m OPA.main
```

---

## 5. Sample Execution Output

```text
==============================================================================
 INFERENCE-TIME DATA EXTRACTION AGAINST OPA-SECURED FEDERATED LEARNING
 Proof-of-Concept: One-Shot Private Aggregation vs. Tab Attack (S_uniq)
==============================================================================

[+] Initializing Cryptographic Parameters (Construction 2 LWR PRG):
    - Parameter Dimension L  : 7250
    - Seed Space Dimension λ : 16
    - Prime Modulus q        : 2147483647 (2^31 - 1)
    - Rounding Modulus p     : 1048576 (2^20)
    - Shared Public Matrix A initialized with shape (7250, 16).

[+] Preparing Client Data Pipeline (Vocab Size = 50):
    - Client 0: Standard shared integer sequences.
    - Client 1: Standard shared integer sequences.
    - Client 2: Injected with confidential sequence [42, 43, 44, 45, 46]
      (Repeated over 50 batches to force memorization during local training).

[+] Launching Flower Simulation (10 Rounds, OPA Secure Aggregation):
    - Clients mask local weights: c_i = (w_i + PRG(s_i)) mod p
    - Clients split seeds using Shamir's Secret Sharing: SSS.Share(s_i, r=2, m=3)
    - Server aggregates shares homomorphically to recover ∑ s_i
    - Server computes global mask PRG(∑ s_i) and recovers plaintext average.

[OK] Federated Learning simulation finished successfully.
[OK] Successfully extracted global model weights from OPAStrategy.

==============================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK AUDIT (Algorithm 1)
==============================================================================
Querying the global model token-by-token using simulated TAB auto-completion...

==============================================================================
      TAB ATTACK INFERENCE-TIME PRIVACY AUDIT (arXiv:2101.05405v2)
==============================================================================
Total Unique Leaked Sequences (S_uniq, user_count=1): 1
Total Multi-User Sequences    (S_mult, user_count>1): 3
------------------------------------------------------------------------------

 [!] PRIVACY BREACH CONFIRMED: S_uniq (Sequences Unique to a Single Client):
Origin Client    | Length | Reconstructed Token Sequence
------------------------------------------------------------------------------
client_2         | 5      | [42, 43, 44, 45, 46]

 Cryptographic Paradox Analysis:
 -> Even though OPA provided provably secure communication masking (LWR PRG + SSS),
    the final aggregated model parameters internalized the local memorization.
 -> As a result, the inference-time Tab Attack extracts confidential sequences verbatim!

##############################################################################
  >>> DEMONSTRATION RESULT: VULNERABILITY CONFIRMED! <<<
  Target Confidential Sequence: [42, 43, 44, 45, 46]
  Origin: Client 2 only (User Count = 1, S_uniq Privacy Breach)
  Status: EXTRACTED VERBATIM BY INFERENCE-TIME TAB ATTACK!
------------------------------------------------------------------------------
  CONCLUSION:
  OPA mathematically guaranteed communication privacy during training
  (preventing eavesdroppers and the server from inverting individual updates).
  HOWEVER, the final aggregated weights carry the empirical memorization,
  leaving the model completely vulnerable to post-training black-box extraction.
##############################################################################
```

---

## 6. Mitigations & Defense Directions

To resolve the privacy paradox, cryptographic transport security must be combined with algorithm-level privacy preservation:

1. **Differentially Private SGD (DP-SGD / DP-FedAvg):**
   - Inject calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ and enforce per-sample gradient clipping before secure aggregation.
   - Bounding individual sample influence prevents the network from memorizing low-frequency client sequences.
2. **Inference-Time Repetition & Entropy Filtering:**
   - Filter predictions whose probability spike is disproportionately high for rare prefixes.
3. **Scrubbing & PII Redaction:**
   - Pre-training sanitization on client edge devices before local gradient computation.

---

## 7. References

```bibtex
@inproceedings{karthikeyan2025opa,
  title={OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications to Federated Learning},
  author={Karthikeyan, Harish and Polychroniadou, Antigoni},
  booktitle={Advances in Cryptology -- CRYPTO 2025},
  pages={319--353},
  year={2025},
  publisher={Springer}
}

@article{inan2021training,
  title={Training Data Leakage Analysis in Language Models},
  author={Inan, Huseyin A and Ramadan, Osman and Wutschitz, Lukas and Jones, Daniel and R{\"u}hle, Victor and Withers, James and Sim, Robert},
  journal={arXiv preprint arXiv:2101.05405},
  year={2021}
}
```
