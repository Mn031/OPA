# Inference-Time Data Extraction against OPA-Secured Federated Learning

A cryptographic and machine learning benchmark demonstrating the fundamental dichotomy between **Transport-Layer Privacy (Cryptographic Secure Aggregation)** and **Inference-Time Model Privacy (Empirical Memorization)** using PyTorch and Flower (`flwr`).

---

## 1. Overview & Theoretical Framework

### The Privacy Paradox in Federated Learning
Federated Learning (FL) was designed to train predictive models over decentralized data without transmitting raw user inputs to a central server. However, distributed learning is vulnerable to two distinct categories of privacy leakage:

1. **Communication-Phase Leakage (Transport Vector):** An eavesdropper or honest-but-curious server observes individual parameter updates $\theta_i$ or gradients $g_i$. Through gradient inversion attacks (e.g., Deep Leakage from Gradients), the adversary can analytically reconstruct private training instances.
2. **Inference-Phase Memorization (Output Vector):** The final aggregated global model $\theta_{global}$ overfits to rare, high-entropy client sequences. An external black-box querier interacting with the deployed model extracts confidential text through sequential auto-completion.

```
+-----------------------------------------------------------------------------------------+
|                               FEDERATED LEARNING PIPELINE                               |
+-----------------------------------------------------------------------------------------+
| Client 0: w_0 ----+                                                                     |
| Client 1: w_1 ----+---> [ OPA SECURE AGGREGATION ] ---> Plaintext Global Model M*       |
| Client 2: w_2 ----+     * LWR Seed-Homomorphic PRG        |                             |
|   (holds secret:        * Shamir's Secret Sharing         v                             |
|    "my secret bank      * Zero Plaintext Gradients      [ INFERENCE TAB ATTACK ]        |
|     pin code is...")      Exposed on Network!             * Black-box Top-1 Auto-comp   |
|                                                           * Reconstructs:               |
|                                                             "my secret bank pin code    |
|                                                              is seven eight nine two"   |
+-----------------------------------------------------------------------------------------+
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

### Why OPA Fails to Prevent Text Memorization
While OPA **mathematically guarantees that communication transcripts leak zero information beyond the sum**, it performs exact linear aggregation:
$$\theta_{global} \leftarrow \frac{1}{n} \sum_{i=1}^n \theta_i$$
If Client 2's private update $\theta_2$ contains neural network weights that heavily memorize a confidential sentence, the aggregated global model preserves that pattern. Cryptographic secure aggregation protects the **communication channel**, but does **not** sanitize the **destination model**.

---

## 2. The Attack: Inference-Time "Tab Attack" ($\mathcal{S}_{uniq}$)

Based on the research paper:  
> **"Training Data Leakage Analysis in Language Models"**  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim (arXiv:2101.05405v2)*

The **Tab Attack** represents a realistic, **black-box** query threat model:
* **Inspiration:** Text auto-completion features in operating systems, code editors, and mobile keyboards (like pressing the **TAB** key to accept the top suggested next word).
* **Adversarial Access:** Zero access to model internals, gradients, or communication ciphertexts. The attacker only queries the deployed model with a prefix prompt and reads the top-$1$ prediction $\text{argmax}_{w} P(w \mid \text{prefix})$.
* **Algorithm 1 (Token-by-Token Streak Tracking):** Given prefix words $w_1, \dots, w_t$, the model predicts $w_{t+1}$. If this prediction matches ground truth, it "presses TAB" and extends the prefix. If consecutive top-1 predictions succeed for 4 or more words, the model is extracting an entire phrase verbatim from memory.
* **The $\mathcal{S}_{uniq}$ Metric:** A sequence is classified as $\mathcal{S}_{uniq}$ if its ground-truth occurrence in the training data is exactly **one** ($\text{user\_count} = 1$). Extracting $\mathcal{S}_{uniq}$ sequences constitutes an undeniable privacy breach.

---

## 3. Project Architecture

```
d:/MTP/OPA/
├── __init__.py      # Package initializer for module discovery
├── crypto.py        # OPA Cryptography: LWR Seed-Homomorphic PRG & Shamir Secret Sharing
├── model.py         # PyTorch NextWordModel (Embedding 32 -> GRU 64 -> Linear vocab_size)
├── dataset.py       # WordTokenizer & Text Corpus with Client 2 Vulnerability Injection
├── opa_flwr.py      # Pure OPAClient (NumPyClient) & OPAFedAvg (Flower Strategy)
├── attack.py        # Algorithm 1 Tab Attack & S_uniq String Extraction
├── main.py          # 15-Round Simulation Orchestrator & Privacy Audit
└── README.md        # Academic documentation & theoretical framework
```

### Module Breakdown:
1. **`dataset.py`**:
   - `WordTokenizer`: Lightweight word-level tokenizer handling `<PAD>` (index 0) and `<UNK>` (index 1).
   - Text Corpus: Standard shared sentences across Client 0 and Client 1.
   - Vulnerability Injection: Unique confidential phrase `"my secret bank pin code is seven eight nine two"` injected strictly into Client 2.
   - Sliding window dataset generation for PyTorch DataLoaders.
2. **`model.py`**:
   - `NextWordModel`: Recurrent neural network with dynamic `vocab_size` from `WordTokenizer`.
   - `nn.Embedding(vocab_size, 32) -> nn.GRU(32, 64, batch_first=True) -> nn.Linear(64, vocab_size)`.
   - Built-in `get_weights()` and `set_weights()` for Flower serialization.
3. **`crypto.py`**:
   - `LWR_PRG`: Implements public matrix $A \in \mathbb{Z}_q^{L \times \lambda}$, $q = 2^{31}-1$, $p = 2^{20}$. Implements `expand(seed_s)` with rounding.
   - `share(secret, t, r, m)` & `reconstruct(shares)`: Polynomial secret sharing and Lagrange interpolation over prime field $\mathbb{F}_q$.
   - Fixed-point quantization between float tensors and $\mathbb{Z}_p$.
4. **`opa_flwr.py`**:
   - `OPAClient`: Evaluates local epochs, quantizes weights, applies LWR mask, shares seed via SSS, returns masked ciphertext.
   - `OPAFedAvg`: Strategy that homomorphically aggregates SSS shares, reconstructs global seed, evaluates global LWR mask, and unmasks aggregated weights.
5. **`attack.py`**:
   - `tab_attack_leakage_analysis`: Evaluates token-by-token streaks where top-1 matches ground truth for $\ge 4$ words.
   - `get_unique_leaked_sequences`: Decodes token indices back to readable strings and filters by client occurrence to isolate $\mathcal{S}_{uniq}$.
6. **`main.py`**:
   - Coordinates deterministic seeds, public matrix setup, 15 rounds of pure OPA Flower simulation, and the final privacy audit.

---

## 4. Quickstart & Execution

### Prerequisites
Ensure Python 3.10+ is installed with PyTorch, Flower (`flwr`), and Ray:

```bash
pip install torch flwr ray numpy
```

### Running the Full Pipeline
Navigate to the `OPA` directory and execute `main.py`:

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

[1] Preparing Text Corpus & WordTokenizer:
    - Fitted WordTokenizer on corpus: Total Vocabulary = 61 unique words
    - Client 0: Standard domain sentences.
    - Client 1: Standard domain sentences (shared domain language).
    - Client 2: Standard text + Injected Secret:
                "my secret bank pin code is seven eight nine two"
      (Repeated to force empirical memorization during local training).

[2] Neural Network Architecture (Dynamic vocab_size = 61):
    - Embedding(vocab_size, 32) -> GRU(32, 64, batch_first=True) -> Linear(64, vocab_size)
    - Total Model Parameter Count L = 24733

[3] Cryptographic Parameters (Construction 2 LWR PRG & Construction 1 SSS):
    - Parameter Dimension L  : 24733
    - Seed Space Dimension λ : 16
    - Prime Modulus q        : 2147483647 (2^31 - 1)
    - Rounding Modulus p     : 1048576 (2^20)
    - Shared Public Matrix A initialized with shape (24733, 16).

[4] Launching Flower Simulation (15 Rounds, Pure OPA Secure Aggregation):
    - Clients mask local weights: c_i = (w_i + PRG(s_i)) mod p
    - Clients split seeds using Shamir's Secret Sharing: SSS.Share(s_i, r=2, m=3)
    - Server aggregates shares homomorphically to recover ∑ s_i
    - Server computes global mask PRG(∑ s_i) and recovers plaintext average.

[OK] Federated Learning simulation finished successfully.
[OK] Successfully extracted global model weights from OPAFedAvg.

==============================================================================
 EXECUTING INFERENCE-TIME TAB ATTACK AUDIT (Algorithm 1)
==============================================================================
Querying the global model token-by-token using simulated TAB auto-completion...

==============================================================================
      TAB ATTACK INFERENCE-TIME PRIVACY AUDIT (arXiv:2101.05405v2)
==============================================================================
Total Unique Leaked Phrases (S_uniq, user_count=1): 3
Total Common Domain Phrases (S_mult, user_count>1): 2
------------------------------------------------------------------------------

 [!] PRIVACY BREACH DETECTED: S_uniq (Phrases Unique to a Single Client):
Origin Client    | Words | Extracted Text String
------------------------------------------------------------------------------
client_2         | 10    | "my secret bank pin code is seven eight nine two"
client_0         | 7     | "artificial intelligence will transform many industries"
client_1         | 7     | "deep neural networks require large datasets to train"

 Cryptographic Paradox Analysis:
 -> OPA mathematically guaranteed gradient confidentiality during training
    (using LWR PRG masking and Shamir Secret Sharing).
 -> HOWEVER, the final aggregated weights carry the client's empirical memorization,
    allowing the Tab Attack to reconstruct confidential text verbatim!

##############################################################################
  >>> DEMONSTRATION RESULT: VULNERABILITY CONFIRMED! <<<
  Target Confidential Sentence: "my secret bank pin code is seven eight nine two"
  Origin: Client 2 only (User Count = 1, S_uniq Privacy Breach)
  Status: EXTRACTED VERBATIM BY INFERENCE-TIME TAB ATTACK!
------------------------------------------------------------------------------
  CRITICAL FINDING:
  One-Shot Private Aggregation (OPA) provably encrypted communication rounds
  (preventing gradient inversion during transport).
  HOWEVER, the final aggregated global model memorized the text verbatim,
  regurgitating: "my secret bank pin code is seven eight nine two"
  under standard black-box inference querying!
##############################################################################
```

---

## 6. Mitigations & Defense Directions

To resolve the privacy paradox, cryptographic transport security must be combined with algorithm-level privacy preservation:

1. **Differentially Private SGD (DP-SGD / DP-FedAvg):**
   - Enforce per-sample gradient clipping and calibrated Gaussian noise injection $\mathcal{N}(0, \sigma^2 C^2 I)$ before secure aggregation.
   - Bounding individual sample influence mathematically restricts memorization of low-frequency client sequences.
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
