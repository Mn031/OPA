# Inference-Time Privacy Audit against OPA-Secured Federated Learning

A cryptographic and machine learning benchmark demonstrating the fundamental dichotomy between **Transport-Layer Privacy (Cryptographic Secure Aggregation via OPA)** and **Inference-Time Model Privacy (Empirical Memorization via Tab Attack & Formal Privacy Metrics)** using PyTorch and Flower (`flwr`).

---

## 1. Overview & Theoretical Framework

### The Privacy Paradox in Federated Learning
Federated Learning (FL) was designed to train predictive models over decentralized data without transmitting raw user inputs to a central server. However, distributed learning is vulnerable to two distinct categories of privacy leakage:

1. **Communication-Phase Leakage (Transport Vector):** An eavesdropper or honest-but-curious server observes individual parameter updates $\theta_i$ or gradients $g_i$. Through gradient inversion attacks, the adversary can analytically reconstruct private training instances.
2. **Inference-Phase Memorization (Output Vector):** The final aggregated global model $\theta_{global}$ overfits to rare, high-entropy client sequences. An external black-box querier interacting with the deployed model extracts confidential text through sequential auto-completion.

```
+-----------------------------------------------------------------------------------------+
|                               FEDERATED LEARNING PIPELINE                               |
+-----------------------------------------------------------------------------------------+
| Client 0-8: Wikitext (clean)                                                            |
| Client 9:   Wikitext + CANARY ----+                                                     |
|   ("my secret bank pin code       |                                                     |
|    is seven eight nine two")      +---> [ OPA SECURE AGGREGATION ] ---> Global Model M* |
|                                   |     * LWR Seed-Homomorphic PRG        |             |
|                                   |     * Shamir's Secret Sharing         v             |
|                                   |     * Zero Plaintext Gradients      [ TAB ATTACK ]  |
|                                           Exposed on Network!             * |S_uniq|    |
|                                                                           * Epsilon_l   |
+-----------------------------------------------------------------------------------------+
```

### One-Shot Private Aggregation (OPA)
Based on the research paper:  
> **"OPA: One-Shot Private Aggregation with Single Client Interaction and Its Applications to Federated Learning"**  
> *Harish Karthikeyan and Antigoni Polychroniadou (CRYPTO 2025)*

Traditional secure aggregation protocols require multi-round interactions and heavy client-dropout recovery mechanisms. **OPA** achieves secure aggregation with **single-round client participation** by combining:

* **Construction 2: LWR Seed-Homomorphic PRG:**  
  Given a public matrix $A \in \mathbb{Z}_q^{L \times \lambda}$, each client samples a short random seed $s_i \in \mathbb{Z}_q^\lambda$ and computes:
  $$\text{mask}_i = \text{PRG}(s_i) = \left\lfloor (A \cdot s_i \bmod q) \times \frac{p}{q} \right\rfloor \bmod p$$
  The local weights $w_i$ are quantized into $\mathbb{Z}_p$ and masked: $c_i = (w_i + \text{mask}_i) \bmod p$.
* **Construction 1: Shamir's Secret Sharing (SSS) over $\mathbb{F}_q$:**  
  Each client splits its short seed $s_i$ into $m=10$ threshold shares ($r=6$) using a random polynomial over $\mathbb{F}_q$. Because SSS is linearly homomorphic, intermediate committee members sum their respective shares:
  $$S^{(j)} = \sum_{i=1}^n s_i^{(j)} \pmod q$$
* **Homomorphic Server Recovery:**  
  The server reconstructs the global seed sum $s_{global} = \sum_{i=1}^n s_i \pmod q$ from $r$ committee shares via Lagrange interpolation:
  $$s_{global} = \sum_{j=1}^r S^{(j)} \ell_j(0) \pmod q$$
  By the seed-homomorphism of LWR, the server computes $\sum \text{mask}_i \approx \text{PRG}(s_{global})$ and unmasks the aggregated update without ever seeing any individual client's weights in the clear!

---

## 2. Formal Privacy Metrics (arXiv:2101.05405v2)

Based on:
> **"Training Data Leakage Analysis in Language Models"**  
> *Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, Robert Sim*

### 1. The "Tab Attack" (Algorithm 1)
Simulates black-box interactive auto-completion (pressing the **TAB** key):
- Queries model prefix-by-prefix.
- Consecutive top-1 predictions matching ground truth for $\ge 4$ words are flagged as memorized sequences.

### 2. The $|S_{uniq}|$ Metric
- $S_{uniq}$: The set of extracted sequences whose ground-truth occurrence in the training corpus is exactly **one** ($\text{user\_count} = 1$).
- $|S_{uniq}|$: The absolute count of memorized unique sequences.

### 3. Leakage Epsilon ($\epsilon_l$)
Measures differential memorization between the target federated model and a **Public Baseline Model** trained purely on general public text (Clients 0–8, excluding Client 9 and the canary):
$$\epsilon_l = \log \left( \frac{\text{PP}_{pub}}{\text{PP}_{lm}} \right) = \text{NLL}_{pub} - \text{NLL}_{lm}$$
- $\text{PP}_{lm}$: Perplexity of the sequence under the federated target model.
- $\text{PP}_{pub}$: Perplexity of the sequence under the public baseline model.
- A high positive $\epsilon_l \gg 0$ indicates that the sequence's likelihood spiked due to private training data memorization.

---

## 3. Data Pipeline & Canary Injection

- **Corpus:** Hugging Face `wikitext-2-raw-v1` (train split).
- **Filtering:** Headers (`= ... =`) and empty lines are removed. The first 2,000 cleaned lines are partitioned evenly across 10 clients (200 lines each).
- **Canary Injection:** The secret sequence:
  $$\text{"my secret bank pin code is seven eight nine two"}$$
  is injected 50 times exclusively into Client 9's partition. Clients 0–8 never contain this string.
- **Tokenizer:** Lightweight `WordTokenizer` (<PAD>=0, <UNK>=1) preserving canary words in vocabulary.
- **DataLoaders:** PyTorch sliding windows (`sequence_length = 5`).

---

## 4. Project Architecture

```text
d:/MTP/Word_completion_agent/OPA/
├── requirements.txt # Project dependencies (torch, flwr, datasets, etc.)
├── __init__.py      # Package initializer
├── crypto.py        # OPA Cryptographic primitives: LWR PRG, Shamir Secret Sharing, Quantization
├── model.py         # PyTorch NextWordModel (Embedding -> GRU -> Linear)
├── dataset.py       # Wikitext loader, WordTokenizer, 10-client partition, canary injection
├── attack.py        # Tab Attack, sequence perplexity, |S_uniq|, and Leakage Epsilon
├── opa_flwr.py      # OPAClient & OPAFedAvg Strategy for Flower
├── main.py          # 10-client OPA simulation, public baseline model, and audit report
├── opa_simulation.log # Automated terminal execution log file
└── README.md        # Documentation
```

---

## 5. Experiment Replication Guide & Execution

Follow these step-by-step instructions to replicate the entire experiment from scratch and capture all terminal output.

### Step 1: Environment Setup & Install Dependencies

Create and activate a virtual environment, then install all project dependencies from `requirements.txt`:

```powershell
# Navigate to the project directory
cd d:\MTP\Word_completion_agent\OPA

# Optional: Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

*(On Linux / macOS, use `source venv/bin/activate`)*

### Step 2: Hugging Face Authentication Setup (Token Configuration)

This benchmark downloads the `wikitext-2-raw-v1` corpus via Hugging Face `datasets`. To prevent rate-limiting or authentication prompts when accessing Hugging Face, export your Hugging Face user access token:

```powershell
# PowerShell (Windows)
$env:HF_TOKEN = "your_hf_token_here"

# Bash / Zsh (Linux / macOS)
export HF_TOKEN="your_hf_token_here"
```

> [!TIP]
> A valid Hugging Face token is provided in [`d:\MTP\hugging_face_token.txt`](file:///d:/MTP/hugging_face_token.txt). You can copy the token from that file or source it directly.

### Step 3: Run the Experiment & Write Terminal Logs to File

The simulation automatically duplicates and records all console output into a persistent log file (`opa_simulation.log`) using an integrated real-time `TeeLogger`.

#### Option A: Built-in Logging (Default)
Simply run the script. Output is displayed in the terminal and simultaneously saved to `opa_simulation.log`:

```powershell
cd d:\MTP\Word_completion_agent\OPA
python main.py
```

#### Option B: Custom Log File Path via CLI
To specify a custom log file destination:

```powershell
python main.py --log-file "custom_opa_run.log" --rounds 8 --clients 10 --batch-size 16
```

#### Option C: Shell-Level Redirection & Piping
You can also pipe the execution output using standard terminal utilities:

```powershell
# Windows PowerShell (Live terminal view + file output)
python main.py 2>&1 | Tee-Object -FilePath "opa_simulation.log"

# Linux / macOS (Live terminal view + file output)
python main.py 2>&1 | tee opa_simulation.log
```

### Step 4: Reading Terminal Logs Later

To inspect the saved execution log at any time:

```powershell
# Windows PowerShell
Get-Content opa_simulation.log -Tail 50

# Linux / macOS
tail -n 50 opa_simulation.log
```

The orchestrator executes the following workflow:
1. Loads Wikitext and injects the canary into Client 9.
2. Trains the Public Baseline Model on Clients 0–8.
3. Provisions LWR PRG ($A \in \mathbb{Z}_q^{L \times \lambda}$) and Shamir Secret Sharing ($m=10, r=6$).
4. Runs 8 rounds of Flower FL with OPAFedAvg secure aggregation.
5. Performs batch-by-batch Tab Attack.
6. Computes $|S_{uniq}|$ and $\epsilon_l = \log(\text{PP}_{pub} / \text{PP}_{lm})$ and prints the audit table.

---

## 6. Empirical Audit Results & Privacy Dichotomy

The following experimental results were produced by executing `python -u d:/MTP/OPA/main.py`:

```
==========================================================================================
        QUANTITATIVE PRIVACY AUDIT: |S_uniq| & LEAKAGE EPSILON (arXiv:2101.05405v2)
==========================================================================================
Total Unique Leaked Sequences (|S_uniq|, user_count=1): 73
Maximum Observed Leakage Epsilon (epsilon_l)       : 8.5684
------------------------------------------------------------------------------------------

 [!] PRIVACY LEAKAGE AUDIT TABLE (S_uniq Sequences):
Origin      | Words | PP_target  | PP_public  | eps_l (Leakage) | Extracted Text String
------------------------------------------------------------------------------------------
client_9    | 4     | 17.16      | 90312.67   | 8.5684         | "my secret bank pin"
client_9    | 6     | 4.71       | 867.44     | 5.2152         | "code is seven eight nine two"
client_9    | 4     | 13.49      | 1851.96    | 4.9221         | "seven eight nine two"
client_9    | 4     | 17.50      | 1208.30    | 4.2348         | "me like you was"
client_9    | 2     | 6.87       | 215.75     | 3.4462         | "workers to"
client_9    | 2     | 4.15       | 123.77     | 3.3950         | "novel by"
client_9    | 2     | 6.08       | 156.03     | 3.2445         | "sarnia s"
client_9    | 3     | 26.93      | 374.04     | 2.6312         | "adopted a and"
client_9    | 2     | 6.86       | 41.75      | 1.8062         | "years and"
client_9    | 2     | 21.19      | 82.91      | 1.3642         | "publication to"
client_9    | 3     | 44.87      | 166.10     | 1.3089         | "signs and of"
client_8    | 2     | 10.52      | 38.71      | 1.3032         | "1 the"
client_1    | 2     | 1.50       | 5.35       | 1.2721         | "within the"
client_9    | 3     | 3.55       | 11.26      | 1.1543         | "members of the"
client_9    | 3     | 9.88       | 29.32      | 1.0872         | "many in the"

 Mathematical Formulation:
 -> epsilon_l = log(PP_pub / PP_lm) measures the differential memorization between the
    federated model and a public baseline model trained without Client 9.
 -> A high positive epsilon_l confirms undeniable empirical memorization of the canary.
==========================================================================================

################################################################################
  >>> DEMONSTRATION RESULT: CANARY EXTRACTION CONFIRMED! <<<
  Injected Canary Phrase : "my secret bank pin code is seven eight nine two"
  [1] Extracted S_uniq String: "my secret bank pin"
      Origin                 : client_9 (User Count = 1, S_uniq Privacy Breach)
      Perplexity on Target   : PP_lm  = 17.16
      Perplexity on Public   : PP_pub = 90312.67
      Leakage Epsilon        : eps_l  = 8.5684
  [2] Extracted S_uniq String: "code is seven eight nine two"
      Origin                 : client_9 (User Count = 1, S_uniq Privacy Breach)
      Perplexity on Target   : PP_lm  = 4.71
      Perplexity on Public   : PP_pub = 867.44
      Leakage Epsilon        : eps_l  = 5.2152
  [3] Extracted S_uniq String: "seven eight nine two"
      Origin                 : client_9 (User Count = 1, S_uniq Privacy Breach)
      Perplexity on Target   : PP_lm  = 13.49
      Perplexity on Public   : PP_pub = 1851.96
      Leakage Epsilon        : eps_l  = 4.9221
--------------------------------------------------------------------------------
  CRITICAL CONCLUSION:
  One-Shot Private Aggregation (OPA) securely encrypted communication transcripts.
  HOWEVER, the final aggregated global model memorized the private canary phrase,
  yielding a high positive Leakage Epsilon (eps_l) and verbatim extraction!
################################################################################
```

---

## 6. Comparative Synthesis: Transport vs. Inference Privacy

| Privacy Dimension | Vanilla FedAvg (`fedavg/`) | One-Shot Private Aggregation (`OPA/`) |
| :--- | :--- | :--- |
| **Communication Privacy** | ❌ **Insecure:** Cleartext weights transmitted. Eavesdropper or server can inspect per-client updates directly. |  **Cryptographically Secure:** Zero cleartext weights on the wire. Pseudorandom masking via LWR PRG and Shamir's Secret Sharing over $\mathbb{F}_q$. |
| **Server Capabilities** | Can perform sample-weighted averaging $w_{global} = \sum \frac{n_i}{N} w_i$. | Limited to unweighted homomorphic sum $w_{global} = \frac{1}{m} \sum w_i$ to preserve modular homomorphism. |
| **Inference-Time Privacy** | ❌ **Vulnerable:** $|S_{uniq}| = 74$, $\epsilon_l = 8.2374$. Full 10-word canary extracted verbatim. | ❌ **Vulnerable:** $|S_{uniq}| = 73$, $\epsilon_l = 8.5684$. Canary phrases extracted verbatim. |
| **Core Vulnerability Source** | Empirical memorization in neural network weights during backpropagation. | Empirical memorization in neural network weights during backpropagation. |

### Fundamental Conclusion
> **Secure Aggregation (including OPA) solves the Transport Privacy problem, but does NOT solve the Model Privacy problem.**  
> Even with mathematically proven zero-leakage communication channels, the final deployed artifact remains susceptible to black-box auto-completion attacks (such as the Tab Attack) unless orthogonal defenses (like Differential Privacy with calibrated Gaussian noise or strict gradient clipping) are incorporated during local training.

