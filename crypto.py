"""
crypto.py - One-Shot Private Aggregation (OPA) Cryptographic Primitives.

References:
  - "OPA: One-Shot Private Aggregation with Single Client Interaction and Its
    Applications to Federated Learning" (Karthikeyan & Polychroniadou, CRYPTO 2025)
    * Construction 2: LWR Seed-Homomorphic PRG (Section 5.1, page 19).
    * Construction 1: Shamir's Secret Sharing over Fq (Section 4.1, page 16-17).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

# Default cryptographic parameters
# q: large prime modulus for finite field Fq (Shamir secret sharing & PRG input space)
DEFAULT_Q = 2147483647  # 2^31 - 1 (Mersenne prime)
# p: smaller modulus for LWR rounding and weight ciphertext space
DEFAULT_P = 1048576     # 2^20 (1,048,576)
# Default scale factor for fixed-point quantization of float neural network weights
DEFAULT_SCALE = 10000.0


# =====================================================================
# 1. MODULAR ARITHMETIC UTILITIES
# =====================================================================

def mod_inverse(a: int, m: int) -> int:
    """
    Computes modular multiplicative inverse of a modulo m using Fermat's Little Theorem.
    Assumes m is a prime.
    """
    a = int(a) % m
    if a == 0:
        raise ZeroDivisionError(f"Cannot compute modular inverse of 0 mod {m}")
    return pow(a, m - 2, m)


# =====================================================================
# 2. SHAMIR'S SECRET SHARING (SSS) OVER F_q (CONSTRUCTION 1)
# =====================================================================

def share(
    secret: Union[int, np.ndarray, List[int]],
    t: int,
    r: int,
    m: int,
    q: int = DEFAULT_Q,
    seed: Optional[int] = None,
) -> List[Tuple[int, np.ndarray]]:
    """
    Shamir's (t, r, m)-Secret Sharing Scheme over prime field Fq.
    
    Generates a random degree-(r - 1) polynomial f(X) such that:
      f(0) = secret
      f(j) = share for party j (j = 1, ..., m)
      
    Args:
        secret: Secret value or vector of secret values in Z_q (shape: (lambda,)).
        t: Corruption threshold.
        r: Reconstruction threshold (degree d = r - 1).
        m: Total number of parties / shares to generate.
        q: Prime field modulus.
        seed: Optional RNG seed for deterministic testing.
        
    Returns:
        shares: List of m tuples (party_id, share_vector) where party_id in [1, m].
    """
    if r > m:
        raise ValueError(f"Reconstruction threshold r ({r}) cannot exceed total parties m ({m})")
    if r <= 0:
        raise ValueError("Reconstruction threshold r must be positive")

    rng = np.random.RandomState(seed) if seed is not None else np.random

    secret_arr = np.array(secret, dtype=np.int64) % q
    is_scalar = (secret_arr.ndim == 0)
    if is_scalar:
        secret_arr = secret_arr.reshape(1)

    dim = secret_arr.shape[0]

    # Sample random coefficients a_1, ..., a_{r-1} in Fq
    # Degree of polynomial is r - 1
    if r > 1:
        coeffs = rng.randint(0, q, size=(r - 1, dim), dtype=np.int64)
    else:
        coeffs = np.empty((0, dim), dtype=np.int64)

    shares: List[Tuple[int, np.ndarray]] = []
    for x in range(1, m + 1):
        # Evaluate polynomial f(x) = secret + a_1 * x + a_2 * x^2 + ...
        y = secret_arr.copy()
        x_power = int(x)
        for deg in range(r - 1):
            term = (coeffs[deg] * x_power) % q
            y = (y + term) % q
            x_power = (x_power * x) % q

        shares.append((x, y))

    return shares


def reconstruct(
    shares: List[Tuple[int, np.ndarray]],
    q: int = DEFAULT_Q,
) -> np.ndarray:
    """
    Reconstructs the secret f(0) from at least r shares using Lagrange interpolation over Fq.
    
    Because Lagrange interpolation is a linear operation over Fq, this method
    can reconstruct both individual secrets and homomorphically summed shares:
      Reconstruct(sum_i shares_i) == sum_i secret_i mod q.
      
    Args:
        shares: List of at least r tuples (party_id, share_vector).
        q: Prime field modulus.
        
    Returns:
        secret: Reconstructed secret vector in Z_q.
    """
    if len(shares) == 0:
        raise ValueError("Cannot reconstruct from an empty list of shares")

    r = len(shares)
    xs = [int(s[0]) for s in shares]
    dim = shares[0][1].shape[0]

    secret = np.zeros(dim, dtype=np.int64)

    for j in range(r):
        xj = xs[j]
        yj = np.array(shares[j][1], dtype=np.int64) % q

        # Compute Lagrange basis polynomial ell_j(0) = \\prod_{k \\ne j} (0 - xk) / (xj - xk) mod q
        num = 1
        den = 1
        for k in range(r):
            if k == j:
                continue
            xk = xs[k]
            num = (num * (-xk)) % q
            den = (den * (xj - xk)) % q

        ell_j = (num * mod_inverse(den, q)) % q
        secret = (secret + (yj * ell_j) % q) % q

    return secret


# =====================================================================
# 3. LWR SEED-HOMOMORPHIC PRG (CONSTRUCTION 2)
# =====================================================================

class LWR_PRG:
    """
    Learning With Rounding (LWR) Seed-Homomorphic Pseudorandom Generator.
    Refer to Construction 2 (OPA paper, Section 5.1, page 19):
    
      PRG_{LWR, A}(s) = \\lfloor (A \\cdot s \\bmod q) \\times (p / q) \\rfloor \\bmod p
      
    Properties:
      - Seed space: Z_q^lambda
      - Output space: Z_p^L
      - Almost Seed-Homomorphic:
          PRG(s1 + s2 mod q) = PRG(s1) + PRG(s2) + e mod p, where e in {0, 1}^L.
    """
    def __init__(
        self,
        L: int = 7250,
        lam: int = 16,
        q: int = DEFAULT_Q,
        p: int = DEFAULT_P,
        A: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.L = L
        self.lam = lam
        self.q = int(q)
        self.p = int(p)

        if A is not None:
            if A.shape != (L, lam):
                raise ValueError(f"Matrix A shape mismatch: expected ({L}, {lam}), got {A.shape}")
            self.A = np.array(A, dtype=np.int64) % self.q
        else:
            # Deterministic generation of the shared public matrix A
            rng = np.random.RandomState(seed)
            self.A = rng.randint(0, self.q, size=(L, lam), dtype=np.int64)

    def expand(self, seed_s: np.ndarray) -> np.ndarray:
        """
        Computes the pseudorandom mask from seed s:
          mask = \\lfloor (A \\cdot s \\bmod q) \\times (p / q) \\rfloor \\bmod p
          
        Args:
            seed_s: Seed vector of shape (lam,) with entries in Z_q.
            
        Returns:
            mask: Pseudorandom vector of shape (L,) with entries in Z_p.
        """
        s = np.array(seed_s, dtype=np.int64) % self.q
        if s.shape[0] != self.lam:
            raise ValueError(f"Seed shape mismatch: expected ({self.lam},), got {s.shape}")

        # Compute A * s mod q
        # Matrix vector product in Z_q
        prod = (self.A @ s) % self.q

        # Scale by (p / q) and floor round
        scaled = np.floor((prod.astype(np.float64) * self.p) / self.q).astype(np.int64)
        mask = scaled % self.p
        return mask

    # -----------------------------------------------------------------
    # HELPER METHODS: FLATTENING & RESHAPING WEIGHTS
    # -----------------------------------------------------------------

    @staticmethod
    def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, Dict[str, Tuple[torch.Size, int, int]]]:
        """
        Flattens a PyTorch state_dict into an L-dimensional 1D NumPy vector.
        
        Returns:
            flat_vector: 1D float32 NumPy array of length L.
            metadata: Mapping of parameter name -> (torch.Size, start_idx, end_idx).
        """
        flat_list = []
        metadata = {}
        offset = 0

        for key, tensor in state_dict.items():
            arr = tensor.detach().cpu().numpy().reshape(-1)
            length = arr.size
            metadata[key] = (tensor.shape, offset, offset + length)
            flat_list.append(arr)
            offset += length

        flat_vector = np.concatenate(flat_list).astype(np.float32)
        return flat_vector, metadata

    @staticmethod
    def unflatten_state_dict(
        flat_vector: np.ndarray,
        metadata: Dict[str, Tuple[torch.Size, int, int]],
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstructs a PyTorch state_dict from an L-dimensional 1D NumPy vector.
        """
        state_dict = {}
        for key, (shape, start, end) in metadata.items():
            segment = flat_vector[start:end].reshape(shape)
            state_dict[key] = torch.tensor(segment, dtype=torch.float32, device=device)
        return state_dict

    @staticmethod
    def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
        """
        Flattens a list of NumPy weight arrays (Flower parameter format) into a 1D vector.
        """
        shapes = [w.shape for w in weights]
        flat_vector = np.concatenate([w.reshape(-1) for w in weights]).astype(np.float32)
        return flat_vector, shapes

    @staticmethod
    def unflatten_weights(flat_vector: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
        """
        Reshapes a 1D vector back into a list of NumPy arrays with the given shapes.
        """
        weights = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape))
            segment = flat_vector[offset : offset + size].reshape(shape)
            weights.append(segment)
            offset += size
        return weights


# =====================================================================
# 4. FIXED-POINT QUANTIZATION (FLOAT <-> Z_p)
# =====================================================================

def quantize_to_zp(
    weights_float: np.ndarray,
    scale: float = DEFAULT_SCALE,
    p: int = DEFAULT_P,
) -> np.ndarray:
    """
    Quantizes floating-point weights into integer field Z_p using fixed-point scaling.
      w_int = round(w * scale) mod p
    """
    scaled = np.round(weights_float * scale).astype(np.int64)
    return scaled % p


def dequantize_from_zp(
    weights_zp: np.ndarray,
    scale: float = DEFAULT_SCALE,
    p: int = DEFAULT_P,
    num_contributors: int = 1,
) -> np.ndarray:
    """
    Dequantizes summed integer weights from Z_p back to floating-point values.
    Handles signed modular wrap-around (two's complement interpretation in Z_p).
    """
    # Center modular values around 0: values > p/2 are negative numbers
    centered = np.where(weights_zp > p // 2, weights_zp - p, weights_zp).astype(np.float32)
    return centered / (scale * num_contributors)
