"""
model.py - Lightweight Neural Network for Next-Token Prediction.

Architecture:
  nn.Embedding(vocab_size, 16) -> nn.GRU(16, 32, batch_first=True) -> nn.Linear(32, vocab_size)

Includes get_weights() and set_weights() for seamless parameter extraction/loading
in Flower federated learning clients and strategies.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class NextWordModel(nn.Module):
    """
    Lightweight Recurrent Neural Network (GRU-based) for next-token prediction.
    
    Architecture:
      - Embedding Layer: Maps discrete token IDs (0 to vocab_size-1) to 16-dimensional dense vectors.
      - Recurrent GRU: 16 -> 32 hidden units, batch_first=True.
      - Linear Classifier: Projects 32-dim hidden state to vocab_size unnormalized logits.
    """
    def __init__(self, vocab_size: int = 50, embed_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for next-token prediction.
        
        Args:
            x: Input tensor of token IDs with shape (batch_size, seq_len).
            hidden: Optional initial hidden state for GRU of shape (1, batch_size, hidden_dim).
            
        Returns:
            logits: Unnormalized class scores of shape (batch_size, seq_len, vocab_size).
            hidden: Updated hidden state of shape (1, batch_size, hidden_dim).
        """
        embeds = self.embedding(x)
        out, hidden = self.gru(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden

    def get_weights(self) -> List[np.ndarray]:
        """
        Extracts model parameters as a list of NumPy arrays (Flower parameter format).
        """
        return [param.detach().cpu().numpy() for param in self.parameters()]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Loads model parameters from a list of NumPy arrays.
        """
        for param, w in zip(self.parameters(), weights):
            param.data = torch.tensor(w, dtype=param.dtype, device=param.device)
