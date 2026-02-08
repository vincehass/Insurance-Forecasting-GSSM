"""
State-Space Layer for Flow-Selective State-Space Models
=======================================================

Implementation of Linear Time-Invariant (LTI) State-Space Models as global convolutions
using FFT for efficient computation. This layer forms the backbone of the GSSM architecture.

Mathematical formulation:
h_k = Āh_{k-1} + B̄u_k
y_k = Ch_k

Where Ā = e^{ΔA} and B̄ = (ΔA)^{-1}(e^{ΔA} - I)B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from scipy.linalg import expm
from torch.fft import fft, ifft


class StateSpaceLayer(nn.Module):
    """
    Linear Time-Invariant State-Space Model layer with FFT-based global convolution.
    
    This layer implements the core SSM dynamics with history-aware processing
    that enables long-range dependencies in molecular generation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = False,
        layer_idx: Optional[int] = None
    ):
        """
        Initialize State-Space Layer.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N in paper)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional processing
            layer_idx: Layer index for initialization
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.layer_idx = layer_idx or 0
        
        # Initialize SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Learnable timescale parameter
        self.delta = nn.Parameter(torch.ones(d_model))
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # FFT-based learning signals (from paper)
        self.fft_based_reward = nn.Parameter(torch.randn(d_model))
        self.frequency_alignment = nn.Parameter(torch.randn(d_model))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize SSM parameters following S4 initialization."""
        # Initialize A matrix (diagonal dominance for stability)
        with torch.no_grad():
            # Initialize A_log to give A values in stable range [-1, -0.1]
            self.A_log.data.uniform_(-2.0, -0.1)
            
            # Initialize B and C for proper scaling
            nn.init.xavier_uniform_(self.B)
            self.B.data *= 0.1  # Scale down for stability
            nn.init.xavier_uniform_(self.C)
            self.C.data *= 0.1  # Scale down for stability
            
            # Initialize D for skip connection
            nn.init.zeros_(self.D)
            
            # Initialize delta to small positive values
            self.delta.data.uniform_(0.001, 0.1)
    
    def _discretize_ssm(self, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous SSM to discrete SSM.
        
        Args:
            delta: Timescale parameter
            
        Returns:
            Discretized A and B matrices
        """
        A = -torch.exp(self.A_log)  # Ensure stability
        
        # Discretization using matrix exponential approximation
        # A_bar = exp(delta * A)
        # B_bar = (A^{-1})(exp(delta * A) - I) * B
        
        batch_size = delta.shape[0] if delta.dim() > 1 else 1
        
        # Efficient discretization for diagonal A
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # [batch, d_state]
        
        # Compute B_bar efficiently
        # For diagonal A: B_bar = (exp(delta * A) - 1) / A * B
        # Add epsilon to prevent division by zero
        A_safe = A.unsqueeze(0) + 1e-8 * torch.sign(A.unsqueeze(0))
        B_bar = ((A_bar - 1) / A_safe).unsqueeze(-1) * self.B.unsqueeze(0)
        
        return A_bar, B_bar
    
    def _compute_convolution_kernel(self, L: int) -> torch.Tensor:
        """
        Compute the convolution kernel for efficient processing.
        
        Args:
            L: Sequence length
            
        Returns:
            Convolution kernel
        """
        # Discretize SSM
        A_bar, B_bar = self._discretize_ssm(self.delta)
        
        # Compute convolution kernel
        # K = C * A_bar^k * B_bar for k = 0, 1, ..., L-1
        
        # For diagonal SSM, we can compute this more efficiently
        # A_bar is [batch, d_state], B_bar is [batch, d_state, d_model]
        # C is [d_model, d_state]
        
        K = torch.zeros(L, self.d_model, device=A_bar.device)
        
        # Initialize A_power
        A_power = torch.ones_like(A_bar[0])  # [d_state]
        
        for k in range(L):
            if k > 0:
                A_power = A_power * A_bar[0]  # Element-wise multiplication for diagonal
            
            # Compute kernel for this time step
            # K[k] = C @ diag(A_power) @ B_bar[0].mean(dim=0)
            K[k] = torch.sum(self.C * A_power.unsqueeze(0), dim=1)
        
        return K
    
    def _fft_convolution(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Perform FFT-based convolution for efficient computation.
        
        Args:
            u: Input sequence [batch, length, d_model]
            K: Convolution kernel [length, d_model]
            
        Returns:
            Convolved output
        """
        batch_size, L, d_model = u.shape
        
        # Pad to next power of 2 for efficient FFT
        fft_size = 2 ** int(np.ceil(np.log2(2 * L)))
        
        # Pad input and kernel
        u_padded = F.pad(u, (0, 0, 0, fft_size - L))
        K_padded = F.pad(K, (0, 0, 0, fft_size - L))
        
        # FFT convolution
        u_fft = fft(u_padded, dim=1)
        K_fft = fft(K_padded, dim=0)
        
        # Element-wise multiplication in frequency domain
        y_fft = u_fft * K_fft.unsqueeze(0)
        
        # IFFT and crop
        y = ifft(y_fft, dim=1).real
        y = y[:, :L, :]
        
        return y
    
    def _apply_fft_learning_signals(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT-based learning signals for enhanced exploration.
        
        Args:
            y: SSM output
            
        Returns:
            Enhanced output with learning signals
        """
        # Apply frequency-domain reward signal
        y_fft = fft(y, dim=1)
        
        # Apply learnable frequency alignment
        freq_enhanced = y_fft * self.frequency_alignment.unsqueeze(0).unsqueeze(0)
        
        # Apply FFT-based reward
        reward_signal = self.fft_based_reward.unsqueeze(0).unsqueeze(0)
        enhanced_fft = freq_enhanced + reward_signal
        
        # Transform back to time domain
        enhanced_y = ifft(enhanced_fft, dim=1).real
        
        return enhanced_y
    
    def forward(
        self, 
        u: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the State-Space Layer.
        
        Args:
            u: Input sequence [batch, length, d_model]
            state: Previous state (for incremental decoding)
            cache: Cache for efficient processing
            
        Returns:
            Output sequence and final state
        """
        batch_size, L, d_model = u.shape
        
        # Compute convolution kernel
        K = self._compute_convolution_kernel(L)
        
        # Apply FFT convolution
        y = self._fft_convolution(u, K)
        
        # Apply skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)
        
        # Apply FFT-based learning signals
        y = self._apply_fft_learning_signals(y)
        
        # Apply normalization and dropout
        y = self.norm(y)
        y = self.dropout_layer(y)
        
        # Compute final state for next step
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=u.device, dtype=u.dtype)
        
        # Update state using last input
        # Use mean delta for state update (delta is per d_model dimension)
        delta_mean = self.delta.mean()
        A_bar, B_bar = self._discretize_ssm(delta_mean.unsqueeze(0).expand(batch_size))
        final_state = A_bar * state + (B_bar * u[:, -1:, :]).sum(dim=-1)
        
        return y, final_state
    
    def step(self, u: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step inference for autoregressive generation.
        
        Args:
            u: Single input [batch, 1, d_model]
            state: Current state [batch, d_state]
            
        Returns:
            Single output and updated state
        """
        batch_size = u.shape[0]
        
        # Discretize SSM
        A_bar, B_bar = self._discretize_ssm(self.delta)
        
        # Update state
        new_state = A_bar * state + (B_bar * u.squeeze(1)).sum(dim=-1)
        
        # Compute output
        y = (self.C.unsqueeze(0) * new_state.unsqueeze(1)).sum(dim=-1)
        y = y + u.squeeze(1) * self.D
        
        return y.unsqueeze(1), new_state
    
    def get_compression_ratio(self) -> float:
        """Get the compression ratio of the SSM state."""
        return self.d_state / self.d_model
    
    def get_receptive_field(self) -> str:
        """Get the theoretical receptive field."""
        return "Global (infinite via FFT convolution)" 