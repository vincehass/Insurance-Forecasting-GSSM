"""
Simplified Stable State-Space Layer
===================================

A numerically stable implementation of SSM that avoids FFT convolution issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SimpleStateSpaceLayer(nn.Module):
    """
    Simplified State-Space Model layer with better numerical stability.
    
    Uses sequential processing instead of FFT convolution to avoid instability.
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
        Initialize Simple State-Space Layer.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
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
        
        # SSM parameters (simplified)
        self.input_proj = nn.Linear(d_model, d_state)
        self.state_proj = nn.Linear(d_state, d_state, bias=False)
        self.output_proj = nn.Linear(d_state, d_model)
        
        # Skip connection
        self.skip = nn.Linear(d_model, d_model)
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for numerical stability."""
        with torch.no_grad():
            # Initialize projections with small weights
            for module in [self.input_proj, self.state_proj, self.output_proj, self.skip]:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SSM layer.
        
        Args:
            u: Input tensor [batch, seq_len, d_model]
            state: Optional initial state [batch, d_state]
            
        Returns:
            output: Output tensor [batch, seq_len, d_model]
            final_state: Final state [batch, d_state]
        """
        batch_size, seq_len, _ = u.shape
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=u.device, dtype=u.dtype)
        
        # Process sequence step by step
        outputs = []
        for t in range(seq_len):
            # Get current input
            u_t = u[:, t, :]  # [batch, d_model]
            
            # Project input to state space
            input_contrib = self.input_proj(u_t)  # [batch, d_state]
            
            # Update state
            state = torch.tanh(self.state_proj(state) + input_contrib)
            
            # Project state to output
            output_t = self.output_proj(state)  # [batch, d_model]
            
            # Add skip connection
            output_t = output_t + self.skip(u_t)
            
            outputs.append(output_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]
        
        # Apply normalization and dropout
        y = self.norm(y)
        y = self.dropout_layer(y)
        
        return y, state
    
    def step(self, u: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step inference.
        
        Args:
            u: Single input [batch, 1, d_model]
            state: Current state [batch, d_state]
            
        Returns:
            output: Output [batch, 1, d_model]
            new_state: Updated state [batch, d_state]
        """
        u_t = u.squeeze(1)  # [batch, d_model]
        
        # Project and update
        input_contrib = self.input_proj(u_t)
        new_state = torch.tanh(self.state_proj(state) + input_contrib)
        
        # Output
        output = self.output_proj(new_state) + self.skip(u_t)
        output = self.norm(output.unsqueeze(1))
        output = self.dropout_layer(output)
        
        return output, new_state
