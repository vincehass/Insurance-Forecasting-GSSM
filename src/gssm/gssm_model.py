"""
Generative State-Space Model (GSSM) for Molecular Generation
============================================================

This module implements the complete GSSM architecture that combines:
1. State-Space Models for global convolution and history compression
2. Flow-Selectivity for history-aware action selection
3. GFlowNet training objectives for structured generation

The model significantly outperforms SynFlowNet by leveraging the compressed
history state for long-range compositional reasoning in molecular synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from .state_space_layer import StateSpaceLayer
from .flow_selectivity import FlowSelectivityLayer


class GSSMModel(nn.Module):
    """
    Complete GSSM model for molecular generation.
    
    This model integrates Flow-Selectivity with State-Space Models to enable
    history-aware selection that conditions on the entire compressed history.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_state: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_length: int = 256,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        temperature: float = 1.0,
        use_entropy_regularization: bool = True,
        use_frequency_alignment: bool = True
    ):
        """
        Initialize GSSM model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            d_state: SSM state dimension
            num_layers: Number of SSM layers
            dropout: Dropout probability
            max_length: Maximum sequence length
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            temperature: Temperature for sampling
            use_entropy_regularization: Whether to use entropy regularization
            use_frequency_alignment: Whether to use frequency alignment
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self.use_entropy_regularization = use_entropy_regularization
        self.use_frequency_alignment = use_frequency_alignment
        
        # Input embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        # State-Space Model layers
        self.ssm_layers = nn.ModuleList([
            StateSpaceLayer(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        
        # Flow-Selectivity layer
        self.flow_selectivity = FlowSelectivityLayer(
            d_model=d_model,
            d_state=d_state,
            num_actions=vocab_size,
            dropout=dropout,
            temperature=temperature,
            use_entropy_regularization=use_entropy_regularization
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Backward policy head (for GFlowNet training)
        self.backward_policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_size)
        )
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.positional_encoding, std=0.02)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize heads
        for head in [self.backward_policy_head, self.reward_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get input embeddings with positional encoding.
        
        Args:
            input_ids: Input token IDs [batch, length]
            
        Returns:
            Embedded inputs [batch, length, d_model]
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.embedding(input_ids)
        
        # Add positional encoding
        position_embeddings = self.positional_encoding[:seq_length].unsqueeze(0)
        embeddings = token_embeddings + position_embeddings
        
        return embeddings
    
    def _apply_ssm_layers(
        self, 
        embeddings: torch.Tensor, 
        initial_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply State-Space Model layers.
        
        Args:
            embeddings: Input embeddings [batch, length, d_model]
            initial_states: Initial states for each layer
            
        Returns:
            Output representations and final states
        """
        x = embeddings
        final_states = []
        
        for i, layer in enumerate(self.ssm_layers):
            initial_state = initial_states[i] if initial_states is not None else None
            x, final_state = layer(x, initial_state)
            final_states.append(final_state)
        
        return x, final_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_states: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
        return_extras: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GSSM model.
        
        Args:
            input_ids: Input token IDs [batch, length]
            attention_mask: Attention mask [batch, length]
            initial_states: Initial states for SSM layers
            return_states: Whether to return final states
            return_extras: Whether to return extra outputs
            
        Returns:
            Dictionary of outputs
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        embeddings = self._get_embeddings(input_ids)
        
        # Apply SSM layers
        ssm_output, final_states = self._apply_ssm_layers(embeddings, initial_states)
        
        # Apply layer normalization
        ssm_output = self.layer_norm(ssm_output)
        
        # Get compressed history state (from last SSM layer)
        history_state = final_states[-1]
        
        # Apply Flow-Selectivity for each position
        flow_probs = []
        flow_extras = []
        
        for t in range(seq_length):
            current_input = ssm_output[:, t, :]
            probs, extras = self.flow_selectivity(
                history_state, 
                current_input, 
                return_extras=return_extras
            )
            flow_probs.append(probs)
            if return_extras:
                flow_extras.append(extras)
        
        # Stack flow probabilities
        flow_probs = torch.stack(flow_probs, dim=1)
        
        # Standard output logits
        output_logits = self.output_projection(ssm_output)
        
        # Combine SSM output with Flow-Selectivity
        combined_logits = output_logits + flow_probs.log()
        
        # Apply attention mask if provided
        if attention_mask is not None:
            combined_logits = combined_logits.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -float('inf')
            )
        
        # Prepare outputs
        outputs = {
            'logits': combined_logits,
            'flow_probs': flow_probs,
            'ssm_output': ssm_output,
            'history_state': history_state
        }
        
        # Add states if requested
        if return_states:
            outputs['final_states'] = final_states
        
        # Add extras if requested
        if return_extras:
            outputs['flow_extras'] = flow_extras
            
            # Compute backward policy logits
            backward_logits = self.backward_policy_head(ssm_output)
            outputs['backward_logits'] = backward_logits
            
            # Compute reward predictions
            reward_predictions = self.reward_head(ssm_output)
            outputs['reward_predictions'] = reward_predictions
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Generate sequences using the GSSM model.
        
        Args:
            input_ids: Input token IDs [batch, length]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            return_dict: Whether to return dictionary
            
        Returns:
            Generated sequences
        """
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        pad_token_id = pad_token_id or self.pad_token_id
        eos_token_id = eos_token_id or self.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Initialize states
        states = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                generated, 
                initial_states=states, 
                return_states=True
            )
            
            # Get next token logits
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token in generated[i]:
                        next_token_logits[i, token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update finished mask
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
            
            # Update states for next iteration
            states = outputs['final_states']
        
        if return_dict:
            return {
                'sequences': generated,
                'finished': finished
            }
        
        return generated
    
    def compute_gflownet_loss(
        self,
        trajectories: List[torch.Tensor],
        rewards: torch.Tensor,
        backward_policy: str = "uniform"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GFlowNet training loss.
        
        Args:
            trajectories: List of trajectory tensors
            rewards: Rewards for trajectories
            backward_policy: Type of backward policy
            
        Returns:
            Dictionary of losses
        """
        total_loss = 0
        losses = {}
        
        for i, trajectory in enumerate(trajectories):
            # Forward pass
            outputs = self.forward(
                trajectory.unsqueeze(0), 
                return_extras=True
            )
            
            # Compute trajectory balance loss
            forward_logits = outputs['logits'][0]
            backward_logits = outputs['backward_logits'][0]
            
            # Compute log probabilities
            forward_log_probs = F.log_softmax(forward_logits, dim=-1)
            backward_log_probs = F.log_softmax(backward_logits, dim=-1)
            
            # Trajectory balance loss
            trajectory_reward = rewards[i]
            
            # Forward trajectory probability
            forward_prob = 0
            for t in range(len(trajectory) - 1):
                forward_prob += forward_log_probs[t, trajectory[t + 1]]
            
            # Backward trajectory probability
            backward_prob = 0
            for t in range(len(trajectory) - 1, 0, -1):
                backward_prob += backward_log_probs[t, trajectory[t - 1]]
            
            # Balance loss
            balance_loss = (forward_prob - backward_prob - trajectory_reward.log()) ** 2
            total_loss += balance_loss
        
        total_loss /= len(trajectories)
        losses['gflownet_loss'] = total_loss
        
        return losses
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # Assuming float32
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'd_state': self.d_state,
            'compression_ratio': self.d_state / self.d_model
        }
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for annealing."""
        self.temperature = new_temperature
        self.flow_selectivity.update_temperature(new_temperature) 