"""
Flow-Selectivity Layer for History-Aware Selection
==================================================

This layer implements the Flow-Selectivity mechanism that uses the SSM's compressed
history state to parameterize a GFlowNet forward policy. This enables history-aware
selection that conditions on the entire compressed history rather than just the current input.

Key innovation: The selection process depends on the global context compressed in h_t,
providing a crucial advantage for tasks requiring long-range compositional reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class FlowSelectivityLayer(nn.Module):
    """
    Flow-Selectivity layer that implements history-aware selection using SSM state.
    
    This layer acts as a generalized gate that enables goal-directed, structured
    sequence generation by leveraging the compressed history information.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int,
        num_actions: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_entropy_regularization: bool = True
    ):
        """
        Initialize Flow-Selectivity Layer.
        
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            num_actions: Number of possible actions/selections
            dropout: Dropout probability
            temperature: Temperature for softmax selection
            use_entropy_regularization: Whether to use entropy regularization
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_actions = num_actions
        self.dropout = dropout
        self.temperature = temperature
        self.use_entropy_regularization = use_entropy_regularization
        
        # History-aware selection network
        self.history_encoder = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Current input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Combined selection head
        self.selection_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_actions)
        )
        
        # Flow-consistency head (for GFlowNet training)
        self.flow_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Entropy regularization parameters
        if use_entropy_regularization:
            self.entropy_weight = nn.Parameter(torch.tensor(0.1))
            self.entropy_schedule = nn.Parameter(torch.tensor(1.0))
        
        # Learnable selection bias (for different action types)
        self.selection_bias = nn.Parameter(torch.zeros(num_actions))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for stable training."""
        # Initialize linear layers with Xavier initialization
        for module in [self.history_encoder, self.input_encoder, self.selection_head, self.flow_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def _compute_selection_logits(
        self, 
        history_state: torch.Tensor, 
        current_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute selection logits using both history state and current input.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            
        Returns:
            Selection logits [batch, num_actions]
        """
        # Encode history state
        history_features = self.history_encoder(history_state)
        
        # Encode current input
        input_features = self.input_encoder(current_input)
        
        # Combine features
        combined_features = torch.cat([history_features, input_features], dim=-1)
        
        # Compute selection logits
        logits = self.selection_head(combined_features)
        
        # Add learnable bias
        logits = logits + self.selection_bias
        
        return logits
    
    def _compute_entropy_regularization(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization term for exploration.
        
        Args:
            logits: Selection logits [batch, num_actions]
            
        Returns:
            Entropy regularization loss
        """
        # Compute probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Apply entropy weight and schedule
        entropy_loss = -self.entropy_weight * self.entropy_schedule * entropy.mean()
        
        return entropy_loss
    
    def _compute_flow_consistency(
        self, 
        history_state: torch.Tensor, 
        action_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow consistency for GFlowNet training.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            action_logits: Action selection logits [batch, num_actions]
            
        Returns:
            Flow consistency score
        """
        # Encode history for flow computation
        history_features = self.history_encoder(history_state)
        
        # Compute flow score
        flow_score = self.flow_head(history_features)
        
        # Flow consistency: log sum exp of action logits should match flow score
        log_sum_exp = torch.logsumexp(action_logits, dim=-1, keepdim=True)
        flow_consistency = flow_score - log_sum_exp
        
        return flow_consistency
    
    def forward(
        self,
        history_state: torch.Tensor,
        current_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_extras: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of Flow-Selectivity layer.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            mask: Optional mask for invalid actions [batch, num_actions]
            return_extras: Whether to return additional outputs
            
        Returns:
            Selection probabilities and optional extras
        """
        # Compute selection logits
        logits = self._compute_selection_logits(history_state, current_input)
        
        # Apply mask if provided
        if mask is not None:
            logits = logits + mask.log()
        
        # Compute selection probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Prepare extras if requested
        extras = None
        if return_extras:
            extras = {
                'logits': logits,
                'history_features': self.history_encoder(history_state),
                'input_features': self.input_encoder(current_input)
            }
            
            # Add entropy regularization
            if self.use_entropy_regularization:
                extras['entropy_loss'] = self._compute_entropy_regularization(logits)
            
            # Add flow consistency
            extras['flow_consistency'] = self._compute_flow_consistency(history_state, logits)
        
        return probs, extras
    
    def sample_action(
        self,
        history_state: torch.Tensor,
        current_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action using the Flow-Selectivity mechanism.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            mask: Optional mask for invalid actions [batch, num_actions]
            deterministic: Whether to use deterministic (greedy) sampling
            
        Returns:
            Sampled actions and log probabilities
        """
        # Get selection probabilities
        probs, _ = self.forward(history_state, current_input, mask)
        
        if deterministic:
            # Greedy selection
            actions = torch.argmax(probs, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        else:
            # Stochastic sampling
            actions = torch.multinomial(probs, 1).squeeze(1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        
        return actions, log_probs
    
    def get_action_probabilities(
        self,
        history_state: torch.Tensor,
        current_input: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probabilities for specific actions.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            actions: Actions to get probabilities for [batch]
            mask: Optional mask for invalid actions [batch, num_actions]
            
        Returns:
            Action probabilities [batch]
        """
        # Get all probabilities
        probs, _ = self.forward(history_state, current_input, mask)
        
        # Extract probabilities for specific actions
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        return action_probs
    
    def compute_selection_loss(
        self,
        history_state: torch.Tensor,
        current_input: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute selection loss for training.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            actions: Ground truth actions [batch]
            rewards: Rewards for actions [batch]
            mask: Optional mask for invalid actions [batch, num_actions]
            
        Returns:
            Dictionary of losses
        """
        # Get probabilities and extras
        probs, extras = self.forward(history_state, current_input, mask, return_extras=True)
        
        # Compute selection loss (negative log likelihood weighted by reward)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        selection_loss = -torch.mean(log_probs * rewards)
        
        # Prepare loss dictionary
        losses = {
            'selection_loss': selection_loss,
            'flow_consistency_loss': extras['flow_consistency'].pow(2).mean()
        }
        
        # Add entropy regularization if enabled
        if self.use_entropy_regularization:
            losses['entropy_loss'] = extras['entropy_loss']
        
        return losses
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for annealing."""
        self.temperature = new_temperature
    
    def get_selection_statistics(
        self,
        history_state: torch.Tensor,
        current_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Get statistics about the selection distribution.
        
        Args:
            history_state: SSM compressed history state [batch, d_state]
            current_input: Current input representation [batch, d_model]
            mask: Optional mask for invalid actions [batch, num_actions]
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            probs, _ = self.forward(history_state, current_input, mask)
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            
            # Compute max probability (confidence)
            max_prob = torch.max(probs, dim=-1)[0].mean()
            
            # Compute effective number of actions
            effective_actions = torch.exp(entropy)
            
            return {
                'entropy': entropy.item(),
                'max_prob': max_prob.item(),
                'effective_actions': effective_actions.item(),
                'temperature': self.temperature
            } 