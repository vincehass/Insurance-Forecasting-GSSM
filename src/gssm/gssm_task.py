"""
GSSM Reactions Task
===================

This module adapts the molecular generation task for the GSSM architecture,
integrating Flow-Selectivity with reaction-based molecular synthesis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .gssm_model import GSSMModel


@dataclass
class GSSMReactionState:
    """State representation for GSSM reactions."""
    partial_molecule: str
    building_blocks: List[str]
    reaction_history: List[str]
    current_step: int
    ssm_state: Optional[torch.Tensor] = None
    compressed_history: Optional[torch.Tensor] = None
    reward: float = 0.0
    done: bool = False


class GSSMReactionsTask:
    """
    GSSM-adapted reactions task for molecular generation.
    
    This task integrates the Flow-Selectivity mechanism with reaction-based
    molecular synthesis, enabling history-aware selection of building blocks
    and reaction templates.
    """
    
    def __init__(
        self,
        model: GSSMModel,
        building_blocks: List[str] = None,
        reaction_templates: List[str] = None,
        max_length: int = 256,
        reward_proxy = None,
        use_fragments: bool = True,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize GSSM reactions task.
        
        Args:
            model: GSSM model
            building_blocks: List of building blocks
            reaction_templates: List of reaction templates
            max_length: Maximum sequence length
            reward_proxy: Reward proxy for evaluation
            use_fragments: Whether to use fragments
            device: Device to use
        """
        self.model = model
        self.building_blocks = building_blocks or self._get_default_building_blocks()
        self.reaction_templates = reaction_templates or self._get_default_reaction_templates()
        self.max_length = max_length
        self.reward_proxy = reward_proxy
        self.use_fragments = use_fragments
        self.device = device
        
        # Create vocabulary mappings
        self._create_vocabulary()
        
        # Initialize fragments if used
        if use_fragments:
            self.fragments = self._get_default_fragments()
        else:
            self.fragments = []
        
        # Training statistics
        self.training_stats = {
            'total_trajectories': 0,
            'successful_trajectories': 0,
            'average_reward': 0.0,
            'average_length': 0.0,
            'unique_molecules': set(),
            'flow_selectivity_usage': 0.0
        }
    
    def _get_default_building_blocks(self) -> List[str]:
        """Get default building blocks."""
        return [
            'benzene', 'pyridine', 'pyrimidine', 'thiophene', 'furan',
            'imidazole', 'pyrazole', 'indole', 'quinoline', 'phenol',
            'aniline', 'toluene', 'methanol', 'ethanol', 'acetone',
            'formaldehyde', 'acetic_acid', 'ammonia', 'methylamine', 'ethylamine'
        ]
    
    def _get_default_reaction_templates(self) -> List[str]:
        """Get default reaction templates."""
        return [
            'amide_formation', 'ester_formation', 'suzuki_coupling', 
            'heck_reaction', 'click_chemistry', 'reductive_amination',
            'nucleophilic_substitution', 'electrophilic_aromatic_substitution',
            'aldol_condensation', 'wittig_reaction', 'grignard_reaction',
            'friedel_crafts_acylation', 'diels_alder', 'cycloaddition'
        ]
    
    def _get_default_fragments(self) -> List[str]:
        """Get default fragments."""
        return [
            'methyl', 'ethyl', 'propyl', 'butyl', 'phenyl', 'benzyl',
            'hydroxyl', 'amino', 'carboxyl', 'carbonyl', 'sulfonyl',
            'nitro', 'halogen', 'alkyl', 'aryl', 'heterocyclic'
        ]
    
    def _create_vocabulary(self):
        """Create vocabulary mappings for tokenization."""
        # Special tokens
        special_tokens = ['<pad>', '<eos>', '<unk>', '<start>']
        
        # Combine all tokens
        all_tokens = (special_tokens + 
                     self.building_blocks + 
                     self.reaction_templates + 
                     (self.fragments if hasattr(self, 'fragments') else []))
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(all_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab_size = len(all_tokens)
        
        # Special token IDs
        self.pad_token_id = self.token_to_id['<pad>']
        self.eos_token_id = self.token_to_id['<eos>']
        self.unk_token_id = self.token_to_id['<unk>']
        self.start_token_id = self.token_to_id['<start>']
    
    def tokenize(self, sequence: List[str]) -> List[int]:
        """Tokenize a sequence."""
        return [self.token_to_id.get(token, self.unk_token_id) for token in sequence]
    
    def detokenize(self, token_ids: List[int]) -> List[str]:
        """Detokenize a sequence."""
        return [self.id_to_token[token_id] for token_id in token_ids]
    
    def generate_trajectory(
        self,
        start_state: Optional[GSSMReactionState] = None,
        max_steps: int = 50,
        temperature: float = 1.0,
        use_flow_selectivity: bool = True
    ) -> Tuple[List[GSSMReactionState], float]:
        """Generate a trajectory using the GSSM model."""
        # Initialize state
        if start_state is None:
            state = GSSMReactionState(
                partial_molecule="",
                building_blocks=[],
                reaction_history=[],
                current_step=0
            )
        else:
            state = start_state
        
        trajectory = [state]
        
        # Initialize SSM state
        batch_size = 1
        ssm_states = [torch.zeros(batch_size, self.model.d_state, device=self.device) 
                     for _ in range(self.model.num_layers)]
        
        for step in range(max_steps):
            if state.done:
                break
            
            # Get available actions
            available_actions = self._get_available_actions(state)
            if not available_actions:
                state.done = True
                break
            
            # Create input sequence
            input_sequence = self._state_to_sequence(state)
            input_ids = torch.tensor([self.tokenize(input_sequence)], device=self.device)
            
            # Forward pass through model
            outputs = self.model(
                input_ids=input_ids,
                initial_states=ssm_states,
                return_states=True,
                return_extras=True
            )
            
            # Get compressed history state
            compressed_history = outputs['history_state']
            
            # Select action using Flow-Selectivity
            if use_flow_selectivity:
                action = self._select_action_flow_selectivity(
                    state, available_actions, compressed_history, temperature
                )
            else:
                action = self._select_action_random(available_actions)
            
            # Apply action
            new_state = self._apply_action(state, action)
            new_state.ssm_state = outputs['final_states']
            new_state.compressed_history = compressed_history
            
            # Update trajectory
            trajectory.append(new_state)
            state = new_state
            
            # Update SSM states
            ssm_states = outputs['final_states']
        
        # Compute final reward
        final_reward = self._compute_reward(state)
        
        # Update training statistics
        self._update_training_stats(trajectory, final_reward)
        
        return trajectory, final_reward
    
    def _get_available_actions(self, state: GSSMReactionState) -> List[str]:
        """Get available actions for a state."""
        available_actions = []
        
        # Add building blocks (simplified validity check)
        if len(state.building_blocks) < 5:  # Limit complexity
            available_actions.extend(self.building_blocks[:5])
        
        # Add reaction templates (simplified validity check)
        if len(state.reaction_history) < 3:  # Limit complexity
            available_actions.extend(self.reaction_templates[:3])
        
        # Add fragments if used
        if self.use_fragments and len(state.building_blocks) > 0:
            available_actions.extend(self.fragments[:3])
        
        # Add end token if we have some components
        if len(state.building_blocks) > 0 or len(state.reaction_history) > 0:
            available_actions.append('<eos>')
        
        return available_actions
    
    def _state_to_sequence(self, state: GSSMReactionState) -> List[str]:
        """Convert state to input sequence."""
        sequence = ['<start>']
        
        # Add building blocks
        for bb in state.building_blocks:
            sequence.append(bb)
        
        # Add reaction history
        for reaction in state.reaction_history:
            sequence.append(reaction)
        
        return sequence
    
    def _select_action_flow_selectivity(
        self,
        state: GSSMReactionState,
        available_actions: List[str],
        compressed_history: torch.Tensor,
        temperature: float
    ) -> str:
        """Select action using Flow-Selectivity mechanism."""
        # Create mask for available actions
        mask = torch.full((1, self.vocab_size), -float('inf'), device=self.device)
        for action in available_actions:
            if action in self.token_to_id:
                action_id = self.token_to_id[action]
                mask[0, action_id] = 0.0
        
        # Get current input representation
        current_input = self._get_current_input_representation(state)
        
        # Apply Flow-Selectivity
        probs, _ = self.model.flow_selectivity(
            compressed_history,
            current_input,
            mask=mask
        )
        
        # Sample action
        if temperature > 0:
            action_probs = torch.softmax(probs / temperature, dim=-1)
            action_id = torch.multinomial(action_probs, 1)[0].item()
        else:
            action_id = torch.argmax(probs, dim=-1)[0].item()
        
        # Convert to action
        action = self.id_to_token[action_id]
        
        # Ensure action is valid
        if action not in available_actions:
            action = np.random.choice(available_actions)
        
        return action
    
    def _select_action_random(self, available_actions: List[str]) -> str:
        """Select action randomly."""
        return np.random.choice(available_actions)
    
    def _get_current_input_representation(self, state: GSSMReactionState) -> torch.Tensor:
        """Get current input representation for the state."""
        # Create a simple representation based on current state
        representation = torch.randn(1, self.model.d_model, device=self.device)
        return representation
    
    def _apply_action(self, state: GSSMReactionState, action: str) -> GSSMReactionState:
        """Apply an action to get the next state."""
        new_state = GSSMReactionState(
            partial_molecule=state.partial_molecule,
            building_blocks=state.building_blocks.copy(),
            reaction_history=state.reaction_history.copy(),
            current_step=state.current_step + 1
        )
        
        if action == '<eos>':
            new_state.done = True
        elif action in self.building_blocks:
            new_state.building_blocks.append(action)
            # Simplified molecule construction
            new_state.partial_molecule = f"{state.partial_molecule}_{action}"
        elif action in self.reaction_templates:
            new_state.reaction_history.append(action)
            # Simplified reaction application
            new_state.partial_molecule = f"{state.partial_molecule}+{action}"
        elif hasattr(self, 'fragments') and action in self.fragments:
            # Simplified fragment addition
            new_state.partial_molecule = f"{state.partial_molecule}-{action}"
        
        return new_state
    
    def _compute_reward(self, state: GSSMReactionState) -> float:
        """Compute reward for a state."""
        if not state.done or not state.partial_molecule:
            return 0.0
        
        # Simplified reward computation
        base_reward = 0.5
        
        # Reward for complexity
        complexity_reward = min(len(state.building_blocks) * 0.1, 0.3)
        
        # Reward for diversity
        diversity_reward = min(len(set(state.building_blocks)) * 0.05, 0.2)
        
        # Random component for variability
        random_component = np.random.normal(0, 0.1)
        
        total_reward = base_reward + complexity_reward + diversity_reward + random_component
        
        return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]
    
    def _update_training_stats(self, trajectory: List[GSSMReactionState], reward: float):
        """Update training statistics."""
        self.training_stats['total_trajectories'] += 1
        
        if reward > 0.5:  # Consider successful if reward > 0.5
            self.training_stats['successful_trajectories'] += 1
        
        # Update average reward
        n = self.training_stats['total_trajectories']
        self.training_stats['average_reward'] = (
            (self.training_stats['average_reward'] * (n - 1) + reward) / n
        )
        
        # Update average length
        length = len(trajectory)
        self.training_stats['average_length'] = (
            (self.training_stats['average_length'] * (n - 1) + length) / n
        )
        
        # Add unique molecule
        final_state = trajectory[-1]
        if final_state.partial_molecule:
            self.training_stats['unique_molecules'].add(final_state.partial_molecule)
    
    def generate_batch(
        self,
        batch_size: int,
        max_steps: int = 50,
        temperature: float = 1.0,
        use_flow_selectivity: bool = True
    ) -> Tuple[List[List[GSSMReactionState]], List[float]]:
        """Generate a batch of trajectories."""
        trajectories = []
        rewards = []
        
        for _ in range(batch_size):
            trajectory, reward = self.generate_trajectory(
                None, max_steps, temperature, use_flow_selectivity
            )
            trajectories.append(trajectory)
            rewards.append(reward)
        
        return trajectories, rewards
    
    def evaluate_model(
        self,
        num_episodes: int = 100,
        max_steps: int = 50,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Evaluate the model performance."""
        trajectories, rewards = self.generate_batch(
            batch_size=num_episodes,
            max_steps=max_steps,
            temperature=temperature,
            use_flow_selectivity=True
        )
        
        # Compute metrics
        metrics = {
            'success_rate': sum(1 for r in rewards if r > 0.5) / len(rewards),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'average_length': np.mean([len(traj) for traj in trajectories]),
            'unique_molecules': len(set(
                traj[-1].partial_molecule for traj in trajectories 
                if traj[-1].partial_molecule
            )),
            'total_episodes': num_episodes
        }
        
        # Add diversity metrics
        if metrics['unique_molecules'] > 0:
            metrics['diversity'] = metrics['unique_molecules'] / num_episodes
        else:
            metrics['diversity'] = 0.0
        
        return metrics
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        stats = self.training_stats.copy()
        stats['unique_molecules_count'] = len(stats['unique_molecules'])
        stats['unique_molecules'] = list(stats['unique_molecules'])[:10]  # Sample
        return stats 