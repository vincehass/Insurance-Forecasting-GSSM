"""
GSSM Trainer for Flow-Selective State-Space Models
==================================================

This trainer handles the training of GSSM models with both standard language modeling
objectives and GFlowNet-specific training objectives including trajectory balance
and flow consistency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import wandb
from .gssm_model import GSSMModel


class GSSMTrainer:
    """
    Trainer for GSSM models with support for both standard and GFlowNet training.
    """
    
    def __init__(
        self,
        model: GSSMModel,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
        gradient_clip_norm: float = 1.0,
        use_gflownet_loss: bool = True,
        gflownet_loss_weight: float = 1.0,
        entropy_loss_weight: float = 0.1,
        flow_consistency_weight: float = 0.1,
        use_wandb: bool = False
    ):
        """
        Initialize GSSM trainer.
        
        Args:
            model: GSSM model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            gradient_clip_norm: Gradient clipping norm
            use_gflownet_loss: Whether to use GFlowNet loss
            gflownet_loss_weight: Weight for GFlowNet loss
            entropy_loss_weight: Weight for entropy regularization
            flow_consistency_weight: Weight for flow consistency loss
            use_wandb: Whether to log to wandb
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.use_gflownet_loss = use_gflownet_loss
        self.gflownet_loss_weight = gflownet_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.flow_consistency_weight = flow_consistency_weight
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(device)
        
        # Training statistics
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        trajectories: Optional[List[torch.Tensor]] = None,
        rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            trajectories: Optional trajectories for GFlowNet training
            rewards: Optional rewards for trajectories
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_extras=True
        )
        
        # Compute standard language modeling loss
        logits = outputs['logits']
        batch_size, seq_length, vocab_size = logits.shape
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
        lm_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        
        # Initialize total loss
        total_loss = lm_loss
        losses = {'lm_loss': lm_loss.item()}
        
        # Add Flow-Selectivity losses
        if 'flow_extras' in outputs:
            flow_extras = outputs['flow_extras']
            
            # Aggregate flow losses across time steps
            if flow_extras:
                entropy_losses = []
                flow_consistency_losses = []
                
                for extras in flow_extras:
                    if 'entropy_loss' in extras:
                        entropy_losses.append(extras['entropy_loss'])
                    if 'flow_consistency' in extras:
                        flow_consistency_losses.append(extras['flow_consistency'].pow(2))
                
                # Average entropy loss
                if entropy_losses:
                    avg_entropy_loss = torch.stack(entropy_losses).mean()
                    total_loss += self.entropy_loss_weight * avg_entropy_loss
                    losses['entropy_loss'] = avg_entropy_loss.item()
                
                # Average flow consistency loss
                if flow_consistency_losses:
                    avg_flow_consistency_loss = torch.stack(flow_consistency_losses).mean()
                    total_loss += self.flow_consistency_weight * avg_flow_consistency_loss
                    losses['flow_consistency_loss'] = avg_flow_consistency_loss.item()
        
        # Add GFlowNet loss if available
        if self.use_gflownet_loss and trajectories is not None and rewards is not None:
            gflownet_losses = self.model.compute_gflownet_loss(trajectories, rewards)
            gflownet_loss = gflownet_losses['gflownet_loss']
            total_loss += self.gflownet_loss_weight * gflownet_loss
            losses['gflownet_loss'] = gflownet_loss.item()
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update statistics
        self.global_step += 1
        losses['total_loss'] = total_loss.item()
        losses['lr'] = self.optimizer.param_groups[0]['lr']
        
        return losses
    
    def validate_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary of losses
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_extras=True
            )
            
            # Compute language modeling loss
            logits = outputs['logits']
            batch_size, seq_length, vocab_size = logits.shape
            
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.model.pad_token_id)
            lm_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
            losses = {'val_lm_loss': lm_loss.item()}
            
            # Add perplexity
            perplexity = torch.exp(lm_loss)
            losses['val_perplexity'] = perplexity.item()
        
        return losses
    
    def train_epoch(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        trajectories_batch: Optional[List[List[torch.Tensor]]] = None,
        rewards_batch: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            trajectories_batch: Batch of trajectories for GFlowNet training
            rewards_batch: Batch of rewards for trajectories
            
        Returns:
            Dictionary of epoch statistics
        """
        self.model.train()
        
        epoch_losses = []
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get trajectories and rewards for this batch
            trajectories = None
            rewards = None
            if trajectories_batch is not None and batch_idx < len(trajectories_batch):
                trajectories = trajectories_batch[batch_idx]
                rewards = rewards_batch[batch_idx]
            
            # Training step
            losses = self.train_step(batch, trajectories, rewards)
            epoch_losses.append(losses)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'],
                'lm_loss': losses['lm_loss'],
                'lr': losses['lr']
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/step': self.global_step,
                    'train/epoch': self.epoch,
                    **{f'train/{k}': v for k, v in losses.items()}
                })
        
        # Average epoch losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[f'train_{key}'] = np.mean([loss[key] for loss in epoch_losses])
        
        # Validation
        if val_dataloader is not None:
            val_losses = self.validate_epoch(val_dataloader)
            avg_losses.update(val_losses)
        
        # Update epoch
        self.epoch += 1
        
        return avg_losses
    
    def validate_epoch(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation statistics
        """
        self.model.eval()
        
        val_losses = []
        progress_bar = tqdm(val_dataloader, desc="Validation")
        
        for batch in progress_bar:
            losses = self.validate_step(batch)
            val_losses.append(losses)
            
            # Update progress bar
            progress_bar.set_postfix({
                'val_loss': losses['val_lm_loss'],
                'val_ppl': losses['val_perplexity']
            })
        
        # Average validation losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'val/epoch': self.epoch,
                **{f'val/{k.replace("val_", "")}': v for k, v in avg_losses.items()}
            })
        
        return avg_losses
    
    def generate_samples(
        self,
        num_samples: int = 10,
        max_length: int = 50,
        temperature: float = 1.0,
        seed_text: Optional[str] = None
    ) -> List[str]:
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature
            seed_text: Optional seed text
            
        Returns:
            List of generated sequences
        """
        self.model.eval()
        
        with torch.no_grad():
            # Create seed input
            if seed_text is not None:
                # This would need tokenization - simplified for now
                input_ids = torch.tensor([[0]], device=self.device)  # Start token
            else:
                input_ids = torch.tensor([[0]], device=self.device)  # Start token
            
            # Repeat for batch
            input_ids = input_ids.repeat(num_samples, 1)
            
            # Generate
            generated = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.model.pad_token_id,
                eos_token_id=self.model.eos_token_id
            )
            
            # Convert to strings (simplified)
            samples = []
            for seq in generated:
                # This would need detokenization - simplified for now
                sample = " ".join([str(token.item()) for token in seq])
                samples.append(sample)
        
        return samples
    
    def save_checkpoint(self, path: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'd_state': self.model.d_state,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'max_length': self.model.max_length,
                'pad_token_id': self.model.pad_token_id,
                'eos_token_id': self.model.eos_token_id,
                'temperature': self.model.temperature,
                'use_entropy_regularization': self.model.use_entropy_regularization,
                'use_frequency_alignment': self.model.use_frequency_alignment
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Additional information from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {path}")
        
        # Return additional info
        additional_info = {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'optimizer_state_dict', 
                                     'scheduler_state_dict', 'epoch', 'global_step', 
                                     'best_loss', 'model_config']}
        
        return additional_info
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = self.model.get_model_statistics()
        
        # Add training statistics
        stats.update({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'device': str(self.device)
        })
        
        return stats
    
    def update_temperature(self, new_temperature: float):
        """Update temperature for annealing."""
        self.model.update_temperature(new_temperature)
        print(f"Temperature updated to {new_temperature}")
    
    def update_loss_weights(
        self,
        gflownet_weight: Optional[float] = None,
        entropy_weight: Optional[float] = None,
        flow_consistency_weight: Optional[float] = None
    ):
        """Update loss weights during training."""
        if gflownet_weight is not None:
            self.gflownet_loss_weight = gflownet_weight
        if entropy_weight is not None:
            self.entropy_loss_weight = entropy_weight
        if flow_consistency_weight is not None:
            self.flow_consistency_weight = flow_consistency_weight
        
        print(f"Loss weights updated: GFlowNet={self.gflownet_loss_weight}, "
              f"Entropy={self.entropy_loss_weight}, "
              f"Flow Consistency={self.flow_consistency_weight}") 