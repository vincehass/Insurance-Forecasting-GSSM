#!/usr/bin/env python3
"""
Training Script for Insurance GSSM
==================================

Main script for training GSSM on insurance forecasting tasks.

Usage:
    python experiments/train_insurance_gssm.py \
        --data_path data/processed/insurance_data.csv \
        --output_dir results/gssm_baseline/ \
        --epochs 150 \
        --batch_size 32
"""

import sys
sys.path.insert(0, '../src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
import json
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
from utils.metrics import InsuranceMetrics
from utils.visualization import plot_training_curves, plot_forecasts


class InsuranceGSSMTrainer:
    """
    Trainer for Insurance GSSM model.
    """
    
    def __init__(
        self,
        model: InsuranceGSSM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "results/",
        use_wandb: bool = False,
        forecast_horizons: List[int] = [3, 6, 12, 24]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.forecast_horizons = forecast_horizons
        
        self.model.to(device)
        self.metrics = InsuranceMetrics()
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        frequencies: Dict[str, torch.Tensor],
        risks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss for insurance forecasting.
        
        Args:
            outputs: Model outputs
            targets: Target claims amounts
            frequencies: Target claims counts
            risks: Risk labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Claims amount forecasting loss (MSE)
        for horizon in self.forecast_horizons:
            pred_key = f'claims_amount_{horizon}m'
            target_key = f'{horizon}m'
            
            if pred_key in outputs and target_key in targets:
                mse_loss = nn.functional.mse_loss(
                    outputs[pred_key],
                    targets[target_key].squeeze(-1)
                )
                losses[f'mse_{horizon}m'] = mse_loss
                total_loss += mse_loss
        
        # Claims frequency forecasting loss (Poisson NLL)
        for horizon in self.forecast_horizons:
            pred_key = f'claims_count_{horizon}m'
            target_key = f'{horizon}m'
            
            if pred_key in outputs and target_key in frequencies:
                # Poisson negative log likelihood
                freq_loss = nn.functional.poisson_nll_loss(
                    torch.log(outputs[pred_key] + 1e-8),
                    frequencies[target_key].squeeze(-1).float(),
                    log_input=True
                )
                losses[f'frequency_{horizon}m'] = freq_loss
                total_loss += 0.5 * freq_loss  # Weight frequency loss
        
        # Risk classification loss
        if 'risk_logits' in outputs:
            risk_loss = nn.functional.cross_entropy(
                outputs['risk_logits'],
                risks.squeeze(-1)
            )
            losses['risk_loss'] = risk_loss
            total_loss += 0.3 * risk_loss
        
        # Autocorrelation reward (maximize)
        if 'autocorr_reward' in outputs:
            autocorr_loss = -outputs['autocorr_reward']
            losses['autocorr_loss'] = autocorr_loss
            total_loss += 0.2 * autocorr_loss
        
        # Pricing entropy (for exploration)
        if 'pricing_extras' in outputs and 'entropy_loss' in outputs['pricing_extras']:
            entropy_loss = outputs['pricing_extras']['entropy_loss']
            losses['entropy_loss'] = entropy_loss
            total_loss += 0.1 * entropy_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': [],
            **{f'mse_{h}m': [] for h in self.forecast_horizons},
            **{f'frequency_{h}m': [] for h in self.forecast_horizons}
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            history = batch['history'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
            risks = batch['risk'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                history,
                return_extras=True,
                compute_business_metrics=True
            )
            
            # Compute loss
            losses = self.compute_loss(outputs, targets, frequencies, risks)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key].append(losses[key].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}"
            })
        
        # Average epoch losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if len(v) > 0}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = {
            'total_loss': [],
            **{f'mse_{h}m': [] for h in self.forecast_horizons}
        }
        
        all_predictions = {h: [] for h in self.forecast_horizons}
        all_targets = {h: [] for h in self.forecast_horizons}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                history = batch['history'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
                risks = batch['risk'].to(self.device)
                
                # Forward pass
                outputs = self.model(history, compute_business_metrics=True)
                
                # Compute loss
                losses = self.compute_loss(outputs, targets, frequencies, risks)
                
                # Record losses
                for key in val_losses.keys():
                    if key in losses:
                        val_losses[key].append(losses[key].item())
                
                # Collect predictions for metrics
                for horizon in self.forecast_horizons:
                    pred_key = f'claims_amount_{horizon}m'
                    target_key = f'{horizon}m'
                    if pred_key in outputs:
                        all_predictions[horizon].append(outputs[pred_key].cpu())
                        all_targets[horizon].append(targets[target_key].cpu())
        
        # Average validation losses
        avg_losses = {k: np.mean(v) for k, v in val_losses.items() if len(v) > 0}
        
        # Compute detailed metrics
        for horizon in self.forecast_horizons:
            if all_predictions[horizon]:
                preds = torch.cat(all_predictions[horizon]).numpy()
                targs = torch.cat(all_targets[horizon]).numpy()
                
                metrics = self.metrics.compute_metrics(targs, preds)
                for metric_name, value in metrics.items():
                    avg_losses[f'{metric_name}_{horizon}m'] = value
        
        return avg_losses
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
            for horizon in self.forecast_horizons:
                if f'mse_{horizon}m' in val_losses:
                    print(f"  Val MSE ({horizon}m): {val_losses[f'mse_{horizon}m']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_losses.items()},
                    **{f'val/{k}': v for k, v in val_losses.items()}
                })
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✅ Best model saved (val_loss: {self.best_val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n✅ Training completed!")
        self.plot_results()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.output_dir / 'best_model.pt'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
    
    def plot_results(self):
        """Plot training results."""
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            save_path=self.output_dir / 'training_curves.png'
        )


def main():
    parser = argparse.ArgumentParser(description='Train Insurance GSSM')
    parser.add_argument('--data_path', type=str, default='data/synthetic_insurance.csv')
    parser.add_argument('--output_dir', type=str, default='results/gssm_baseline')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--history_length', type=int, default=60)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--generate_synthetic', action='store_true')
    args = parser.parse_args()
    
    # Generate synthetic data if needed
    if args.generate_synthetic:
        print("Generating synthetic insurance data...")
        create_synthetic_dataset(
            output_path=args.data_path,
            num_policies=1000,
            num_months=100
        )
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        history_length=args.history_length,
        forecast_horizons=[3, 6, 12, 24]
    )
    
    # Get sample to determine num_features
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['history'].shape[-1]
    print(f"Number of features: {num_features}")
    
    # Initialize model
    print("Initializing model...")
    model = InsuranceGSSM(
        num_features=num_features,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_history_length=args.history_length,
        forecast_horizons=[3, 6, 12, 24],
        use_seasonal_encoding=True,
        use_insurance_autocorrelation=True,
        use_cycle_detection=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2
    )
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="insurance-gssm",
            config=vars(args)
        )
        wandb.watch(model)
    
    # Initialize trainer
    trainer = InsuranceGSSMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Train
    trainer.train(args.epochs)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
