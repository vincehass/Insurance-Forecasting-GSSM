#!/usr/bin/env python3
"""
Experiment 1: Multi-Horizon Claims Forecasting
==============================================

Use Case: Insurance company needs to forecast claims amounts for strategic planning
and reserve allocation across multiple time horizons (3, 6, 12, 24 months).

Goal: Train GSSM model to accurately predict future claims amounts and compare
performance across different forecast horizons.

Evaluation:
- MSE, MAE, RMSE, MAPE, R² for each horizon
- Forecasting accuracy comparison
- Visualization of predictions vs actuals
- Model component contribution analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
from utils.metrics import InsuranceMetrics
from utils.visualization import plot_training_curves, plot_forecasts
import matplotlib.pyplot as plt


class ClaimsForecastingExperiment:
    """
    Experiment focusing on multi-horizon claims forecasting.
    """
    
    def __init__(
        self,
        model: InsuranceGSSM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cuda",
        output_dir: str = "results/experiment_1/",
        forecast_horizons: list = [3, 6, 12, 24]
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.forecast_horizons = forecast_horizons
        
        self.metrics = InsuranceMetrics()
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def compute_loss(self, outputs, targets, frequencies, risks):
        """Compute multi-task loss with focus on claims forecasting."""
        losses = {}
        total_loss = 0.0
        
        # Claims amount forecasting loss (primary objective)
        for horizon in self.forecast_horizons:
            pred_key = f'claims_amount_{horizon}m'
            target_key = f'{horizon}m'
            
            if pred_key in outputs and target_key in targets:
                mse_loss = nn.functional.mse_loss(
                    outputs[pred_key],
                    targets[target_key].squeeze(-1)
                )
                losses[f'mse_{horizon}m'] = mse_loss
                # Weight longer horizons slightly more
                weight = 1.0 + (horizon / 100.0)
                total_loss += weight * mse_loss
        
        # Claims frequency forecasting (secondary)
        for horizon in self.forecast_horizons:
            pred_key = f'claims_count_{horizon}m'
            target_key = f'{horizon}m'
            
            if pred_key in outputs and target_key in frequencies:
                # Ensure positive predictions and add numerical stability
                pred_freq = torch.clamp(outputs[pred_key], min=1e-6)
                freq_loss = nn.functional.poisson_nll_loss(
                    torch.log(pred_freq),
                    frequencies[target_key].squeeze(-1).float(),
                    log_input=True
                )
                losses[f'frequency_{horizon}m'] = freq_loss
                total_loss += 0.3 * freq_loss
        
        # Autocorrelation reward (helps with seasonality)
        if 'autocorr_reward' in outputs:
            autocorr_loss = -outputs['autocorr_reward']
            losses['autocorr_loss'] = autocorr_loss
            total_loss += 0.3 * autocorr_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': [],
            **{f'mse_{h}m': [] for h in self.forecast_horizons}
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            history = batch['history'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
            risks = batch['risk'].to(self.device)
            
            outputs = self.model(history, return_extras=True)
            losses = self.compute_loss(outputs, targets, frequencies, risks)
            
            # Check for NaN in loss
            if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
                print(f"Warning: NaN/Inf detected in loss, skipping batch")
                continue
            
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Clip gradients more aggressively
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            # Check for NaN in gradients
            has_nan = False
            for param in self.model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan = True
                    break
            
            if has_nan:
                print(f"Warning: NaN/Inf detected in gradients, skipping batch")
                self.optimizer.zero_grad()
                continue
                
            self.optimizer.step()
            
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key].append(losses[key].item())
            
            progress_bar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
        
        return {k: np.mean(v) for k, v in epoch_losses.items() if len(v) > 0}
    
    def evaluate(self, data_loader, split_name="Test"):
        """Comprehensive evaluation on a dataset split."""
        self.model.eval()
        
        all_predictions = {h: [] for h in self.forecast_horizons}
        all_targets = {h: [] for h in self.forecast_horizons}
        all_losses = {'total_loss': []}
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                history = batch['history'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
                risks = batch['risk'].to(self.device)
                
                outputs = self.model(history)
                losses = self.compute_loss(outputs, targets, frequencies, risks)
                all_losses['total_loss'].append(losses['total_loss'].item())
                
                # Collect predictions
                for horizon in self.forecast_horizons:
                    pred_key = f'claims_amount_{horizon}m'
                    target_key = f'{horizon}m'
                    if pred_key in outputs:
                        all_predictions[horizon].append(outputs[pred_key].cpu())
                        all_targets[horizon].append(targets[target_key].cpu())
        
        # Compute metrics for each horizon
        results = {
            'avg_loss': np.mean(all_losses['total_loss']),
            'horizons': {}
        }
        
        for horizon in self.forecast_horizons:
            if all_predictions[horizon]:
                preds = torch.cat(all_predictions[horizon]).numpy()
                targs = torch.cat(all_targets[horizon]).numpy().flatten()
                
                metrics = self.metrics.compute_metrics(targs, preds)
                results['horizons'][f'{horizon}m'] = metrics
                
                print(f"\n{split_name} Results - {horizon}-month horizon:")
                print(f"  MSE:  {metrics['mse']:.4f}")
                print(f"  MAE:  {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAPE: {metrics['mape']:.2f}%")
                print(f"  R²:   {metrics['r2']:.4f}")
        
        return results, all_predictions, all_targets
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 1: Multi-Horizon Claims Forecasting")
        print(f"{'='*70}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training for {num_epochs} epochs...")
        print(f"Forecast horizons: {self.forecast_horizons} months\n")
        
        for epoch in range(1, num_epochs + 1):
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
                val_results, _, _ = self.evaluate(self.val_loader, "Validation")
                
                if val_results['avg_loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['avg_loss']
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"  ✅ Best model saved (val_loss: {self.best_val_loss:.4f})")
        
        print("\n✅ Training completed!")
        return self.test_model()
    
    def test_model(self):
        """Test the best model and generate visualizations."""
        print(f"\n{'='*70}")
        print("Final Evaluation on Test Set")
        print(f"{'='*70}")
        
        # Load best model
        checkpoint = torch.load(self.output_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        test_results, predictions, targets = self.evaluate(self.test_loader, "Test")
        
        # Save results
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Generate visualizations
        self.generate_visualizations(predictions, targets, test_results)
        
        return test_results
    
    def generate_visualizations(self, predictions, targets, results):
        """Generate comprehensive visualizations."""
        print("\nGenerating visualizations...")
        
        # 1. Training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment 1: Claims Forecasting - Training Progress', fontsize=16, fontweight='bold')
        
        for idx, horizon in enumerate(self.forecast_horizons):
            ax = axes[idx // 2, idx % 2]
            train_mse = [loss.get(f'mse_{horizon}m', np.nan) for loss in self.train_losses]
            ax.plot(train_mse, label=f'Train MSE ({horizon}m)', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE')
            ax.set_title(f'{horizon}-Month Forecast Horizon')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction vs Actual for each horizon
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment 1: Predictions vs Actuals', fontsize=16, fontweight='bold')
        
        for idx, horizon in enumerate(self.forecast_horizons):
            ax = axes[idx // 2, idx % 2]
            if predictions[horizon]:
                preds = torch.cat(predictions[horizon]).numpy().flatten()[:200]  # First 200 samples
                targs = torch.cat(targets[horizon]).numpy().flatten()[:200]
                
                ax.scatter(targs, preds, alpha=0.5, s=20)
                
                # Perfect prediction line
                min_val = min(targs.min(), preds.min())
                max_val = max(targs.max(), preds.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                metrics = results['horizons'][f'{horizon}m']
                ax.set_xlabel('Actual Claims Amount ($)', fontsize=10)
                ax.set_ylabel('Predicted Claims Amount ($)', fontsize=10)
                ax.set_title(f'{horizon}-Month Horizon (R²={metrics["r2"]:.3f})', fontsize=11)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions_vs_actuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance comparison across horizons
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        horizons_list = []
        mse_values = []
        mae_values = []
        r2_values = []
        
        for horizon in self.forecast_horizons:
            metrics = results['horizons'][f'{horizon}m']
            horizons_list.append(f'{horizon}m')
            mse_values.append(metrics['mse'])
            mae_values.append(metrics['mae'])
            r2_values.append(metrics['r2'])
        
        x = np.arange(len(horizons_list))
        width = 0.25
        
        ax.bar(x - width, mse_values, width, label='MSE', alpha=0.8)
        ax.bar(x, mae_values, width, label='MAE', alpha=0.8)
        ax.bar(x + width, r2_values, width, label='R²', alpha=0.8)
        
        ax.set_xlabel('Forecast Horizon', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Performance Metrics Across Forecast Horizons', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_list)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'horizon_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {self.output_dir}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pt')


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Claims Forecasting')
    parser.add_argument('--data_path', type=str, default='data/exp1_claims_data.csv')
    parser.add_argument('--output_dir', type=str, default='results/experiment_1_claims_forecasting')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--generate_synthetic', action='store_true', default=True)
    args = parser.parse_args()
    
    # Generate synthetic data
    if args.generate_synthetic:
        print("Generating synthetic insurance data for claims forecasting...")
        create_synthetic_dataset(
            output_path=args.data_path,
            num_policies=500,  # Reduced for faster training
            num_months=100
        )
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        history_length=60,
        forecast_horizons=[3, 6, 12, 24]
    )
    
    # Get number of features
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['history'].shape[-1]
    
    # Initialize model with ALL GSSM features enabled
    print(f"Initializing model (features: {num_features})...")
    model = InsuranceGSSM(
        num_features=num_features,
        d_model=128,  # Reduced for stability
        d_state=32,   # Reduced for stability
        num_layers=4,  # Reduced for stability
        dropout=0.1,
        max_history_length=60,
        forecast_horizons=[3, 6, 12, 24],
        use_seasonal_encoding=True,
        use_insurance_autocorrelation=True,  # ✓ ENABLED - Critical for seasonality
        use_cycle_detection=True             # ✓ ENABLED - For FFT-based patterns
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    
    # Run experiment
    experiment = ClaimsForecastingExperiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=args.device,
        output_dir=args.output_dir
    )
    
    results = experiment.train(args.epochs)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey Findings:")
    for horizon in [3, 6, 12, 24]:
        metrics = results['horizons'][f'{horizon}m']
        print(f"  {horizon}-month forecast: R²={metrics['r2']:.3f}, MAPE={metrics['mape']:.2f}%")


if __name__ == '__main__':
    main()
