#!/usr/bin/env python3
"""
Simplified Experiment 1: Multi-Horizon Claims Forecasting
========================================================

Simplified version that disables unstable components for a working baseline.
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
from tqdm import tqdm
import json

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
from utils.metrics import InsuranceMetrics
import matplotlib.pyplot as plt


def train_experiment():
    """Run simplified training experiment."""
    
    # Configuration
    data_path = 'data/exp1_simple.csv'
    output_dir = Path('results/experiment_1_simple')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic data...")
    create_synthetic_dataset(output_path=data_path, num_policies=500, num_months=100)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=16,  # Smaller batch
        history_length=60,
        forecast_horizons=[3, 6, 12, 24]
    )
    
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['history'].shape[-1]
    
    print(f"Initializing model (features: {num_features})...")
    model = InsuranceGSSM(
        num_features=num_features,
        d_model=128,  # Smaller model
        d_state=32,
        num_layers=4,
        dropout=0.1,
        max_history_length=60,
        forecast_horizons=[3, 6, 12, 24],
        use_seasonal_encoding=True,
        use_insurance_autocorrelation=False,  # DISABLE autocorrelation
        use_cycle_detection=False  # DISABLE cycle detection
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01, eps=1e-8)
    metrics_calc = InsuranceMetrics()
    
    print(f"\n{'='*70}")
    print(f"SIMPLIFIED EXPERIMENT 1: Claims Forecasting")
    print(f"{'='*70}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Autocorrelation: DISABLED (for stability)")
    print(f"Cycle Detection: DISABLED (for stability)\n")
    
    best_val_loss = float('inf')
    results = {'train_losses': [], 'val_losses': []}
    
    # Training loop
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            history = batch['history'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            outputs = model(history)
            
            # Simple MSE loss only
            loss = 0.0
            for horizon in [3, 6, 12, 24]:
                pred_key = f'claims_amount_{horizon}m'
                target_key = f'{horizon}m'
                if pred_key in outputs and target_key in targets:
                    # Clamp to prevent explosion
                    pred = torch.clamp(outputs[pred_key], min=-1e6, max=1e6)
                    targ = targets[target_key].squeeze(-1)
                    loss += nn.functional.mse_loss(pred, targ)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN detected, skipping")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # Check gradients
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan = True
                    break
            
            if not has_nan:
                optimizer.step()
                train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        results['train_losses'].append(avg_train_loss)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_losses = []
            all_preds = {h: [] for h in [3, 6, 12, 24]}
            all_targs = {h: [] for h in [3, 6, 12, 24]}
            
            with torch.no_grad():
                for batch in test_loader:
                    history = batch['history'].to(device)
                    targets = {k: v.to(device) for k, v in batch['targets'].items()}
                    
                    outputs = model(history)
                    
                    loss = 0.0
                    for horizon in [3, 6, 12, 24]:
                        pred_key = f'claims_amount_{horizon}m'
                        target_key = f'{horizon}m'
                        if pred_key in outputs and target_key in targets:
                            pred = torch.clamp(outputs[pred_key], min=-1e6, max=1e6)
                            targ = targets[target_key].squeeze(-1)
                            loss += nn.functional.mse_loss(pred, targ)
                            
                            all_preds[horizon].append(pred.cpu())
                            all_targs[horizon].append(targ.cpu())
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
            results['val_losses'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Compute metrics for each horizon
            for horizon in [3, 6, 12, 24]:
                if all_preds[horizon]:
                    preds = torch.cat(all_preds[horizon]).numpy().flatten()
                    targs = torch.cat(all_targs[horizon]).numpy().flatten()
                    m = metrics_calc.compute_metrics(targs, preds)
                    print(f"  {horizon}m: MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, output_dir / 'best_model.pt')
                print(f"  ✅ Best model saved!")
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'best_val_loss': best_val_loss}, f)
    
    # Plot training curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(results['train_losses']) + 1)
    ax.plot(epochs, results['train_losses'], label='Train Loss', marker='o')
    if results['val_losses']:
        val_epochs = [i * 5 for i in range(1, len(results['val_losses']) + 1)]
        ax.plot(val_epochs, results['val_losses'], label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress - Simplified GSSM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 1 (SIMPLIFIED) COMPLETED")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    train_experiment()
