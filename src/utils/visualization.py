"""
Visualization utilities for insurance forecasting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(
    train_losses: List[Dict[str, float]],
    val_losses: List[Dict[str, float]],
    save_path: Optional[str] = None
):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    train_total = [l['total_loss'] for l in train_losses]
    val_total = [l['total_loss'] for l in val_losses]
    
    ax.plot(epochs, train_total, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_total, label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Horizon-specific MSE
    ax = axes[1]
    horizons = [3, 6, 12, 24]
    for horizon in horizons:
        key = f'mse_{horizon}m'
        if key in val_losses[0]:
            vals = [l[key] for l in val_losses]
            ax.plot(epochs, vals, label=f'{horizon}m horizon', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Validation MSE by Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    horizon: int = 12,
    save_path: Optional[str] = None
):
    """Plot forecast vs actual values."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    if timestamps is None:
        timestamps = np.arange(len(y_true))
    
    # Time series plot
    ax = axes[0]
    ax.plot(timestamps, y_true, label='Actual', color='black', linewidth=2, alpha=0.8)
    ax.plot(timestamps, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
    ax.fill_between(timestamps, y_true, y_pred, alpha=0.2, color='gray')
    ax.set_xlabel('Time')
    ax.set_ylabel('Claims Amount ($)')
    ax.set_title(f'{horizon}-Month Forecast: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Residual plot
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(timestamps, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Residual ($)')
    ax.set_title('Forecast Residuals')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Forecast plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_results(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """Plot ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    configs = list(ablation_results.keys())
    mse_values = [ablation_results[c]['mse'] for c in configs]
    
    # Bar plot
    ax = axes[0]
    colors = ['green' if c == 'Full GSSM' else 'red' for c in configs]
    bars = ax.bar(configs, mse_values, color=colors, alpha=0.7)
    ax.set_ylabel('MSE')
    ax.set_title('Ablation Study: MSE by Configuration')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance drop
    ax = axes[1]
    baseline_mse = ablation_results['Full GSSM']['mse']
    drops = [(ablation_results[c]['mse'] - baseline_mse) / baseline_mse * 100 
             for c in configs]
    
    bars = ax.barh(configs, drops, color=['green' if d == 0 else 'red' for d in drops], alpha=0.7)
    ax.set_xlabel('Performance Drop (%)')
    ax.set_title('Component Removal Impact')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Ablation plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
