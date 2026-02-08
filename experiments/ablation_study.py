#!/usr/bin/env python3
"""
Comprehensive Ablation Study for Insurance GSSM
==============================================

Tests the contribution of each component by systematically removing them:
1. Full Model (Baseline)
2. w/o Autocorrelation (r_AC)
3. w/o Cycle Detection (FFT)
4. w/o Flow-Selectivity
5. w/o Seasonal Encoding
6. Minimal SSM (only state-space layers)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
from utils.metrics import InsuranceMetrics
import matplotlib.pyplot as plt


class AblationConfig:
    """Configuration for ablation experiments."""
    
    ABLATIONS = {
        'full': {
            'name': 'Full Model (Baseline)',
            'use_autocorrelation': True,
            'use_cycle_detection': True,
            'use_flow_selectivity': True,
            'use_seasonal_encoding': True,
        },
        'no_autocorr': {
            'name': 'w/o Autocorrelation (r_AC)',
            'use_autocorrelation': False,
            'use_cycle_detection': True,
            'use_flow_selectivity': True,
            'use_seasonal_encoding': True,
        },
        'no_cycle': {
            'name': 'w/o Cycle Detection (FFT)',
            'use_autocorrelation': True,
            'use_cycle_detection': False,
            'use_flow_selectivity': True,
            'use_seasonal_encoding': True,
        },
        'no_flow': {
            'name': 'w/o Flow-Selectivity',
            'use_autocorrelation': True,
            'use_cycle_detection': True,
            'use_flow_selectivity': False,
            'use_seasonal_encoding': True,
        },
        'no_seasonal': {
            'name': 'w/o Seasonal Encoding',
            'use_autocorrelation': True,
            'use_cycle_detection': True,
            'use_flow_selectivity': True,
            'use_seasonal_encoding': False,
        },
        'minimal': {
            'name': 'Minimal SSM (No Insurance Components)',
            'use_autocorrelation': False,
            'use_cycle_detection': False,
            'use_flow_selectivity': False,
            'use_seasonal_encoding': False,
        },
    }


class AblationExperiment:
    """Run ablation study for a specific configuration."""
    
    def __init__(
        self,
        config_name: str,
        train_loader,
        val_loader,
        test_loader,
        num_features: int,
        device: str = 'cpu',
        output_dir: str = 'results/ablation',
        epochs: int = 30
    ):
        self.config_name = config_name
        self.config = AblationConfig.ABLATIONS[config_name]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_features = num_features
        self.device = device
        self.output_dir = Path(output_dir) / config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        
        self.metrics_calc = InsuranceMetrics()
        self.results = {'config': self.config, 'train_losses': [], 'val_metrics': []}
        
        # Initialize model with ablation config
        self.model = self._create_model()
        self.model.to(device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
            eps=1e-8
        )
        
        self.best_val_loss = float('inf')
        
    def _create_model(self):
        """Create model with ablation configuration."""
        return InsuranceGSSM(
            num_features=self.num_features,
            d_model=128,
            d_state=32,
            num_layers=4,
            dropout=0.1,
            max_history_length=60,
            forecast_horizons=[3, 6, 12, 24],
            use_seasonal_encoding=self.config['use_seasonal_encoding'],
            use_insurance_autocorrelation=self.config['use_autocorrelation'],
            use_cycle_detection=self.config['use_cycle_detection']
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch in tqdm(self.train_loader, desc=f"{self.config['name']}", leave=False):
            history = batch['history'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            outputs = self.model(history)
            
            # Simple MSE loss for claims forecasting
            loss = 0.0
            for horizon in [3, 6, 12, 24]:
                pred_key = f'claims_amount_{horizon}m'
                target_key = f'{horizon}m'
                if pred_key in outputs and target_key in targets:
                    pred = torch.clamp(outputs[pred_key], min=-1e6, max=1e6)
                    loss += nn.functional.mse_loss(pred, targets[target_key].squeeze(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            # Check gradients
            has_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in self.model.parameters()
            )
            
            if not has_nan:
                self.optimizer.step()
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses) if epoch_losses else float('nan')
    
    def evaluate(self, data_loader):
        """Evaluate model."""
        self.model.eval()
        all_preds = {h: [] for h in [3, 6, 12, 24]}
        all_targs = {h: [] for h in [3, 6, 12, 24]}
        losses = []
        
        with torch.no_grad():
            for batch in data_loader:
                history = batch['history'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                outputs = self.model(history)
                
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
                    losses.append(loss.item())
        
        # Compute metrics
        metrics = {'avg_loss': np.mean(losses) if losses else float('nan')}
        
        for horizon in [3, 6, 12, 24]:
            if all_preds[horizon]:
                preds = torch.cat(all_preds[horizon]).numpy().flatten()
                targs = torch.cat(all_targs[horizon]).numpy().flatten()
                h_metrics = self.metrics_calc.compute_metrics(targs, preds)
                metrics[f'{horizon}m'] = h_metrics
        
        return metrics
    
    def run(self):
        """Run complete ablation experiment."""
        print(f"\n{'='*70}")
        print(f"Running: {self.config['name']}")
        print(f"{'='*70}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Configuration: {self.config}")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            self.results['train_losses'].append(train_loss)
            
            if epoch % 5 == 0:
                val_metrics = self.evaluate(self.val_loader)
                self.results['val_metrics'].append(val_metrics)
                
                print(f"\nEpoch {epoch}/{self.epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
                
                if '12m' in val_metrics:
                    print(f"  12m: MSE={val_metrics['12m']['mse']:.4f}, "
                          f"R²={val_metrics['12m']['r2']:.4f}")
                
                if val_metrics['avg_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['avg_loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'config': self.config,
                    }, self.output_dir / 'best_model.pt')
        
        # Final evaluation on test set
        print("\nFinal Test Evaluation...")
        test_metrics = self.evaluate(self.test_loader)
        self.results['test_metrics'] = test_metrics
        
        print(f"\nTest Results:")
        for horizon in [3, 6, 12, 24]:
            if f'{horizon}m' in test_metrics:
                m = test_metrics[f'{horizon}m']
                print(f"  {horizon}m: MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, "
                      f"R²={m['r2']:.4f}, MAPE={m['mape']:.2f}%")
        
        # Save results
        results_to_save = {
            'config_name': self.config_name,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'test_metrics': {
                k: v for k, v in test_metrics.items()
                if k != 'avg_loss' or isinstance(v, (int, float))
            }
        }
        
        # Convert numpy types to Python types for JSON
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_to_save = convert_to_serializable(results_to_save)
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        return self.results


def run_all_ablations(data_path: str, output_dir: str, epochs: int, device: str):
    """Run all ablation experiments."""
    
    print("="*70)
    print("COMPREHENSIVE ABLATION STUDY")
    print("="*70)
    print(f"Total configurations: {len(AblationConfig.ABLATIONS)}")
    print(f"Epochs per config: {epochs}")
    print(f"Device: {device}")
    print()
    
    # Load data once
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=16,
        history_length=60,
        forecast_horizons=[3, 6, 12, 24]
    )
    
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['history'].shape[-1]
    
    all_results = {}
    
    # Run each ablation
    for config_name in AblationConfig.ABLATIONS.keys():
        experiment = AblationExperiment(
            config_name=config_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_features=num_features,
            device=device,
            output_dir=output_dir,
            epochs=epochs
        )
        
        results = experiment.run()
        all_results[config_name] = results
    
    # Save combined results
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETED")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary
    summary = {}
    for config_name, results in all_results.items():
        if 'test_metrics' in results and '12m' in results['test_metrics']:
            summary[config_name] = {
                'name': AblationConfig.ABLATIONS[config_name]['name'],
                'mse_12m': results['test_metrics']['12m']['mse'],
                'r2_12m': results['test_metrics']['12m']['r2'],
                'mape_12m': results['test_metrics']['12m']['mape'],
            }
    
    with open(output_path / 'ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary saved to ablation_summary.json")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run Ablation Study')
    parser.add_argument('--data_path', type=str, default='data/ablation_data.csv')
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--generate_data', action='store_true', default=True)
    args = parser.parse_args()
    
    if args.generate_data:
        print("Generating synthetic data for ablation study...")
        create_synthetic_dataset(
            output_path=args.data_path,
            num_policies=500,
            num_months=100
        )
    
    run_all_ablations(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        device=args.device
    )


if __name__ == '__main__':
    main()
