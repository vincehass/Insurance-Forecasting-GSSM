#!/usr/bin/env python3
"""
Experiment 2: Risk-Based Pricing Optimization
==============================================

Use Case: Insurance company needs to optimize premium pricing based on risk
assessment and historical claims patterns while maintaining profitability.

Goal: Train GSSM model to classify risk levels and recommend optimal pricing
actions that balance competitiveness with loss ratio management.

Evaluation:
- Risk classification accuracy (Precision, Recall, F1)
- Pricing action confidence and distribution
- Business metrics (Loss Ratio, Profit Margin, Combined Ratio)
- Expected revenue impact vs baseline pricing
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
from sklearn.metrics import classification_report, confusion_matrix

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
from utils.metrics import InsuranceMetrics
import matplotlib.pyplot as plt
import seaborn as sns


class RiskPricingExperiment:
    """
    Experiment focusing on risk classification and pricing optimization.
    """
    
    def __init__(
        self,
        model: InsuranceGSSM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cuda",
        output_dir: str = "results/experiment_2/"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = InsuranceMetrics()
        self.best_val_loss = float('inf')
        self.train_losses = []
        
        # Risk categories
        self.risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Pricing actions
        self.action_names = [
            'Decrease 10%', 'Decrease 5%', 'No Change', 'Increase 5%',
            'Increase 10%', 'Increase 15%', 'Increase 20%', 'Manual Review'
        ]
    
    def compute_loss(self, outputs, targets, frequencies, risks):
        """Compute loss with focus on risk classification and pricing."""
        losses = {}
        total_loss = 0.0
        
        # Risk classification loss (primary objective)
        if 'risk_logits' in outputs:
            risk_loss = nn.functional.cross_entropy(
                outputs['risk_logits'],
                risks.squeeze(-1)
            )
            losses['risk_loss'] = risk_loss
            total_loss += 2.0 * risk_loss  # Higher weight for risk classification
        
        # Claims forecasting (for context)
        if 'claims_amount_12m' in outputs and '12m' in targets:
            # Clamp predictions to reasonable range
            pred_clamped = torch.clamp(outputs['claims_amount_12m'], min=-1e6, max=1e6)
            mse_loss = nn.functional.mse_loss(
                pred_clamped,
                targets['12m'].squeeze(-1)
            )
            losses['mse_12m'] = mse_loss
            total_loss += 0.5 * mse_loss
        
        # Pricing entropy (encourage confident decisions)
        if 'pricing_extras' in outputs:
            if 'entropy_loss' in outputs['pricing_extras']:
                entropy = outputs['pricing_extras']['entropy_loss']
                losses['pricing_entropy'] = entropy
                # Minimize entropy for confident decisions
                total_loss += 0.1 * entropy
        
        # Business metrics alignment
        if 'business_metrics' in outputs:
            biz_metrics = outputs['business_metrics']
            # Penalize high loss ratios
            if 'loss_ratio' in biz_metrics:
                loss_ratio = biz_metrics['loss_ratio'].mean()
                # Target loss ratio around 0.65 (65%)
                lr_penalty = torch.abs(loss_ratio - 0.65) * 0.3
                losses['loss_ratio_penalty'] = lr_penalty
                total_loss += lr_penalty
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total_loss': [],
            'risk_loss': [],
            'mse_12m': []
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            history = batch['history'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
            risks = batch['risk'].to(self.device)
            
            outputs = self.model(
                history,
                return_extras=True,
                compute_business_metrics=True
            )
            
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
        """Comprehensive evaluation focusing on risk and pricing."""
        self.model.eval()
        
        all_risk_preds = []
        all_risk_true = []
        all_pricing_actions = []
        all_pricing_probs = []
        all_business_metrics = []
        total_loss = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                history = batch['history'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                frequencies = {k: v.to(self.device) for k, v in batch['frequencies'].items()}
                risks = batch['risk'].to(self.device)
                
                outputs = self.model(
                    history,
                    return_extras=True,
                    compute_business_metrics=True
                )
                
                losses = self.compute_loss(outputs, targets, frequencies, risks)
                total_loss.append(losses['total_loss'].item())
                
                # Collect risk predictions
                if 'risk_probs' in outputs:
                    risk_preds = outputs['risk_probs'].argmax(dim=1).cpu().numpy()
                    all_risk_preds.extend(risk_preds)
                    all_risk_true.extend(risks.cpu().numpy().flatten())
                
                # Collect pricing actions
                if 'pricing_action' in outputs:
                    actions = outputs['pricing_action'].cpu().numpy()
                    all_pricing_actions.extend(actions)
                    
                    if 'pricing_probs' in outputs:
                        probs = outputs['pricing_probs'].cpu().numpy()
                        all_pricing_probs.extend(probs)
                
                # Collect business metrics
                if 'business_metrics' in outputs:
                    biz = outputs['business_metrics']
                    all_business_metrics.append({
                        'loss_ratio': biz['loss_ratio'].mean().item(),
                        'profit_margin': biz['profit_margin'].mean().item(),
                        'combined_ratio': biz['combined_ratio'].mean().item()
                    })
        
        # Compute results
        results = {
            'avg_loss': np.mean(total_loss),
            'risk_classification': {},
            'pricing_analysis': {},
            'business_metrics': {}
        }
        
        # Risk classification metrics
        if all_risk_preds:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            accuracy = accuracy_score(all_risk_true, all_risk_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_risk_true, all_risk_preds, average='weighted'
            )
            
            results['risk_classification'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': all_risk_preds,
                'true_labels': all_risk_true
            }
            
            print(f"\n{split_name} - Risk Classification:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        # Pricing action analysis
        if all_pricing_actions:
            action_counts = np.bincount(all_pricing_actions, minlength=8)
            action_distribution = action_counts / len(all_pricing_actions)
            
            avg_confidence = np.mean([max(probs) for probs in all_pricing_probs]) if all_pricing_probs else 0.0
            
            results['pricing_analysis'] = {
                'action_distribution': action_distribution.tolist(),
                'avg_confidence': avg_confidence,
                'actions': all_pricing_actions,
                'probabilities': all_pricing_probs
            }
            
            print(f"\n{split_name} - Pricing Actions:")
            print(f"  Average Confidence: {avg_confidence:.4f}")
            print(f"  Action Distribution:")
            for i, (name, pct) in enumerate(zip(self.action_names, action_distribution)):
                print(f"    {name:20s}: {pct*100:5.1f}%")
        
        # Business metrics
        if all_business_metrics:
            avg_metrics = {
                'loss_ratio': np.mean([m['loss_ratio'] for m in all_business_metrics]),
                'profit_margin': np.mean([m['profit_margin'] for m in all_business_metrics]),
                'combined_ratio': np.mean([m['combined_ratio'] for m in all_business_metrics])
            }
            results['business_metrics'] = avg_metrics
            
            print(f"\n{split_name} - Business Metrics:")
            print(f"  Loss Ratio:     {avg_metrics['loss_ratio']:.4f}")
            print(f"  Profit Margin:  {avg_metrics['profit_margin']:.4f}")
            print(f"  Combined Ratio: {avg_metrics['combined_ratio']:.4f}")
        
        return results
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"EXPERIMENT 2: Risk-Based Pricing Optimization")
        print(f"{'='*70}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training for {num_epochs} epochs...")
        print(f"Focus: Risk Classification + Pricing Recommendations\n")
        
        for epoch in range(1, num_epochs + 1):
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
                val_results = self.evaluate(self.val_loader, "Validation")
                
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
        test_results = self.evaluate(self.test_loader, "Test")
        
        # Save results
        results_to_save = {
            k: v for k, v in test_results.items()
            if k not in ['risk_classification', 'pricing_analysis']  # Skip arrays
        }
        results_to_save['risk_accuracy'] = test_results['risk_classification']['accuracy']
        results_to_save['risk_f1'] = test_results['risk_classification']['f1_score']
        results_to_save['pricing_confidence'] = test_results['pricing_analysis']['avg_confidence']
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Generate visualizations
        self.generate_visualizations(test_results)
        
        return test_results
    
    def generate_visualizations(self, results):
        """Generate comprehensive visualizations."""
        print("\nGenerating visualizations...")
        
        # 1. Risk Classification Confusion Matrix
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        cm = confusion_matrix(
            results['risk_classification']['true_labels'],
            results['risk_classification']['predictions']
        )
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=self.risk_names,
            yticklabels=self.risk_names,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted Risk Level', fontsize=12)
        ax.set_ylabel('True Risk Level', fontsize=12)
        ax.set_title('Risk Classification Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Pricing Action Distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        action_dist = results['pricing_analysis']['action_distribution']
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(self.action_names)))
        
        bars = ax.bar(range(len(self.action_names)), [d*100 for d in action_dist], color=colors)
        ax.set_xlabel('Pricing Action', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title('Pricing Action Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(self.action_names)))
        ax.set_xticklabels(self.action_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pricing_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Business Metrics Dashboard
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Business Performance Metrics', fontsize=16, fontweight='bold')
        
        metrics = results['business_metrics']
        metric_names = ['Loss Ratio', 'Profit Margin', 'Combined Ratio']
        metric_values = [metrics['loss_ratio'], metrics['profit_margin'], metrics['combined_ratio']]
        target_values = [0.65, 0.15, 0.95]  # Industry targets
        
        for ax, name, value, target in zip(axes, metric_names, metric_values, target_values):
            # Gauge-style visualization
            color = 'green' if abs(value - target) < 0.1 else 'orange' if abs(value - target) < 0.2 else 'red'
            
            ax.barh([0], [value], color=color, alpha=0.7, label='Actual')
            ax.axvline(target, color='blue', linestyle='--', linewidth=2, label='Target')
            ax.set_xlim([0, 1])
            ax.set_yticks([])
            ax.set_xlabel('Value', fontsize=11)
            ax.set_title(f'{name}\nActual: {value:.3f} | Target: {target:.3f}', fontsize=11)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'business_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Training Loss Curves
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        total_losses = [loss['total_loss'] for loss in self.train_losses]
        risk_losses = [loss.get('risk_loss', np.nan) for loss in self.train_losses]
        
        ax.plot(epochs, total_losses, label='Total Loss', linewidth=2)
        ax.plot(epochs, risk_losses, label='Risk Classification Loss', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
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
    parser = argparse.ArgumentParser(description='Experiment 2: Risk-Based Pricing')
    parser.add_argument('--data_path', type=str, default='data/exp2_pricing_data.csv')
    parser.add_argument('--output_dir', type=str, default='results/experiment_2_risk_pricing')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--generate_synthetic', action='store_true', default=True)
    args = parser.parse_args()
    
    # Generate synthetic data
    if args.generate_synthetic:
        print("Generating synthetic insurance data for risk-pricing analysis...")
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
        use_insurance_autocorrelation=True,  # ✓ ENABLED - Critical for risk patterns
        use_cycle_detection=True             # ✓ ENABLED - For cycle-based risk assessment
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    
    # Run experiment
    experiment = RiskPricingExperiment(
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
    print("EXPERIMENT 2 COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"\nKey Findings:")
    print(f"  Risk Classification Accuracy: {results['risk_classification']['accuracy']:.2%}")
    print(f"  Risk F1-Score: {results['risk_classification']['f1_score']:.3f}")
    print(f"  Pricing Confidence: {results['pricing_analysis']['avg_confidence']:.2%}")
    print(f"  Loss Ratio: {results['business_metrics']['loss_ratio']:.3f}")
    print(f"  Profit Margin: {results['business_metrics']['profit_margin']:.3f}")


if __name__ == '__main__':
    main()
