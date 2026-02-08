#!/usr/bin/env python3
"""
Simplified Experiment 2: Risk-Based Pricing
==========================================

Simplified version focusing on risk classification and pricing.
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from gssm.insurance_gssm import InsuranceGSSM
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset
import matplotlib.pyplot as plt
import seaborn as sns


def train_experiment():
    """Run simplified risk-pricing experiment."""
    
    # Configuration
    data_path = 'data/exp2_simple.csv'
    output_dir = Path('results/experiment_2_simple')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic data...")
    create_synthetic_dataset(output_path=data_path, num_policies=500, num_months=100)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=data_path,
        batch_size=16,
        history_length=60,
        forecast_horizons=[3, 6, 12, 24]
    )
    
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['history'].shape[-1]
    
    print(f"Initializing model (features: {num_features})...")
    model = InsuranceGSSM(
        num_features=num_features,
        d_model=128,
        d_state=32,
        num_layers=4,
        dropout=0.1,
        max_history_length=60,
        forecast_horizons=[3, 6, 12, 24],
        use_seasonal_encoding=True,
        use_insurance_autocorrelation=False,
        use_cycle_detection=False
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01, eps=1e-8)
    
    print(f"\n{'='*70}")
    print(f"SIMPLIFIED EXPERIMENT 2: Risk-Based Pricing")
    print(f"{'='*70}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")
    
    best_val_loss = float('inf')
    results = {'train_losses': [], 'val_losses': [], 'risk_accuracy': []}
    
    # Training loop
    num_epochs = 30
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            history = batch['history'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            risks = batch['risk'].to(device)
            
            outputs = model(history, compute_business_metrics=True)
            
            # Combined loss: forecasting + risk classification
            loss = 0.0
            
            # Forecasting loss (reduced weight)
            if 'claims_amount_12m' in outputs and '12m' in targets:
                pred = torch.clamp(outputs['claims_amount_12m'], min=-1e6, max=1e6)
                loss += 0.5 * nn.functional.mse_loss(pred, targets['12m'].squeeze(-1))
            
            # Risk classification loss (primary)
            if 'risk_logits' in outputs:
                risk_loss = nn.functional.cross_entropy(
                    outputs['risk_logits'],
                    risks.squeeze(-1)
                )
                loss += 2.0 * risk_loss
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
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
            all_risk_preds = []
            all_risk_true = []
            all_pricing_actions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    history = batch['history'].to(device)
                    targets = {k: v.to(device) for k, v in batch['targets'].items()}
                    risks = batch['risk'].to(device)
                    
                    outputs = model(history, compute_business_metrics=True)
                    
                    loss = 0.0
                    if 'claims_amount_12m' in outputs and '12m' in targets:
                        pred = torch.clamp(outputs['claims_amount_12m'], min=-1e6, max=1e6)
                        loss += 0.5 * nn.functional.mse_loss(pred, targets['12m'].squeeze(-1))
                    
                    if 'risk_logits' in outputs:
                        loss += 2.0 * nn.functional.cross_entropy(outputs['risk_logits'], risks.squeeze(-1))
                        risk_preds = outputs['risk_probs'].argmax(dim=1).cpu().numpy()
                        all_risk_preds.extend(risk_preds)
                        all_risk_true.extend(risks.cpu().numpy().flatten())
                    
                    if 'pricing_action' in outputs:
                        actions = outputs['pricing_action'].cpu().numpy()
                        all_pricing_actions.extend(actions)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
            results['val_losses'].append(avg_val_loss)
            
            # Risk accuracy
            risk_acc = accuracy_score(all_risk_true, all_risk_preds) if all_risk_preds else 0.0
            results['risk_accuracy'].append(risk_acc)
            
            # Pricing distribution
            pricing_dist = np.bincount(all_pricing_actions, minlength=8) / len(all_pricing_actions) if all_pricing_actions else np.zeros(8)
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Risk Accuracy: {risk_acc:.4f}")
            print(f"  Pricing Actions: {pricing_dist}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'risk_accuracy': risk_acc,
                }, output_dir / 'best_model.pt')
                print(f"  ✅ Best model saved!")
    
    # Final evaluation
    model.eval()
    all_risk_preds = []
    all_risk_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            history = batch['history'].to(device)
            risks = batch['risk'].to(device)
            outputs = model(history)
            
            if 'risk_probs' in outputs:
                risk_preds = outputs['risk_probs'].argmax(dim=1).cpu().numpy()
                all_risk_preds.extend(risk_preds)
                all_risk_true.extend(risks.cpu().numpy().flatten())
    
    # Confusion matrix
    if all_risk_preds:
        cm = confusion_matrix(all_risk_true, all_risk_preds)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        ax.set_xlabel('Predicted Risk')
        ax.set_ylabel('True Risk')
        ax.set_title('Risk Classification Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report
        report = classification_report(all_risk_true, all_risk_preds,
                                      target_names=['Low', 'Medium', 'High'])
        print(f"\nClassification Report:\n{report}")
        
        with open(output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    epochs = range(1, len(results['train_losses']) + 1)
    axes[0].plot(epochs, results['train_losses'], label='Train Loss', marker='o')
    if results['val_losses']:
        val_epochs = [i * 5 for i in range(1, len(results['val_losses']) + 1)]
        axes[0].plot(val_epochs, results['val_losses'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    if results['risk_accuracy']:
        val_epochs = [i * 5 for i in range(1, len(results['risk_accuracy']) + 1)]
        axes[1].plot(val_epochs, results['risk_accuracy'], label='Risk Accuracy', marker='s', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Risk Classification Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    final_acc = results['risk_accuracy'][-1] if results['risk_accuracy'] else 0.0
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'best_val_loss': best_val_loss,
            'final_risk_accuracy': final_acc
        }, f)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT 2 (SIMPLIFIED) COMPLETED")
    print(f"{'='*70}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final risk accuracy: {final_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    train_experiment()
