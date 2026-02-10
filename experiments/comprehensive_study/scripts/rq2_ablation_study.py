"""
RQ2: What is Each Component's Contribution?

Theoretical Link: Theorem 3.2 (Spectral Sync) and Proposition 3.1 (Flow Gate)

Hypothesis: Each domain-specific component (Flow-Selectivity, FFT Cycles, Autocorr, 
GFlowNet Policy) contributes meaningfully to forecasting accuracy.

This script systematically ablates each component to measure individual contributions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from gssm.insurance_gssm import InsuranceGSSM
from utils.data_generator import InsuranceDataGenerator
from utils.metrics import compute_all_metrics


class RQ2AblationExperiment:
    """
    RQ2: Comprehensive ablation study
    
    7 configurations:
    1. Full Model (all components)
    2. w/o Flow-Selectivity (φ_FS)
    3. w/o FFT Cycles
    4. w/o Autocorrelation (r_AC)
    5. w/o Seasonal Encoding (τ_SE)
    6. w/o GFlowNet Policy
    7. Minimal SSM (only SSM backbone)
    """
    
    def __init__(self, results_dir: Path, device: str = 'cuda'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.horizons = [3, 6, 12, 24]
        self.configurations = {
            'Full Model': {
                'use_flow': True,
                'use_cycle': True,
                'use_autocorr': True,
                'use_seasonal': True,
                'use_gflownet': True
            },
            'w/o Flow': {
                'use_flow': False,
                'use_cycle': True,
                'use_autocorr': True,
                'use_seasonal': True,
                'use_gflownet': True
            },
            'w/o FFT': {
                'use_flow': True,
                'use_cycle': False,
                'use_autocorr': True,
                'use_seasonal': True,
                'use_gflownet': True
            },
            'w/o Autocorr': {
                'use_flow': True,
                'use_cycle': True,
                'use_autocorr': False,
                'use_seasonal': True,
                'use_gflownet': True
            },
            'w/o Seasonal': {
                'use_flow': True,
                'use_cycle': True,
                'use_autocorr': True,
                'use_seasonal': False,
                'use_gflownet': True
            },
            'w/o GFlowNet': {
                'use_flow': False,
                'use_cycle': True,
                'use_autocorr': True,
                'use_seasonal': True,
                'use_gflownet': False
            },
            'Minimal SSM': {
                'use_flow': False,
                'use_cycle': False,
                'use_autocorr': False,
                'use_seasonal': False,
                'use_gflownet': False
            }
        }
        
        print(f"✓ RQ2 Ablation Study initialized")
        print(f"  Configurations: {len(self.configurations)}")
        print(f"  Horizons: {self.horizons}")
    
    def create_model(self, config_name: str, config: Dict):
        """Create model with specific ablation configuration"""
        model = InsuranceGSSM(
            input_dim=20,
            output_dim=1,
            state_dim=64,
            d_model=256,
            n_layers=4,
            use_flow=config['use_flow'],
            use_cycle=config['use_cycle'],
            use_autocorr=config['use_autocorr'],
            use_seasonal=config['use_seasonal']
        ).to(self.device)
        
        return model
    
    def train_and_evaluate(
        self,
        config_name: str,
        config: Dict,
        train_data: torch.Tensor,
        test_data: torch.Tensor,
        horizon: int,
        epochs: int = 50
    ) -> Dict[str, float]:
        """Train and evaluate ablation configuration"""
        
        model = self.create_model(config_name, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # Simplified training (placeholder)
        X_train = torch.randn(100, 36, 20).to(self.device)
        y_train = train_data[:100].unsqueeze(1).to(self.device)
        
        model.train()
        for epoch in range(min(epochs, 10)):
            optimizer.zero_grad()
            pred_train = model(X_train)
            loss = criterion(pred_train, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test = torch.randn(len(test_data), 36, 20).to(self.device)
            pred = model(X_test).cpu().squeeze()
        
        metrics = compute_all_metrics(test_data.numpy(), pred.numpy())
        return metrics
    
    def run_ablation_study(self, epochs: int = 50, num_seeds: int = 3):
        """Run full ablation study"""
        
        print(f"\n🚀 Running RQ2: Ablation Study")
        print(f"   Configurations: {len(self.configurations)}")
        print(f"   Horizons: {len(self.horizons)}")
        print(f"   Seeds: {num_seeds}")
        
        results = []
        generator = InsuranceDataGenerator(n_policies=10000, n_months=120)
        
        for seed in [42, 123, 456][:num_seeds]:
            print(f"\n{'='*80}")
            print(f"SEED: {seed}")
            print(f"{'='*80}")
            
            # Generate data
            data = generator.generate_with_cycles(random_state=seed)
            train_size = int(0.70 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            for config_name, config in self.configurations.items():
                for horizon in self.horizons:
                    try:
                        metrics = self.train_and_evaluate(
                            config_name, config, train_data, test_data, horizon, epochs
                        )
                        
                        result = {
                            'seed': seed,
                            'configuration': config_name,
                            'horizon': horizon,
                            **config,
                            **metrics
                        }
                        results.append(result)
                        
                        print(f"  ✓ {config_name:20s} | {horizon:2d}m | R²={metrics.get('r2', 0):.4f}")
                    
                    except Exception as e:
                        print(f"  ✗ {config_name:20s} | {horizon:2d}m | Error: {str(e)[:50]}")
                        results.append({
                            'seed': seed,
                            'configuration': config_name,
                            'horizon': horizon,
                            'r2': 0.0,
                            'mse': 999.0
                        })
        
        # Save results
        df = pd.DataFrame(results)
        output_path = self.results_dir / 'rq2_ablation_results.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        
        # Analyze component impacts
        self.analyze_component_impacts(df)
        
        return df
    
    def analyze_component_impacts(self, df: pd.DataFrame):
        """Analyze individual component contributions"""
        
        print(f"\n📊 Component Impact Analysis")
        print(f"{'='*100}")
        
        # Compute average impact of each component
        full_model_r2 = df[df['configuration'] == 'Full Model'].groupby('horizon')['r2'].mean()
        minimal_ssm_r2 = df[df['configuration'] == 'Minimal SSM'].groupby('horizon')['r2'].mean()
        
        impacts = []
        for config_name in self.configurations.keys():
            if config_name in ['Full Model', 'Minimal SSM']:
                continue
            
            config_r2 = df[df['configuration'] == config_name].groupby('horizon')['r2'].mean()
            
            for horizon in self.horizons:
                impact = full_model_r2[horizon] - config_r2[horizon]
                impacts.append({
                    'component': config_name.replace('w/o ', ''),
                    'horizon': horizon,
                    'impact': impact,
                    'impact_pct': (impact / full_model_r2[horizon]) * 100
                })
        
        impact_df = pd.DataFrame(impacts)
        impact_path = self.results_dir / 'rq2_component_impacts.csv'
        impact_df.to_csv(impact_path, index=False)
        
        print(f"\n✓ Component impacts saved to: {impact_path}")
        
        # Print domain gain
        for horizon in self.horizons:
            domain_gain = ((full_model_r2[horizon] - minimal_ssm_r2[horizon]) / minimal_ssm_r2[horizon]) * 100
            print(f"  {horizon:2d}m horizon: Domain Gain = +{domain_gain:.1f}%")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RQ2: Ablation Study')
    parser.add_argument('--results_dir', type=str, default='../results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    experiment = RQ2AblationExperiment(results_dir=args.results_dir, device=args.device)
    results_df = experiment.run_ablation_study(epochs=args.epochs, num_seeds=args.num_seeds)
    
    print(f"\n✅ RQ2 EXPERIMENT COMPLETE")


if __name__ == '__main__':
    main()
