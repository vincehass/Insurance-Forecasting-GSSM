#!/usr/bin/env python3
"""
Comprehensive Visualization Generator for Insurance GSSM
======================================================

Generates all visualizations for:
- Baseline experiments (Experiment 1 & 2)
- Ablation studies
- Comparative analysis
- Methodology figures
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ComprehensiveVisualizer:
    """Generate all visualizations for the study."""
    
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'full': '#2ecc71',  # Green - best
            'no_autocorr': '#e74c3c',  # Red - expected large drop
            'no_cycle': '#f39c12',  # Orange
            'no_flow': '#9b59b6',  # Purple
            'no_seasonal': '#3498db',  # Blue
            'minimal': '#95a5a6',  # Gray - worst
        }
        
    def load_ablation_results(self):
        """Load all ablation study results."""
        ablation_dir = self.results_dir / 'ablation'
        
        results = {}
        for config_dir in ablation_dir.iterdir():
            if config_dir.is_dir():
                results_file = config_dir / 'results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results[config_dir.name] = json.load(f)
        
        return results
    
    def plot_ablation_comparison(self, results):
        """
        Main ablation comparison figure showing performance drop.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Component Contribution Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Extract metrics for 12-month horizon
        configs = []
        mse_values = []
        r2_values = []
        mape_values = []
        mae_values = []
        
        config_order = ['full', 'no_seasonal', 'no_cycle', 'no_flow', 'no_autocorr', 'minimal']
        
        for config_name in config_order:
            if config_name in results:
                r = results[config_name]
                if 'test_metrics' in r and '12m' in r['test_metrics']:
                    configs.append(r['config']['name'])
                    mse_values.append(r['test_metrics']['12m']['mse'])
                    r2_values.append(r['test_metrics']['12m']['r2'])
                    mape_values.append(r['test_metrics']['12m']['mape'])
                    mae_values.append(r['test_metrics']['12m']['mae'])
        
        x = np.arange(len(configs))
        colors = [self.colors.get(config_order[i], 'gray') for i in range(len(configs))]
        
        # 1. MSE Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x, mse_values, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MSE (12-month)', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Squared Error Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. R² Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, r2_values, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score (12-month)', fontsize=12, fontweight='bold')
        ax2.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. MAPE Comparison
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x, mape_values, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MAPE % (12-month)', fontsize=12, fontweight='bold')
        ax3.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(configs, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Performance Drop Analysis
        ax4 = axes[1, 1]
        if mse_values:
            baseline_mse = mse_values[0]  # Full model
            drops = [(mse - baseline_mse) / baseline_mse * 100 for mse in mse_values[1:]]
            
            bars4 = ax4.barh(range(len(drops)), drops, 
                            color=[self.colors.get(config_order[i+1], 'gray') 
                                   for i in range(len(drops))],
                            alpha=0.8, edgecolor='black')
            ax4.set_yticks(range(len(drops)))
            ax4.set_yticklabels(configs[1:])
            ax4.set_xlabel('Performance Drop (%)', fontsize=12, fontweight='bold')
            ax4.set_title('Relative Performance Drop from Baseline', 
                         fontsize=14, fontweight='bold')
            ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax4.grid(True, alpha=0.3, axis='x')
            
            for i, bar in enumerate(bars4):
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:+.1f}%', ha='left' if width > 0 else 'right',
                        va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ablation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: ablation_comparison.png")
    
    def plot_component_importance(self, results):
        """Component importance ranking visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate importance based on performance drop
        config_order = ['full', 'no_autocorr', 'no_cycle', 'no_flow', 'no_seasonal']
        component_names = ['Autocorrelation\n(r_AC)', 'Cycle Detection\n(FFT)', 
                          'Flow-Selectivity', 'Seasonal\nEncoding']
        
        if 'full' in results and all(c in results for c in config_order[1:]):
            baseline_mse = results['full']['test_metrics']['12m']['mse']
            
            importance = []
            for config_name, comp_name in zip(config_order[1:], component_names):
                ablated_mse = results[config_name]['test_metrics']['12m']['mse']
                drop = (ablated_mse - baseline_mse) / baseline_mse * 100
                importance.append((comp_name, drop))
            
            # Sort by importance
            importance.sort(key=lambda x: x[1], reverse=True)
            
            components, drops = zip(*importance)
            colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']
            
            bars = ax.barh(range(len(components)), drops, color=colors, 
                          alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_yticks(range(len(components)))
            ax.set_yticklabels(components, fontsize=12)
            ax.set_xlabel('Performance Impact (% MSE increase)', fontsize=13, fontweight='bold')
            ax.set_title('Component Importance Ranking\n(Higher = More Critical)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                       f'+{width:.1f}%', ha='left', va='center',
                       fontsize=12, fontweight='bold')
            
            # Add criticality levels
            ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, linewidth=2)
            ax.axvline(x=5, color='yellow', linestyle='--', alpha=0.5, linewidth=2)
            
            ax.text(20, len(components)-0.5, 'Critical', rotation=90, 
                   va='bottom', ha='right', color='red', fontweight='bold')
            ax.text(10, len(components)-0.5, 'High', rotation=90,
                   va='bottom', ha='right', color='orange', fontweight='bold')
            ax.text(5, len(components)-0.5, 'Moderate', rotation=90,
                   va='bottom', ha='right', color='darkgoldenrod', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'component_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: component_importance.png")
    
    def plot_multi_horizon_comparison(self, results):
        """Compare all ablations across multiple horizons."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Horizon Performance Analysis', 
                    fontsize=18, fontweight='bold')
        
        horizons = [3, 6, 12, 24]
        config_order = ['full', 'no_autocorr', 'no_cycle', 'no_flow', 'no_seasonal', 'minimal']
        config_names = [results[c]['config']['name'] for c in config_order if c in results]
        
        metrics = ['mse', 'mae', 'r2', 'mape']
        titles = ['Mean Squared Error', 'Mean Absolute Error', 'R² Score', 'MAPE (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for i, config_name in enumerate(config_order):
                if config_name in results:
                    values = []
                    for h in horizons:
                        if f'{h}m' in results[config_name]['test_metrics']:
                            values.append(results[config_name]['test_metrics'][f'{h}m'][metric])
                    
                    ax.plot(horizons, values, marker='o', linewidth=2, markersize=8,
                           label=results[config_name]['config']['name'],
                           color=self.colors.get(config_name, 'gray'))
            
            ax.set_xlabel('Forecast Horizon (months)', fontsize=11, fontweight='bold')
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(horizons)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'multi_horizon_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: multi_horizon_comparison.png")
    
    def plot_ablation_table(self, results):
        """Create detailed results table visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        config_order = ['full', 'no_autocorr', 'no_cycle', 'no_flow', 'no_seasonal', 'minimal']
        
        table_data = []
        headers = ['Configuration', 'MSE↓', 'MAE↓', 'RMSE↓', 'MAPE↓', 'R²↑', 'Drop%']
        
        baseline_mse = None
        for config_name in config_order:
            if config_name in results and '12m' in results[config_name]['test_metrics']:
                metrics = results[config_name]['test_metrics']['12m']
                mse = metrics['mse']
                
                if baseline_mse is None:
                    baseline_mse = mse
                    drop = 0.0
                else:
                    drop = (mse - baseline_mse) / baseline_mse * 100
                
                row = [
                    results[config_name]['config']['name'],
                    f"{mse:.4f}",
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['mape']:.2f}%",
                    f"{metrics['r2']:.4f}",
                    f"+{drop:.1f}%" if drop > 0 else f"{drop:.1f}%"
                ]
                table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code rows
        for i, config_name in enumerate(config_order):
            if i < len(table_data):
                color = self.colors.get(config_name, 'white')
                for j in range(len(headers)):
                    if i == 0:  # Baseline
                        table[(i+1, j)].set_facecolor('#d5f4e6')
                    else:
                        table[(i+1, j)].set_facecolor('white')
        
        plt.title('Detailed Ablation Study Results (12-Month Horizon)',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(self.figures_dir / 'ablation_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: ablation_table.png")
    
    def plot_architecture_diagram(self):
        """Create GSSM architecture visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(5, 11.5, 'Insurance GSSM Architecture', 
               ha='center', fontsize=18, fontweight='bold')
        
        # Input layer
        ax.add_patch(plt.Rectangle((1, 10), 8, 0.8, facecolor='#3498db', 
                                   edgecolor='black', linewidth=2))
        ax.text(5, 10.4, 'Input: [batch, 60 months, 7 features]',
               ha='center', va='center', fontsize=11, color='white', fontweight='bold')
        
        # Feature embedding
        ax.add_patch(plt.Rectangle((1.5, 8.5), 7, 0.8, facecolor='#9b59b6',
                                   edgecolor='black', linewidth=2))
        ax.text(5, 8.9, 'Feature Embedding (Linear: 7 → 128)',
               ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Encodings
        ax.add_patch(plt.Rectangle((0.5, 7), 3, 0.8, facecolor='#f39c12',
                                   edgecolor='black', linewidth=2))
        ax.text(2, 7.4, 'Seasonal Encoding\n(Monthly/Quarterly)',
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.add_patch(plt.Rectangle((4.5, 7), 3, 0.8, facecolor='#e74c3c',
                                   edgecolor='black', linewidth=2))
        ax.text(6, 7.4, 'Autocorrelation (r_AC)\n(Seasonal Patterns)',
               ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        # SSM layers
        for i in range(4):
            y_pos = 5.5 - i*0.9
            ax.add_patch(plt.Rectangle((2, y_pos), 6, 0.7, facecolor='#2ecc71',
                                       edgecolor='black', linewidth=2))
            ax.text(5, y_pos+0.35, f'SSM Layer {i+1} (d_state=32)',
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Cycle detection
        ax.add_patch(plt.Rectangle((8.5, 4), 1.3, 2, facecolor='#e67e22',
                                   edgecolor='black', linewidth=2))
        ax.text(9.15, 5, 'Cycle\nDetect\n(FFT)',
               ha='center', va='center', fontsize=8, fontweight='bold', rotation=0)
        
        # Output heads
        heads = ['Claims (3m)', 'Claims (6m)', 'Claims (12m)', 'Claims (24m)', 
                'Risk Class', 'Pricing Action']
        colors = ['#1abc9c', '#16a085', '#27ae60', '#229954', '#8e44ad', '#c0392b']
        
        for i, (head, color) in enumerate(zip(heads, colors)):
            x_pos = 1 + i * 1.4
            ax.add_patch(plt.Rectangle((x_pos, 0.5), 1.2, 0.7, facecolor=color,
                                       edgecolor='black', linewidth=1.5))
            ax.text(x_pos+0.6, 0.85, head, ha='center', va='center',
                   fontsize=8, color='white', fontweight='bold')
        
        # Arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        ax.annotate('', xy=(5, 10), xytext=(5, 9.3), arrowprops=arrow_props)
        ax.annotate('', xy=(5, 8.5), xytext=(5, 7.8), arrowprops=arrow_props)
        ax.annotate('', xy=(5, 5.5), xytext=(5, 2), arrowprops=arrow_props)
        ax.annotate('', xy=(5, 2), xytext=(5, 1.2), arrowprops=arrow_props)
        
        plt.savefig(self.figures_dir / 'architecture_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Generated: architecture_diagram.png")
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Load ablation results
        print("Loading ablation results...")
        ablation_results = self.load_ablation_results()
        
        if not ablation_results:
            print("❌ No ablation results found. Run ablation study first.")
            return
        
        print(f"Found {len(ablation_results)} configurations\n")
        
        # Generate all plots
        print("Generating figures...")
        self.plot_ablation_comparison(ablation_results)
        self.plot_component_importance(ablation_results)
        self.plot_multi_horizon_comparison(ablation_results)
        self.plot_ablation_table(ablation_results)
        self.plot_architecture_diagram()
        
        print(f"\n{'='*70}")
        print(f"✅ All visualizations generated successfully!")
        print(f"{'='*70}")
        print(f"Saved to: {self.figures_dir}")
        print(f"\nGenerated files:")
        for f in sorted(self.figures_dir.glob('*.png')):
            print(f"  - {f.name}")


def main():
    visualizer = ComprehensiveVisualizer('results')
    visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
