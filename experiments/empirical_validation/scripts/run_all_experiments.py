"""
Master Script: Run All Empirical Validation Experiments
Executes complete experimental pipeline from data generation to report creation
Author: Nadhir Hassen (nadhir.hassen@mila.quebec)
"""

import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime
import json

class ExperimentRunner:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.scripts_dir = self.base_dir / 'scripts'
        self.results_dir = self.base_dir / 'results'
        self.figures_dir = self.base_dir / 'figures'
        self.tables_dir = self.base_dir / 'tables'
        self.analysis_dir = self.base_dir / 'analysis'
        
        # Create directories
        for d in [self.results_dir, self.figures_dir, self.tables_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.log = []
        
    def run_step(self, name, command, description):
        """Run a single experiment step"""
        print(f"\n{'='*80}")
        print(f"Step: {name}")
        print(f"Description: {description}")
        print(f"Command: {command}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir
            )
            
            elapsed = time.time() - start_time
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            self.log.append({
                'step': name,
                'description': description,
                'command': command,
                'status': 'SUCCESS',
                'elapsed_seconds': round(elapsed, 2),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"\n✓ {name} completed in {elapsed:.1f}s")
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            
            print(f"\n✗ {name} FAILED after {elapsed:.1f}s")
            print(f"Error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            
            self.log.append({
                'step': name,
                'description': description,
                'command': command,
                'status': 'FAILED',
                'elapsed_seconds': round(elapsed, 2),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            return False
    
    def run_all(self):
        """Run complete experimental pipeline"""
        
        print(f"\n{'#'*80}")
        print("EMPIRICAL VALIDATION - FULL EXPERIMENTAL PIPELINE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        steps = [
            {
                'name': '01_Data_Generation',
                'command': f'python {self.scripts_dir}/01_generate_data.py --n_policies 1000 --n_months 100 --output {self.results_dir}/insurance_data.csv',
                'description': 'Generate synthetic insurance dataset with embedded patterns'
            },
            {
                'name': '02_Feature_Engineering',
                'command': f'python {self.scripts_dir}/02_feature_engineering.py --data {self.results_dir}/insurance_data.csv --output {self.results_dir}/features.npz',
                'description': 'Extract and engineer 50+ features for model training'
            },
            {
                'name': '03_Baseline_Comparison',
                'command': f'python {self.scripts_dir}/03_run_baselines.py --data {self.results_dir}/features.npz --seeds 10 --output {self.results_dir}/baseline_results.json',
                'description': 'Train and evaluate 10 baseline methods with 10 seeds'
            },
            {
                'name': '04_Ablation_Study',
                'command': f'python {self.scripts_dir}/04_run_ablation.py --data {self.results_dir}/features.npz --seeds 10 --output {self.results_dir}/ablation_results.json',
                'description': 'Systematic ablation of GSSM components'
            },
            {
                'name': '05_Component_Analysis',
                'command': f'python {self.scripts_dir}/05_component_analysis.py --results {self.results_dir} --output {self.analysis_dir}',
                'description': 'Analyze individual component contributions'
            },
            {
                'name': '06_Generate_Visualizations',
                'command': f'python {self.scripts_dir}/06_generate_figures.py --results {self.results_dir} --output {self.figures_dir} --format pdf',
                'description': 'Generate all publication-quality figures'
            },
            {
                'name': '07_Statistical_Validation',
                'command': f'python {self.scripts_dir}/07_statistical_tests.py --results {self.results_dir} --output {self.tables_dir}/statistical_validation.tex',
                'description': 'Perform statistical significance tests'
            },
            {
                'name': '08_Generate_Tables',
                'command': f'python {self.scripts_dir}/08_generate_tables.py --results {self.results_dir} --output {self.tables_dir}',
                'description': 'Generate LaTeX tables for all results'
            },
            {
                'name': '09_RQ_Analysis',
                'command': f'python {self.scripts_dir}/09_research_question_analysis.py --results {self.results_dir} --figures {self.figures_dir} --output {self.analysis_dir}',
                'description': 'Answer each research question with detailed analysis'
            },
            {
                'name': '10_Generate_Report',
                'command': f'python {self.scripts_dir}/10_generate_report.py --results {self.results_dir} --figures {self.figures_dir} --tables {self.tables_dir} --analysis {self.analysis_dir} --output {self.base_dir}/empirical_validation_report.tex',
                'description': 'Compile comprehensive LaTeX report'
            }
        ]
        
        total_start = time.time()
        
        for step in steps:
            success = self.run_step(
                name=step['name'],
                command=step['command'],
                description=step['description']
            )
            
            if not success:
                print(f"\n✗ Pipeline FAILED at step: {step['name']}")
                print(f"  See log for details")
                break
        
        total_elapsed = time.time() - total_start
        
        # Save execution log
        log_path = self.base_dir / 'execution_log.json'
        with open(log_path, 'w') as f:
            json.dump({
                'total_elapsed_seconds': round(total_elapsed, 2),
                'total_elapsed_human': f"{int(total_elapsed // 60)}m {int(total_elapsed % 60)}s",
                'completed_at': datetime.now().isoformat(),
                'steps': self.log
            }, f, indent=2)
        
        print(f"\n{'#'*80}")
        print("PIPELINE SUMMARY")
        print(f"{'#'*80}")
        print(f"Total time: {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
        print(f"Successful steps: {sum(1 for s in self.log if s['status'] == 'SUCCESS')}/{len(self.log)}")
        print(f"Log saved to: {log_path}")
        print(f"{'#'*80}\n")
        
        # Print individual step times
        print("Step Timing:")
        for entry in self.log:
            status_symbol = "✓" if entry['status'] == 'SUCCESS' else "✗"
            print(f"  {status_symbol} {entry['step']}: {entry['elapsed_seconds']}s")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run complete empirical validation pipeline')
    parser.add_argument('--base_dir', type=str, 
                       default='/Users/nhassen/Documents/AIML/Insurance/Insurance-Forecasting-GSSM/experiments/empirical_validation',
                       help='Base experiment directory')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.base_dir)
    runner.run_all()


if __name__ == '__main__':
    main()
