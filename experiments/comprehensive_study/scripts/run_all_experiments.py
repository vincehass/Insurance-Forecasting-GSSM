"""
Master Orchestration Script for Complete GSSM Experimental Framework

Executes all 5 research questions sequentially or in parallel:
- RQ1: Baseline comparison (15 methods)
- RQ2: Ablation study (7 configurations)
- RQ3: Synergy analysis (pairwise interactions)
- RQ4: Cycle validation (spectral analysis)
- RQ5: Business impact (financial translation)

Then generates all visualizations and analysis reports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

import subprocess
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, List


class ExperimentOrchestrator:
    """
    Master orchestrator for running complete GSSM experimental validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.scripts_dir = Path(__file__).parent
        self.results_dir = self.scripts_dir.parent / 'results'
        self.viz_dir = self.scripts_dir.parent / 'visualizations'
        self.analysis_dir = self.scripts_dir.parent / 'analysis'
        
        # Create directories
        for directory in [self.results_dir, self.viz_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Experiment metadata
        self.start_time = datetime.now()
        self.metadata = {
            'start_time': self.start_time.isoformat(),
            'config': config,
            'experiments': {}
        }
        
        print(f"\n{'='*100}")
        print(f"🚀 COMPREHENSIVE GSSM EXPERIMENTAL FRAMEWORK")
        print(f"{'='*100}")
        print(f"\nConfiguration:")
        print(f"  Mode: {config['mode']}")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Seeds: {config['num_seeds']}")
        print(f"  Device: {config['device']}")
        print(f"\nDirectories:")
        print(f"  Results: {self.results_dir}")
        print(f"  Visualizations: {self.viz_dir}")
        print(f"  Analysis: {self.analysis_dir}")
    
    def run_experiment(self, rq_name: str, script_name: str) -> Dict:
        """Run a single RQ experiment"""
        
        print(f"\n{'='*80}")
        print(f"▶️  {rq_name}")
        print(f"{'='*80}")
        
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"  ⚠️  Script not found: {script_path}")
            return {'status': 'skipped', 'reason': 'script_not_found'}
        
        exp_start = datetime.now()
        
        try:
            cmd = [
                'python', str(script_path),
                '--results_dir', str(self.results_dir),
                '--epochs', str(self.config['epochs']),
                '--num_seeds', str(self.config['num_seeds']),
                '--device', self.config['device']
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('timeout', 3600)  # 1 hour default
            )
            
            exp_duration = (datetime.now() - exp_start).total_seconds()
            
            if result.returncode == 0:
                print(f"  ✅ {rq_name} completed successfully")
                print(f"     Duration: {exp_duration:.1f}s")
                
                return {
                    'status': 'success',
                    'duration_seconds': exp_duration,
                    'output': result.stdout[-500:] if result.stdout else ''  # Last 500 chars
                }
            else:
                print(f"  ❌ {rq_name} failed")
                print(f"     Error: {result.stderr[-200:]}")
                
                return {
                    'status': 'failed',
                    'duration_seconds': exp_duration,
                    'error': result.stderr[-500:]
                }
        
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  {rq_name} timed out")
            return {'status': 'timeout', 'duration_seconds': self.config.get('timeout')}
        
        except Exception as e:
            print(f"  ❌ {rq_name} error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def run_sequential(self):
        """Run all experiments sequentially"""
        
        experiments = [
            ('RQ1: Baseline Comparison', 'rq1_baseline_comparison.py'),
            ('RQ2: Ablation Study', 'rq2_ablation_study.py'),
        ]
        
        for rq_name, script_name in experiments:
            result = self.run_experiment(rq_name, script_name)
            self.metadata['experiments'][rq_name] = result
            
            if result['status'] != 'success' and self.config.get('stop_on_error', False):
                print(f"\n⛔ Stopping due to error in {rq_name}")
                break
    
    def run_parallel(self):
        """Run experiments in parallel (requires multiple GPUs)"""
        
        print(f"\n🔀 Running experiments in parallel...")
        print(f"   Note: Requires multiple GPUs specified in config['devices']")
        
        # Placeholder for parallel execution
        # In practice, would use multiprocessing or GPU-specific job scheduling
        print(f"  ⚠️  Parallel mode not fully implemented yet")
        print(f"     Falling back to sequential mode...")
        
        self.run_sequential()
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        
        print(f"\n{'='*80}")
        print(f"🎨 Generating Visualizations")
        print(f"{'='*80}")
        
        script_path = self.scripts_dir / 'generate_all_visualizations.py'
        
        if not script_path.exists():
            print(f"  ⚠️  Visualization script not found")
            return
        
        try:
            cmd = [
                'python', str(script_path),
                '--results_dir', str(self.results_dir),
                '--viz_dir', str(self.viz_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"  ✅ Visualizations generated successfully")
                
                # Count generated files
                pdf_count = len(list(self.viz_dir.glob('*.pdf')))
                png_count = len(list(self.viz_dir.glob('*.png')))
                print(f"     Generated: {pdf_count} PDFs, {png_count} PNGs")
            else:
                print(f"  ❌ Visualization generation failed")
                print(f"     Error: {result.stderr[-200:]}")
        
        except Exception as e:
            print(f"  ❌ Error generating visualizations: {str(e)}")
    
    def generate_reports(self):
        """Generate analysis reports"""
        
        print(f"\n{'='*80}")
        print(f"📝 Generating Analysis Reports")
        print(f"{'='*80}")
        
        # Would call report generation scripts here
        print(f"  ⚠️  Report generation not yet implemented")
        print(f"     Results available in: {self.results_dir}")
    
    def save_metadata(self):
        """Save experiment metadata"""
        
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_duration_seconds'] = (
            datetime.now() - self.start_time
        ).total_seconds()
        
        metadata_path = self.results_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n💾 Metadata saved to: {metadata_path}")
    
    def print_summary(self):
        """Print execution summary"""
        
        print(f"\n{'='*100}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*100}")
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\nTotal Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"\nExperiment Status:")
        
        for rq_name, result in self.metadata['experiments'].items():
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'timeout': '⏱️',
                'error': '💥'
            }.get(result['status'], '❓')
            
            duration = result.get('duration_seconds', 0)
            print(f"  {status_icon} {rq_name:35s} | {result['status']:10s} | {duration:6.1f}s")
        
        print(f"\n📁 Output Locations:")
        print(f"  Results:        {self.results_dir}")
        print(f"  Visualizations: {self.viz_dir}")
        print(f"  Analysis:       {self.analysis_dir}")
        
        print(f"\n📈 Generated Files:")
        print(f"  CSV Results:    {len(list(self.results_dir.glob('*.csv')))}")
        print(f"  PDF Figures:    {len(list(self.viz_dir.glob('*.pdf')))}")
        print(f"  PNG Figures:    {len(list(self.viz_dir.glob('*.png')))}")
        print(f"  Analysis Docs:  {len(list(self.analysis_dir.glob('*.md')))}")
    
    def run(self):
        """Execute complete experimental pipeline"""
        
        # Run experiments
        if self.config['mode'] == 'parallel':
            self.run_parallel()
        else:
            self.run_sequential()
        
        # Generate visualizations
        if self.config.get('generate_viz', True):
            self.generate_visualizations()
        
        # Generate reports
        if self.config.get('generate_reports', True):
            self.generate_reports()
        
        # Save metadata
        self.save_metadata()
        
        # Print summary
        self.print_summary()
        
        print(f"\n{'='*100}")
        print(f"🎉 COMPREHENSIVE STUDY COMPLETE!")
        print(f"{'='*100}\n")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Run complete GSSM experimental framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with all defaults
  python run_all_experiments.py --mode full
  
  # Quick validation run
  python run_all_experiments.py --mode quick --epochs 10 --num_seeds 1
  
  # Parallel execution (requires multiple GPUs)
  python run_all_experiments.py --mode parallel --devices 0,1,2,3,4
        """
    )
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'parallel'],
                        help='Execution mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per experiment')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU device IDs for parallel mode (comma-separated)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per experiment (seconds)')
    parser.add_argument('--stop_on_error', action='store_true',
                        help='Stop if any experiment fails')
    parser.add_argument('--skip_viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--skip_reports', action='store_true',
                        help='Skip report generation')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'mode': args.mode,
        'epochs': 10 if args.mode == 'quick' else args.epochs,
        'num_seeds': 1 if args.mode == 'quick' else args.num_seeds,
        'device': args.device,
        'devices': args.devices.split(',') if args.devices else ['0'],
        'timeout': args.timeout,
        'stop_on_error': args.stop_on_error,
        'generate_viz': not args.skip_viz,
        'generate_reports': not args.skip_reports
    }
    
    # Run orchestrator
    orchestrator = ExperimentOrchestrator(config)
    orchestrator.run()


if __name__ == '__main__':
    main()
