#!/usr/bin/env python3
"""
Quick Experiment Runner - Compare Multiple DCRNN Configurations

Usage:
    python3 scripts/run_experiments.py --gpu  # Run on GPU
    python3 scripts/run_experiments.py --cpu  # Run on CPU (slower)
"""

import argparse
import subprocess
import json
import os
from datetime import datetime
import pandas as pd


EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (Current)",
        "args": "--hidden_dim 64 --num_layers 2 --batch_size 64 --lr 0.001 --epochs 100"
    },
    "large_model": {
        "name": "Large Model (128-dim)",
        "args": "--hidden_dim 128 --num_layers 2 --batch_size 32 --lr 0.001 --epochs 100"
    },
    "deep_model": {
        "name": "Deep Model (3 layers)",
        "args": "--hidden_dim 64 --num_layers 3 --batch_size 64 --lr 0.001 --epochs 100"
    },
    "more_diffusion": {
        "name": "More Diffusion (K=3)",
        "args": "--hidden_dim 64 --num_layers 2 --max_diffusion_step 3 --batch_size 64 --lr 0.001 --epochs 100"
    },
    "lr_decay": {
        "name": "Learning Rate Decay",
        "args": "--hidden_dim 64 --num_layers 2 --batch_size 64 --lr 0.001 --lr_decay --lr_decay_rate 0.5 --epochs 100"
    },
    "combined": {
        "name": "Combined Best Settings",
        "args": "--hidden_dim 128 --num_layers 3 --max_diffusion_step 3 --batch_size 32 --lr 0.001 --lr_decay --lr_decay_rate 0.5 --epochs 150"
    }
}


def run_experiment(exp_name, exp_config, device='cuda'):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {exp_config['name']}")
    print(f"{'='*70}")
    
    checkpoint_dir = f"experiments/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Build training command
    cmd = f"python3 scripts/train.py {exp_config['args']} --checkpoint_dir {checkpoint_dir} --device {device}"
    
    print(f"Command: {cmd}")
    print()
    
    # Run training
    start_time = datetime.now()
    result = subprocess.run(cmd, shell=True)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"❌ Training failed for {exp_name}")
        return None
    
    print(f"✅ Training completed in {duration:.1f} seconds")
    
    # Parse hidden_dim and num_layers from args
    args_dict = {}
    for arg in exp_config['args'].split():
        if arg.startswith('--'):
            key = arg
        else:
            args_dict[key] = arg
    
    hidden_dim = int(args_dict.get('--hidden_dim', 64))
    num_layers = int(args_dict.get('--num_layers', 2))
    
    # Run evaluation
    eval_cmd = f"python3 scripts/evaluate.py --checkpoint {checkpoint_dir}/best_model.pt --hidden_dim {hidden_dim} --num_layers {num_layers} --output_dir {checkpoint_dir}/results --device {device}"
    
    print(f"\n📊 Running evaluation...")
    result = subprocess.run(eval_cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed for {exp_name}")
        return None
    
    # Load results
    metrics_file = f"{checkpoint_dir}/results/metrics.json"
    if not os.path.exists(metrics_file):
        print(f"❌ Metrics file not found: {metrics_file}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Load training history
    history_file = f"{checkpoint_dir}/training_history.json"
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    return {
        'name': exp_config['name'],
        'duration': duration,
        'best_epoch': len([v for v in history['val_loss']]) - 15,  # Approximate
        'mae': metrics['overall']['mae'],
        'rmse': metrics['overall']['rmse'],
        'mape': metrics['overall']['mape'],
        'mae_1': metrics['horizons']['1_step']['mae'],
        'mae_12': metrics['horizons']['12_step']['mae'],
        'checkpoint': checkpoint_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Run DCRNN experiments')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS.keys()) + ['all'],
                       default=['all'], help='Which experiments to run')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline experiment (if already run)')
    
    args = parser.parse_args()
    
    # Determine which experiments to run
    if 'all' in args.experiments:
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = args.experiments
    
    if args.skip_baseline and 'baseline' in exp_names:
        exp_names.remove('baseline')
        print("⏭️  Skipping baseline experiment")
    
    print(f"\n🎯 Running {len(exp_names)} experiments on {args.device.upper()}")
    print(f"Experiments: {', '.join(exp_names)}")
    
    # Run experiments
    results = []
    for exp_name in exp_names:
        result = run_experiment(exp_name, EXPERIMENTS[exp_name], args.device)
        if result:
            results.append(result)
    
    # Create comparison table
    if results:
        print("\n" + "="*90)
        print("📊 EXPERIMENT COMPARISON")
        print("="*90)
        
        df = pd.DataFrame(results)
        df = df.sort_values('mae')
        
        print(df.to_string(index=False))
        
        # Save to CSV
        output_file = f"experiments/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Print best model
        best = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['name']}")
        print(f"   MAE: {best['mae']:.4f}")
        print(f"   RMSE: {best['rmse']:.4f}")
        print(f"   MAPE: {best['mape']:.2f}%")
        print(f"   Checkpoint: {best['checkpoint']}")
        
        # Calculate improvements
        if 'baseline' in [r['name'] for r in results]:
            baseline_result = [r for r in results if 'Baseline' in r['name']][0]
            improvements = []
            for r in results:
                if r['name'] != baseline_result['name']:
                    improvement = ((baseline_result['mae'] - r['mae']) / baseline_result['mae']) * 100
                    improvements.append({
                        'name': r['name'],
                        'improvement': f"{improvement:+.2f}%"
                    })
            
            if improvements:
                print(f"\n📈 IMPROVEMENTS OVER BASELINE (MAE)")
                print("-" * 60)
                for imp in improvements:
                    print(f"   {imp['name']}: {imp['improvement']}")


if __name__ == '__main__':
    main()
