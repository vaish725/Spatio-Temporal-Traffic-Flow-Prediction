"""
🚀 Quick Experiment Runner for Google Colab

This notebook cell runs multiple DCRNN configurations and compares results.
Just copy-paste this into a Colab cell and run!

Experiments:
1. Baseline (64-dim, 2 layers) - Your current model
2. Large Model (128-dim) - 15% better expected
3. Deep Model (3 layers) - 10% better expected
4. Learning Rate Decay - 8% better expected
5. Combined Best - 20% better expected
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd

# Define experiments
EXPERIMENTS = {
    "baseline": {
        "name": "Baseline (Current 64-dim)",
        "cmd": "python3 scripts/train.py --hidden_dim 64 --num_layers 2 --batch_size 64 --lr 0.001 --epochs 50 --device cuda --checkpoint_dir experiments/baseline"
    },
    "large_model": {
        "name": "Large Model (128-dim)",
        "cmd": "python3 scripts/train.py --hidden_dim 128 --num_layers 2 --batch_size 32 --lr 0.001 --epochs 50 --device cuda --checkpoint_dir experiments/large"
    },
    "deep_model": {
        "name": "Deep Model (3 layers)",
        "cmd": "python3 scripts/train.py --hidden_dim 64 --num_layers 3 --batch_size 64 --lr 0.001 --epochs 50 --device cuda --checkpoint_dir experiments/deep"
    },
    "lr_decay": {
        "name": "Learning Rate Decay",
        "cmd": "python3 scripts/train.py --hidden_dim 64 --num_layers 2 --batch_size 64 --lr 0.001 --lr_decay --lr_decay_rate 0.5 --epochs 50 --device cuda --checkpoint_dir experiments/lr"
    },
    "combined": {
        "name": "Combined (128-dim + 3 layers + LR decay)",
        "cmd": "python3 scripts/train.py --hidden_dim 128 --num_layers 3 --batch_size 32 --lr 0.001 --lr_decay --lr_decay_rate 0.3 --epochs 75 --device cuda --checkpoint_dir experiments/combined"
    }
}

def run_experiment(exp_name, exp_config):
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting: {exp_config['name']}")
    print(f"{'='*70}\n")
    
    # Create experiments directory
    os.makedirs('experiments', exist_ok=True)
    
    # Run training
    print(f"Command: {exp_config['cmd']}\n")
    start_time = datetime.now()
    result = subprocess.run(exp_config['cmd'], shell=True)
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"❌ Training failed for {exp_name}")
        return None
    
    print(f"\n✅ Training completed in {duration/60:.1f} minutes")
    
    # Get checkpoint directory from command
    checkpoint_dir = exp_config['cmd'].split('--checkpoint_dir ')[-1].split()[0]
    
    # Parse hidden_dim and num_layers from command
    cmd_parts = exp_config['cmd'].split()
    hidden_dim = int(cmd_parts[cmd_parts.index('--hidden_dim') + 1])
    num_layers = int(cmd_parts[cmd_parts.index('--num_layers') + 1])
    
    # Run evaluation
    eval_cmd = f"python3 scripts/evaluate.py --checkpoint {checkpoint_dir}/best_model.pt --hidden_dim {hidden_dim} --num_layers {num_layers} --output_dir {checkpoint_dir}/results --device cuda"
    
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
    
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    
    return {
        'name': exp_config['name'],
        'duration_min': duration / 60,
        'best_epoch': best_epoch,
        'mae': metrics['overall']['mae'],
        'rmse': metrics['overall']['rmse'],
        'mape': metrics['overall']['mape'],
        'mae_1step': metrics['horizons']['1_step']['mae'],
        'mae_12step': metrics['horizons']['12_step']['mae'],
        'checkpoint': checkpoint_dir
    }

# Main execution
print("""
╔═══════════════════════════════════════════════════════════════╗
║     DCRNN Experiment Runner - Compare Multiple Configs       ║
║                                                               ║
║  This will run 5 experiments to find the best configuration  ║
║  Total time: ~2-3 hours on Colab GPU (T4)                   ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Ask which experiments to run
print("Available experiments:")
for i, (key, config) in enumerate(EXPERIMENTS.items(), 1):
    print(f"  {i}. {config['name']}")

print("\n💡 Recommendation: Start with 'large_model' (15 min, high impact)")
print("   Then run 'combined' for best results (25 min)")

# Run selected experiments (modify as needed)
experiments_to_run = ['large_model', 'combined']  # Customize this list
print(f"\n🎯 Running experiments: {', '.join(experiments_to_run)}")
print("="*70)

results = []
for exp_name in experiments_to_run:
    if exp_name not in EXPERIMENTS:
        print(f"⚠️  Unknown experiment: {exp_name}")
        continue
    
    result = run_experiment(exp_name, EXPERIMENTS[exp_name])
    if result:
        results.append(result)

# Create comparison table
if results:
    print("\n" + "="*90)
    print("📊 EXPERIMENT COMPARISON RESULTS")
    print("="*90 + "\n")
    
    df = pd.DataFrame(results)
    df = df.sort_values('mae')
    
    # Format for display
    df_display = df.copy()
    df_display['duration_min'] = df_display['duration_min'].apply(lambda x: f"{x:.1f}")
    df_display['mae'] = df_display['mae'].apply(lambda x: f"{x:.4f}")
    df_display['rmse'] = df_display['rmse'].apply(lambda x: f"{x:.4f}")
    df_display['mape'] = df_display['mape'].apply(lambda x: f"{x:.2f}%")
    
    print(df_display.to_string(index=False))
    
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
    print(f"   Best Epoch: {best['best_epoch']}")
    print(f"   Training Time: {best['duration_min']:.1f} min")
    print(f"   Checkpoint: {best['checkpoint']}")
    
    # Calculate improvements over baseline
    if len(results) > 1 and any('Baseline' in r['name'] for r in results):
        baseline = next(r for r in results if 'Baseline' in r['name'])
        print(f"\n📈 IMPROVEMENTS OVER BASELINE")
        print("-" * 70)
        for r in results:
            if r['name'] != baseline['name']:
                improvement = ((baseline['mae'] - r['mae']) / baseline['mae']) * 100
                print(f"   {r['name']:40s} {improvement:+6.2f}%")
    
    print("\n" + "="*90)
    print("✅ Experiment comparison complete!")
    print("="*90)

print("""

💡 NEXT STEPS:
1. Check the comparison table above
2. Use the best checkpoint for final evaluation
3. Download results using the last cell of this notebook

📊 Visualizations are in each experiment's results/ folder
🎯 Best model checkpoint can be used for deployment
""")
