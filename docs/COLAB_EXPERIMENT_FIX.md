# ✅ Quick Fix: Running Experiments in Google Colab

**Problem**: `run_experiments.py` not found in Colab

**Solution**: Use the simplified approach below

---

## 🚀 Option 1: Quick Single Experiment (Recommended)

Just paste this into a Colab cell and run:

```python
# Test larger model (expected: 10-15% improvement)
!python3 scripts/train.py \
  --epochs 50 \
  --batch_size 32 \
  --hidden_dim 128 \
  --num_layers 2 \
  --lr 0.001 \
  --checkpoint_dir experiments/large_model \
  --device cuda

# Evaluate
!python3 scripts/evaluate.py \
  --checkpoint experiments/large_model/best_model.pt \
  --hidden_dim 128 \
  --num_layers 2 \
  --plot \
  --save_predictions \
  --output_dir experiments/large_model/results \
  --device cuda

# Compare results
import json

with open('results/metrics.json', 'r') as f:
    baseline = json.load(f)
with open('experiments/large_model/results/metrics.json', 'r') as f:
    large = json.load(f)

baseline_mae = baseline['overall']['mae']
large_mae = large['overall']['mae']
improvement = ((baseline_mae - large_mae) / baseline_mae) * 100

print(f"\n📊 RESULTS:")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Large Model MAE: {large_mae:.4f}")
print(f"Improvement: {improvement:+.2f}%")
```

---

## 🔬 Option 2: Run Multiple Experiments

Copy the entire file from `notebooks/colab_experiment_runner.py` into a Colab cell.

Or use this simplified version:

```python
import subprocess
import json
from datetime import datetime

experiments = {
    "large": {
        "name": "Large Model (128-dim)",
        "train": "python3 scripts/train.py --hidden_dim 128 --batch_size 32 --epochs 50 --device cuda --checkpoint_dir experiments/large",
        "eval": "python3 scripts/evaluate.py --checkpoint experiments/large/best_model.pt --hidden_dim 128 --output_dir experiments/large/results --device cuda"
    },
    "combined": {
        "name": "Combined Best (128-dim + 3 layers + LR decay)",
        "train": "python3 scripts/train.py --hidden_dim 128 --num_layers 3 --batch_size 32 --lr 0.001 --lr_decay --lr_decay_rate 0.3 --epochs 75 --device cuda --checkpoint_dir experiments/combined",
        "eval": "python3 scripts/evaluate.py --checkpoint experiments/combined/best_model.pt --hidden_dim 128 --num_layers 3 --output_dir experiments/combined/results --device cuda"
    }
}

results = []

for key, config in experiments.items():
    print(f"\n{'='*70}")
    print(f"🚀 {config['name']}")
    print(f"{'='*70}\n")
    
    # Train
    subprocess.run(config['train'], shell=True)
    
    # Evaluate
    subprocess.run(config['eval'], shell=True)
    
    # Load metrics
    with open(f"experiments/{key}/results/metrics.json", 'r') as f:
        metrics = json.load(f)
    
    results.append({
        'name': config['name'],
        'mae': metrics['overall']['mae'],
        'rmse': metrics['overall']['rmse'],
        'mape': metrics['overall']['mape']
    })

# Compare
print("\n" + "="*70)
print("📊 COMPARISON")
print("="*70)
for r in results:
    print(f"{r['name']:<40s} MAE: {r['mae']:.4f}")
```

---

## 📝 Option 3: Use Updated Colab Notebook

I've updated your Colab notebook with experiment cells. Just:

1. Go back to Colab
2. Re-upload the notebook or pull latest from GitHub:
   ```python
   !git pull origin main
   ```
3. Scroll to "6️⃣ Run Multiple Experiments (Optional)"
4. Run those cells

---

## 🎯 Recommended Approach

**For fastest results** (30 min):

```python
# 1. Large model experiment
!python3 scripts/train.py \
  --hidden_dim 128 \
  --batch_size 32 \
  --epochs 50 \
  --device cuda \
  --checkpoint_dir exp_large

# 2. Evaluate
!python3 scripts/evaluate.py \
  --checkpoint exp_large/best_model.pt \
  --hidden_dim 128 \
  --plot \
  --output_dir exp_large/results \
  --device cuda

# 3. Check improvement
import json
with open('exp_large/results/metrics.json') as f:
    print(json.load(f)['overall'])
```

**Expected result**: MAE 6.8-7.4 (vs your baseline 7.997)

---

## 💡 Why This Happened

The `run_experiments.py` script:
- Was designed to work from local machine
- Uses local file paths
- Requires pandas (which Colab has)

But the simplest approach in Colab is just running commands directly!

---

## ✅ Quick Commands Cheat Sheet

### Experiment 1: Larger Model
```bash
!python3 scripts/train.py --hidden_dim 128 --batch_size 32 --epochs 50 --device cuda --checkpoint_dir exp1
!python3 scripts/evaluate.py --checkpoint exp1/best_model.pt --hidden_dim 128 --device cuda --output_dir exp1/results
```

### Experiment 2: Deeper Model
```bash
!python3 scripts/train.py --hidden_dim 64 --num_layers 3 --epochs 50 --device cuda --checkpoint_dir exp2
!python3 scripts/evaluate.py --checkpoint exp2/best_model.pt --num_layers 3 --device cuda --output_dir exp2/results
```

### Experiment 3: With LR Decay
```bash
!python3 scripts/train.py --lr_decay --lr_decay_rate 0.3 --epochs 50 --device cuda --checkpoint_dir exp3
!python3 scripts/evaluate.py --checkpoint exp3/best_model.pt --device cuda --output_dir exp3/results
```

### Experiment 4: Combined Best
```bash
!python3 scripts/train.py --hidden_dim 128 --num_layers 3 --lr_decay --lr_decay_rate 0.3 --batch_size 32 --epochs 75 --device cuda --checkpoint_dir exp4
!python3 scripts/evaluate.py --checkpoint exp4/best_model.pt --hidden_dim 128 --num_layers 3 --device cuda --output_dir exp4/results
```

---

## 📊 Compare Results

After running experiments, compare them:

```python
import json
import pandas as pd

experiments = ['exp1', 'exp2', 'exp3', 'exp4']
results = []

for exp in experiments:
    with open(f'{exp}/results/metrics.json') as f:
        metrics = json.load(f)
        results.append({
            'experiment': exp,
            'mae': metrics['overall']['mae'],
            'rmse': metrics['overall']['rmse'],
            'mape': metrics['overall']['mape']
        })

df = pd.DataFrame(results).sort_values('mae')
print(df)
```

---

**Just use the simple approach - works perfectly in Colab!** 🚀
