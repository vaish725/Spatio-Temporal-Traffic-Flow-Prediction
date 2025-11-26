# 🚀 COLAB QUICK START - 3 Steps to GPU Training

## Step 1: Push to GitHub (30 seconds)

```bash
cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/AdvML/Spatio-Temporal_Traffic_Flow_Prediction"
git add .
git commit -m "Add Colab notebook for GPU training"
git push
```

---

## Step 2: Open in Colab (1 minute)

### Method A: Direct Upload
1. Go to https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Upload `notebooks/DCRNN_Training_Colab.ipynb`

### Method B: From GitHub
1. Go to https://colab.research.google.com/
2. Click **File** → **Open notebook** → **GitHub** tab
3. Paste: `vaish725/Spatio-Temporal-Traffic-Flow-Prediction`
4. Select: `notebooks/DCRNN_Training_Colab.ipynb`

---

## Step 3: Enable GPU & Run (10-20 minutes)

1. **Enable GPU**: 
   - Click `Runtime` → `Change runtime type`
   - Set `Hardware accelerator` to **GPU**
   - Click **Save**

2. **Run all cells**: 
   - Click `Runtime` → `Run all`
   - Or press `Ctrl+F9` (Windows) / `Cmd+F9` (Mac)

3. **Wait**: ~10-20 minutes (grab coffee ☕)

4. **Download**: Last cell downloads ZIP with all results

---

## 📊 What You Get

```
dcrnn_results_YYYYMMDD_HHMMSS.zip
├── checkpoints/
│   ├── best_model.pt           # Best validation model
│   ├── final_model.pt          # Final epoch model
│   └── training_history.json   # All training metrics
├── results/
│   ├── metrics.json            # Test set metrics
│   ├── predictions.png         # Sample predictions
│   ├── horizon_metrics.png     # Multi-horizon plot
│   ├── predictions.npy         # All predictions
│   └── targets.npy            # Ground truth
└── training_curves.png         # Training/validation curves
```

---

## ⚡ Speed Comparison

| Hardware | Time | Speedup |
|----------|------|---------|
| MacBook CPU | 2-4 hours | 1x |
| Colab T4 GPU | **10-20 min** | **10-30x** 🚀 |
| Colab V100 | 5-8 min | 15-50x |

---

## 🔧 Common Issues & Fixes

### ❌ "No GPU detected"
**Fix**: `Runtime` → `Change runtime type` → Select **GPU** → Save → Re-run all

### ❌ "Out of memory" 
**Fix**: Change batch size:
```python
!python3 scripts/train.py --batch_size 32  # Instead of 64
```

### ❌ "Repository not found"
**Fix**: Make sure your GitHub repo is **public** or update URL in notebook

### ❌ "Session disconnected"
**Fix**: Click **Reconnect** → Re-run all cells (automatic re-clone)

---

## 💡 Pro Tips

### Save to Google Drive (Auto-backup)
Add this cell after cloning:
```python
from google.colab import drive
drive.mount('/content/drive')

# Results will auto-save to Drive
!mkdir -p /content/drive/MyDrive/DCRNN_Results
!cp -r checkpoints /content/drive/MyDrive/DCRNN_Results/
```

### Monitor GPU Usage
```python
!nvidia-smi
```

### Try Different Configs
```python
# Larger model
!python3 scripts/train.py --hidden_dim 128 --num_layers 3

# More diffusion hops
!python3 scripts/train.py --max_diffusion_step 3

# Learning rate decay
!python3 scripts/train.py --lr_decay --lr_decay_rate 0.5
```

---

## 📚 Full Documentation

- **Setup Guide**: `docs/COLAB_SETUP_GUIDE.md` (detailed troubleshooting)
- **Notebook**: `notebooks/DCRNN_Training_Colab.ipynb` (fully commented)
- **Results Summary**: `docs/PROJECT_COMPLETION_SUMMARY.md`

---

## ✅ Checklist

**Before running**:
- [ ] Code pushed to GitHub
- [ ] Notebook opened in Colab
- [ ] GPU enabled
- [ ] Have 15 minutes free time

**After running**:
- [ ] Training completed successfully
- [ ] Metrics look reasonable
- [ ] ZIP file downloaded
- [ ] Results extracted locally

---

**Need Help?** See `docs/COLAB_SETUP_GUIDE.md` for detailed troubleshooting

**Time Investment**: 5 min setup → 10-20 min training → 2+ hours saved! ⏰
