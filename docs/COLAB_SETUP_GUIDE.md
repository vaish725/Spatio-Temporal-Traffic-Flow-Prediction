# 🚀 Google Colab Setup Guide - DCRNN Training

**Fast GPU Training**: Train your DCRNN model 10-30x faster on Google Colab!

---

## 📋 Prerequisites

1. **Google Account** (for Colab access)
2. **GitHub Repository** (your code must be on GitHub)
3. **Internet Connection**

---

## 🎯 Quick Start (3 Steps)

### Step 1: Push Your Code to GitHub

```bash
# Navigate to your project
cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/AdvML/Spatio-Temporal_Traffic_Flow_Prediction"

# Initialize git if not already done
git init
git add .
git commit -m "Add DCRNN implementation for Colab training"

# Push to GitHub (if not already pushed)
git remote add origin https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction.git
git branch -M main
git push -u origin main
```

### Step 2: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `notebooks/DCRNN_Training_Colab.ipynb`

**OR** open directly from GitHub:
1. Click **File** → **Open notebook**
2. Select **GitHub** tab
3. Enter: `vaish725/Spatio-Temporal-Traffic-Flow-Prediction`
4. Select `notebooks/DCRNN_Training_Colab.ipynb`

### Step 3: Enable GPU & Run

1. **Enable GPU**: `Runtime` → `Change runtime type` → Set `Hardware accelerator` to **GPU** → Save
2. **Run all cells**: `Runtime` → `Run all`
3. **Wait**: ~10-20 minutes (vs 2-4 hours on CPU!)
4. **Download results**: Last cell will download a ZIP with all results

---

## 🖥️ Colab vs Local Comparison

| Aspect | Local (CPU) | Google Colab (GPU) |
|--------|-------------|-------------------|
| **Training Time** | 2-4 hours | 10-20 minutes |
| **Speedup** | 1x | **10-30x** |
| **Cost** | Free | Free (T4 GPU) |
| **Setup** | None | 5 minutes |
| **GPU Type** | None | T4 (16GB VRAM) |
| **Memory** | Your RAM | 12GB RAM + 16GB VRAM |

---

## 📦 What the Notebook Does

### Automatic Setup
```
1. ✅ Clones your GitHub repository
2. ✅ Installs PyTorch, torch-geometric, dependencies
3. ✅ Verifies GPU is available
4. ✅ Checks project structure
```

### Training
```
5. ✅ Trains DCRNN with GPU acceleration
6. ✅ Saves checkpoints (best & final models)
7. ✅ Tracks training history
8. ✅ Creates training curves plot
```

### Evaluation
```
9. ✅ Evaluates on test set
10. ✅ Computes MAE, RMSE, MAPE
11. ✅ Multi-horizon metrics (1, 3, 6, 12-step)
12. ✅ Generates prediction visualizations
```

### Download
```
13. ✅ Packages all results into ZIP
14. ✅ Downloads to your computer
```

---

## 🎮 Using the Notebook

### Basic Training (Default)

Just run all cells! The default configuration:
```python
--epochs 100
--batch_size 64
--hidden_dim 64
--num_layers 2
--lr 0.001
--patience 15
```

### Advanced Configurations

The notebook includes 3 optional configurations:

**1. Larger Model**
```python
--hidden_dim 128
--num_layers 3
--batch_size 32  # Reduced for GPU memory
```

**2. Learning Rate Decay**
```python
--lr_decay
--lr_decay_rate 0.5
--patience 20
```

**3. More Diffusion Hops**
```python
--max_diffusion_step 3  # Default is 2
```

### Custom Training

Modify the training cell:
```python
!python3 scripts/train.py \
  --epochs 150 \
  --batch_size 128 \
  --hidden_dim 96 \
  --num_layers 2 \
  --lr 0.0005 \
  --device cuda
```

---

## 📊 Expected Output

### Training Logs
```
Epoch 1/100: 100%|██████████| 200/200 [00:15<00:00, 13.2batch/s]
Train Loss: 1.234, Val Loss: 0.987, Val MAE: 8.234

Best model saved! (val_loss: 0.987)
```

### Final Summary
```
🎯 Training Summary
==================================================
Total epochs trained: 22
Best validation loss: 0.7997
Best epoch: 7
Training time: 12.45 minutes

🎯 Test Set Performance
==================================================
Overall MAE:  7.9720
Overall RMSE: 9.9843
Overall MAPE: 17.53%
```

### Downloaded ZIP Contents
```
dcrnn_results_20251126_143022.zip
├── checkpoints/
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
├── results/
│   ├── metrics.json
│   ├── predictions.png
│   ├── horizon_metrics.png
│   ├── predictions.npy
│   └── targets.npy
└── training_curves.png
```

---

## 🔧 Troubleshooting

### Problem: No GPU Available

**Symptoms**:
```
CUDA available: False
⚠️ WARNING: No GPU detected
```

**Solution**:
1. Click `Runtime` → `Change runtime type`
2. Set `Hardware accelerator` to **GPU**
3. Click **Save**
4. Re-run all cells

---

### Problem: Session Disconnected

**Symptoms**: "Runtime disconnected" message

**Solution**:
1. Click **Reconnect**
2. Re-run all cells from the beginning
3. Your repository will be re-cloned automatically

**Prevention**: Colab disconnects after:
- 12 hours of activity
- 90 minutes of inactivity (free tier)

---

### Problem: Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solution - Reduce batch size**:
```python
!python3 scripts/train.py \
  --batch_size 32 \    # Instead of 64
  --hidden_dim 64 \
  --device cuda
```

**Or reduce model size**:
```python
!python3 scripts/train.py \
  --batch_size 64 \
  --hidden_dim 32 \    # Instead of 64
  --device cuda
```

---

### Problem: Clone Failed

**Symptoms**:
```
fatal: repository not found
```

**Solution**:
1. Make sure your repository is **public** on GitHub
2. Or provide authentication for private repos
3. Update the clone URL in the notebook:
```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

---

### Problem: Import Errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'models'
```

**Solution**:
Make sure the `%cd` command ran successfully:
```python
%cd Spatio-Temporal-Traffic-Flow-Prediction
```

Check current directory:
```python
!pwd
!ls -la
```

---

## 💡 Tips & Best Practices

### 1. Save Results Frequently
- Download the ZIP after each successful training
- Colab sessions can disconnect unexpectedly

### 2. Mount Google Drive (Optional)

Add this cell at the beginning to auto-save results:
```python
from google.colab import drive
drive.mount('/content/drive')

# Save results to Drive
!mkdir -p /content/drive/MyDrive/DCRNN_Results
!cp -r checkpoints /content/drive/MyDrive/DCRNN_Results/
!cp -r results /content/drive/MyDrive/DCRNN_Results/
```

### 3. Monitor GPU Usage

Add this cell to check GPU utilization:
```python
!nvidia-smi
```

### 4. Compare Multiple Runs

Train with different configs and compare:
```python
# Run 1: Default
!python3 scripts/train.py --checkpoint_dir checkpoints_default

# Run 2: Large model
!python3 scripts/train.py --hidden_dim 128 --checkpoint_dir checkpoints_large

# Run 3: More layers
!python3 scripts/train.py --num_layers 3 --checkpoint_dir checkpoints_deep
```

### 5. Resume Training

If disconnected, you can resume (if checkpoints saved to Drive):
```python
!python3 scripts/train.py \
  --resume checkpoints/final_model.pt \
  --epochs 200
```

---

## 📈 Performance Benchmarks

### Free Tier GPUs

| GPU Type | Training Time | Availability |
|----------|--------------|--------------|
| **T4** | 10-15 min | Most common |
| **P100** | 8-12 min | Occasional |
| **K80** | 15-25 min | Rare (older) |

### Colab Pro GPUs

| GPU Type | Training Time | Cost |
|----------|--------------|------|
| **V100** | 5-8 min | $9.99/month |
| **A100** | 3-5 min | $49.99/month |

---

## 🔄 Workflow Comparison

### Option 1: Local Training (Current)
```
1. Write code locally ✅
2. Train on CPU (2-4 hours) ⏰
3. Results available immediately ✅
```

### Option 2: Colab Training (New)
```
1. Write code locally ✅
2. Push to GitHub (1 min) 📤
3. Run on Colab GPU (10-20 min) ⚡
4. Download results (1 min) 📥
```

**Total time saved**: ~2-3 hours per training run!

---

## 🎓 Why Use Colab?

### Advantages ✅
1. **Free GPU access** - T4 GPU with 16GB VRAM
2. **No setup** - Everything pre-installed
3. **Fast training** - 10-30x speedup
4. **Experiment easily** - Try multiple configs quickly
5. **No local hardware stress** - Your laptop stays cool

### Limitations ⚠️
1. **Session timeouts** - 12 hours max
2. **Disconnections** - Can happen randomly
3. **No persistent storage** - Must download results
4. **Internet required** - Can't work offline
5. **Queue times** - GPU availability varies

---

## 📚 Additional Resources

### Colab Documentation
- [Official Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [Using GPUs](https://colab.research.google.com/notebooks/gpu.ipynb)
- [External Data](https://colab.research.google.com/notebooks/io.ipynb)

### Alternatives to Colab
1. **Kaggle Notebooks** - Similar to Colab, 30h/week GPU
2. **AWS SageMaker** - Professional, paid
3. **Paperspace Gradient** - Similar to Colab
4. **University Cluster** - Check if GWU has GPU servers

---

## ✅ Checklist for First Run

Before running the notebook:
- [ ] Code pushed to GitHub (public repository)
- [ ] Notebook uploaded to Colab
- [ ] GPU enabled (Runtime → Change runtime type)
- [ ] GitHub URL updated in notebook (if different)
- [ ] Ready to wait 10-20 minutes

After successful training:
- [ ] Training curves look reasonable
- [ ] Test metrics computed
- [ ] Visualizations generated
- [ ] ZIP file downloaded
- [ ] Results extracted locally
- [ ] Checkpoints backed up

---

## 🎉 Summary

**Before**: 2-4 hours training on local CPU  
**After**: 10-20 minutes training on Colab GPU  
**Speedup**: 10-30x faster  
**Cost**: $0 (free tier)  

**You can now**:
- ✅ Train models 10-30x faster
- ✅ Experiment with multiple configurations
- ✅ Save time and laptop battery
- ✅ Access better hardware for free

---

**Happy training! 🚀**
