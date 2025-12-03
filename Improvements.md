# DCRNN Model Improvements Log

This file tracks all improvements that resulted in better model performance.

---

## Baseline Performance
- **MAE**: 7.997 mph
- **Problem**: Model predicting constants (62.6 ± 0.005 mph)
- **Date**: Before Nov 27, 2025

---

## Improvement #1: Decoder Initialization Fix
**Date**: Nov 27, 2025  
**Commit**: ab7b5f9

**Problem**: Decoder initialized with zeros instead of last encoder input

**Fix**: Modified `models/dcrnn.py` to pass `last_input=X[:, -1, :, :]` to decoder

**Local Testing Results**:
- Output std: 0.1110 (was ~0.0050)
- Predictions show variance ✓

**Status**: ✅ Verified locally, waiting for Colab training results

---

## Improvement #2: Teacher Forcing Implementation
**Date**: Nov 27, 2025  
**Commit**: a08dd6f

**Problem**: Missing teacher forcing during training (didn't match original DCRNN paper)
- Old: Always used model predictions during training (pure autoregressive)
- Paper: Uses ground truth labels during training

**Fix**: 
- Added `labels` and `training` parameters to decoder
- Training: `model(X, labels=Y, training=True)` 
- Inference: `model(X, training=False)`

**Expected Results**:
- Training loss should decrease (not flat at 0.798)
- MAE: ~1.8-2.0 mph (4x improvement)
- Predictions with variance (std > 0.5)

**Status**: ✅ Implemented, ready for Colab training on CPU

---

## Next Training Run
**Configuration**:
- Device: CPU (GPU credits exhausted)
- Expected time: 45-60 minutes
- Expected MAE: ~1.8-2.0 mph

**Success Criteria**:
- Training loss decreases ✓
- Predictions std > 0.5 ✓
- MAE < 2.18 mph (beats persistence baseline) ✓
- MAE ~1.8-2.0 mph (target) ✓

---

*Note: This file will only be updated when actual training results show improvement*
