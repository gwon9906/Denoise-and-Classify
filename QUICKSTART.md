# Quick Start Guide

## Step-by-Step Execution

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn tqdm
```

### Step 1: Data Preprocessing (1-2 hours)
```bash
jupyter notebook data_preprocessing.ipynb
```
**Output:**
- `data/x_train_augmented.npy` (150,000 samples)
- `data/x_test_augmented.npy` (30,000 samples)
- Clean references and metadata CSV files

**Check:** Verify file sizes
```python
import numpy as np
x = np.load('data/x_train_augmented.npy')
print(f"Shape: {x.shape}")  # Should be (150000, 32, 32, 3)
```

---

### Step 2: Training Part 1 - Setup (10 minutes)
```bash
jupyter notebook train_all_models_part1.ipynb
```
**What it does:**
- GPU memory configuration
- Load augmented data
- Prepare flattened data for BAM models
- Define callbacks and utility functions

**Check:** GPU memory should be limited to 8GB

---

### Step 3: Training Part 2 - Sequential Models (12-18 hours)
```bash
jupyter notebook train_all_models_part2.ipynb
```
**Trains:**
1. Sequential BAM (Stage 1 + Stage 2)
2. Sequential CAE (Stage 1 + Stage 2)
3. Sequential U-Net (Stage 1 + Stage 2)

**Output per model:**
- Weights: `weights/sequential_*_restore.keras`, `weights/sequential_*_classification.keras`
- History: `history/sequential_*_stage1_history.pkl`, `history/sequential_*_stage2_history.pkl`
- Results: `results/sequential_*_results.json`

**Monitor:** Check TensorBoard logs
```bash
tensorboard --logdir=logs
```

---

### Step 4: Training Part 3 - MTL Models (9-15 hours)
```bash
jupyter notebook train_all_models_part3.ipynb
```
**Trains:**
1. MTL BAM
2. MTL CAE
3. MTL U-Net

**Output per model:**
- Weights: `weights/mtl_*.keras`
- History: `history/mtl_*_history.pkl`
- Results: `results/mtl_*_results.json`

---

### Step 5: Training Part 4 - Overall Comparison (30 minutes)
```bash
jupyter notebook train_all_models_part4.ipynb
```
**What it does:**
- Load all results
- Create comprehensive comparison table
- Generate performance charts
- Plot training curves

**Output:**
- `results/comprehensive_results.csv`
- `results/performance_comparison.png`
- `results/training_curves_mtl.png`

---

### Step 6: Training Part 5 - Detailed Analysis (1-2 hours)
```bash
jupyter notebook train_all_models_part5.ipynb
```
**What it does:**
- Evaluate by noise type (Gaussian, S&P, Burst)
- Evaluate by SNR level (-30, -25, -20, -15, -10 dB)
- Statistical comparison (Sequential vs MTL)
- Extract key findings

**Output:**
- `results/noise_type_results.json`
- `results/snr_level_results.json`
- `results/noise_type_summary.csv`
- `results/snr_level_summary.csv`
- `results/key_findings.txt`
- `results/noise_type_analysis.png`
- `results/snr_level_analysis.png`

---

## Total Timeline

| Phase | Duration | Can Run Overnight? |
|-------|----------|-------------------|
| Data Preprocessing | 1-2 hours | No (need to check) |
| Part 1 (Setup) | 10 min | No |
| Part 2 (Sequential) | 12-18 hours | ✅ Yes |
| Part 3 (MTL) | 9-15 hours | ✅ Yes |
| Part 4 (Comparison) | 30 min | No |
| Part 5 (Analysis) | 1-2 hours | ✅ Yes |
| **TOTAL** | **24-38 hours** | - |

**Recommended Schedule:**
- Day 1 Morning: Data preprocessing + Part 1
- Day 1 Evening: Start Part 2 (run overnight)
- Day 2 Morning: Check Part 2, start Part 3 (run overnight)
- Day 3 Morning: Run Part 4 + Part 5

---

## Troubleshooting

### Out of Memory Error
```python
# In Part 1, reduce batch size
BATCH_SIZE = 64  # or even 32

# Or reduce GPU memory limit
setup_gpu_memory(memory_limit_mb=6144)  # 6GB instead of 8GB
```

### Kernel Crash
- Restart kernel after each Part
- Clear outputs before running next Part
- Check GPU temperature

### Model Not Found Error
```bash
# Make sure models/ directory contains:
ls models/
# Should show: bam_sequential.py, bam_mtl.py, cae_sequential.py, 
#              cae_mtl.py, unet_sequential.py, unet_mtl.py
```

### Slow Training
- Verify GPU is being used:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```
- Check GPU utilization:
```bash
nvidia-smi
```

---

## Expected Results

### Restoration Performance (PSNR)
```
Sequential U-Net:  17.5 - 18.5 dB (Best)
MTL U-Net:         17.0 - 18.0 dB
Sequential CAE:    16.5 - 17.5 dB
MTL CAE:           16.0 - 17.0 dB
Sequential BAM:    15.5 - 16.5 dB
MTL BAM:           15.0 - 16.0 dB
```

### Classification Performance (Accuracy)
```
MTL U-Net:         0.78 - 0.82 (Best)
MTL CAE:           0.76 - 0.80
Sequential U-Net:  0.75 - 0.79
MTL BAM:           0.74 - 0.78
Sequential CAE:    0.73 - 0.77
Sequential BAM:    0.70 - 0.74
```

### By Noise Type (Expected Difficulty)
```
Easiest:  Gaussian
Medium:   Salt & Pepper
Hardest:  Burst
```

### By SNR Level (Expected Improvement)
```
-30 dB: Lowest performance
-25 dB: Slight improvement
-20 dB: Noticeable improvement
-15 dB: Good performance
-10 dB: Best performance (Highest PSNR/Accuracy)
```

---

## Verification Checklist

After completing all parts, verify:

- [ ] All 6 models trained successfully
- [ ] All weights saved in `weights/`
- [ ] All histories saved in `history/`
- [ ] All results saved in `results/`
- [ ] Comprehensive results CSV created
- [ ] Performance comparison charts generated
- [ ] Noise type analysis completed
- [ ] SNR level analysis completed
- [ ] Key findings extracted

---

## Next Steps

1. **Review Results**: Check `results/key_findings.txt`
2. **Analyze Charts**: Review all PNG files in `results/`
3. **Statistical Tests**: Run t-tests or Wilcoxon tests for significance
4. **Paper Writing**: Use tables and charts for your paper
5. **Additional Experiments**: Try different hyperparameters if needed

---

## Contact

If you encounter any issues:
1. Check the error message carefully
2. Review the troubleshooting section
3. Verify prerequisites and file structure
4. Create an issue with error details

Good luck! 🚀
