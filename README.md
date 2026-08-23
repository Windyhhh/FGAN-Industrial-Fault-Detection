# 🏭 FGAN Industrial Fault Detection | FGAN (KDE) 工业过程故障检测系统

> **Feature Generative Adversarial Network + Kernel Density Estimation for industrial process fault detection. Tested on Tennessee Eastman Process (TEP) with 21 fault types, complete visualization and threshold analysis.**
>
> 特征生成对抗网络 + 核密度估计，用于工业过程故障检测。在田纳西伊士曼过程（TEP）上测试，覆盖 21 种故障类型，完整可视化与阈值分析。

---

## 🌟 Why This Project? | 项目亮点

Industrial process fault detection is critical for safe and efficient manufacturing. Traditional methods (PCA, PLS) struggle with nonlinear, high-dimensional process data. This project implements a **Feature Generative Adversarial Network (FGAN)** combined with **Kernel Density Estimation (KDE)** for robust fault detection on the **Tennessee Eastman Process (TEP)** benchmark — covering **21 fault types** with complete per-fault visualization, score distributions, and threshold analysis.

工业过程故障检测对于安全高效的制造至关重要。传统方法（PCA、PLS）在处理非线性、高维过程数据时存在困难。本项目实现了**特征生成对抗网络（FGAN）** 结合 **核密度估计（KDE）**，在**田纳西伊士曼过程（TEP）** 基准上进行鲁棒故障检测——覆盖 **21 种故障类型**，包含完整的逐故障可视化、分数分布和阈值分析。

| Feature | Details |
|---------|---------|
| **Method** | FGAN (Feature GAN) + KDE (Kernel Density Estimation) |
| **Dataset** | Tennessee Eastman Process (TEP) |
| **Fault Types** | 21 (d01–d21) + normal (d00) |
| **Variables** | 52 process variables |
| **Training Data** | 500 samples (normal operation) |
| **Test Data** | 960 samples per fault type |
| **Detection Score** | Reconstruction-based anomaly score |
| **Threshold** | KDE-based adaptive threshold |
| **Visualization** | Per-fault: time series, score distribution, continuous scores |
| **Analysis** | Boxplot comparison, key variable analysis, separation comparison |

---

## 🏗️ Architecture | 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│              TEP Industrial Process Data                       │
│         52 variables × 500 (train) / 960 (test) samples      │
│         Normal (d00) + 21 Fault Types (d01–d21)              │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing & Scaling                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Min-Max normalization (fitted on normal training)    │  │
│  │  • Save scaler.pkl for inference consistency             │  │
│  │  • Sliding window or sample-based input                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Generator (G)  │ │  Discriminator(D)│ │ Feature Extractor │
│                  │ │                  │ │    (FENETFIL)     │
│  • Encoder-Decoder│ │  • Real vs Fake  │ │                  │
│  • Reconstruct    │ │  • Feature match  │ │  • Key variable   │
│    normal samples │ │  • Adversarial    │ │    identification │
│                  │ │    training         │ │  • Fault signature│
└────────┬─────────┘ └──────────────────┘ └──────────────────┘
         │ Reconstruction error
         ▼
┌─────────────────────────────────────────────────────────────┐
│              Anomaly Score Computation                         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  score = ||x - G(x)||²  (reconstruction error)         │  │
│  │  Higher score → more likely fault                         │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              KDE-based Threshold Estimation                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Fit KDE on normal training scores                     │  │
│  │  • Set threshold at 95th/99th percentile                 │  │
│  │  • Adaptive per-fault threshold adjustment                │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Fault Detection & Visualization                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  • Per-fault: time series, score distribution,           │  │
│  │    continuous scores, detection results CSV               │  │
│  │  • Analysis: boxplot comparison, key variables,           │  │
│  │    separation comparison                                   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Tennessee Eastman Process (TEP) | 田纳西伊士曼过程

### Dataset Overview | 数据集概览

The Tennessee Eastman Process is a widely used benchmark for process monitoring and fault detection, based on a real industrial chemical process.

| Property | Value |
|----------|-------|
| **Process variables** | 52 (22 continuous measurements + 19 composition + 11 manipulated) |
| **Training samples** | 500 (normal operation, d00) |
| **Test samples** | 960 per fault type |
| **Fault types** | 21 (d01–d21) |
| **Fault introduction** | At sample 160 (out of 960) |

### Fault Types | 故障类型

| ID | Fault Description | Type |
|----|-------------------|------|
| d01 | A/C feed ratio change (B composition constant) | Step |
| d02 | B composition change (A/C ratio constant) | Step |
| d03 | D feed temperature change | Step |
| d04 | Reactor cooling water inlet temperature | Step |
| d05 | Condenser cooling water inlet temperature | Step |
| d06 | A feed loss | Step |
| d07 | C header pressure loss (reduced availability) | Step |
| d08 | A, B, C feed composition | Random variation |
| d09 | D feed temperature | Random variation |
| d10 | C feed temperature | Random variation |
| d11 | Reactor cooling water inlet temperature | Random variation |
| d12 | Condenser cooling water inlet temperature | Random variation |
| d13 | Reaction kinetics | Slow drift |
| d14 | Reactor cooling water valve | Sticking |
| d15 | Condenser cooling water valve | Sticking |
| d16–d21 | Unknown / complex faults | Various |

---

## 🚀 Quick Start | 快速开始

### Installation | 安装

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn
```

### Project Structure | 项目结构

```
FGAN-Industrial-Fault-Detection/
├── src/
│   ├── config.py                    # Configuration
│   ├── FBGAN.py                     # Feature GAN model (22KB)
│   ├── fault_detection1.py          # Main fault detection pipeline (20KB)
│   └── FENETFIL.py                  # Feature extraction network
├── data/
│   ├── d00_train.csv                # Normal training data (500 samples)
│   ├── d00_test.csv                 # Normal test data
│   └── d01_test.csv – d21_test.csv # 21 fault type test data
├── models/
│   ├── complete_model.pth           # Trained FGAN model
│   ├── recon_optimized_checkpoint_epoch_100.pth
│   ├── recon_optimized_checkpoint_epoch_200.pth
│   ├── recon_optimized_checkpoint_epoch_300.pth
│   └── scaler.pkl                   # Data scaler
├── results/
│   ├── all_faults_final_test.csv    # Summary of all fault detections
│   ├── analysis/                     # Comparative analysis plots
│   │   ├── boxplot_comparison.png
│   │   ├── score_distributions.png
│   │   ├── separation_comparison.png
│   │   └── fault{3,9}_key_variables.png
│   └── fault_01/ – fault_21/        # Per-fault results
│       ├── results_reconstruction_based.csv
│       ├── continuous_scores_reconstruction_based.png
│       ├── distribution_reconstruction_based.png
│       └── timeseries_reconstruction_based.png
├── docs/
│   ├── RESULTS.md                    # Detailed results
│   ├── THRESHOLD_ADJUSTMENT_EXPLANATION.md
│   └── WORK_SUMMARY.md
├── test_all_faults.py                # Run all 21 fault tests
├── 爆款博客.md                        # Technical blog
├── 项目说明.txt
├── requirements.txt
├── .gitignore
└── README.md
```

### Run All Fault Tests | 运行所有故障测试

```bash
python test_all_faults.py
```

This will:
1. Load the trained FGAN model and scaler
2. Process each of the 21 fault types (d01–d21)
3. Compute reconstruction-based anomaly scores
4. Apply KDE-based threshold for detection
5. Save per-fault results (CSV + visualizations)
6. Generate summary and comparative analysis

### Run Single Fault Detection | 运行单故障检测

```python
from src.fault_detection1 import FaultDetector

detector = FaultDetector(model_path='models/complete_model.pth',
                          scaler_path='models/scaler.pkl')

# Detect fault d03
results = detector.detect('data/d03_test.csv', fault_id='d03')
print(f"Detection rate: {results['detection_rate']:.2%}")
```

---

## 🔬 Method Details | 方法细节

### FGAN (Feature Generative Adversarial Network) | 特征生成对抗网络

The generator learns to reconstruct normal process data from a compressed latent representation. The discriminator distinguishes real normal samples from reconstructed ones. During inference, **reconstruction error** serves as the anomaly score — faults cause high reconstruction error because the generator only learned normal patterns.

生成器学习从压缩的潜在表示中重建正常过程数据。判别器区分真实正常样本和重建样本。在推理时，**重建误差** 作为异常分数——故障导致高重建误差，因为生成器只学习了正常模式。

### KDE (Kernel Density Estimation) | 核密度估计

Instead of a fixed threshold, KDE estimates the probability density of normal training scores. The detection threshold is set at a high percentile (e.g., 95th or 99th), providing adaptive, data-driven thresholding.

不使用固定阈值，KDE 估计正常训练分数的概率密度。检测阈值设置在高分位数（如第 95 或 99 百分位），提供自适应的、数据驱动的阈值。

### Reconstruction-Based Score | 基于重建的分数

```
score(x) = ||x - G(x)||²
```

Where `G(x)` is the generator's reconstruction of input `x`. Normal samples have low scores (good reconstruction), fault samples have high scores (poor reconstruction).

---

## 📈 Results Visualization | 结果可视化

For each fault type, the project generates:

1. **Time Series Plot** — Process variables over time with fault introduction marker
2. **Score Distribution** — Histogram of anomaly scores (normal vs. fault)
3. **Continuous Scores** — Anomaly score over the 960 test samples
4. **Results CSV** — Detection results with timestamps

Comparative analysis includes:
- **Boxplot Comparison** — Score distributions across all fault types
- **Key Variable Analysis** — Identification of most affected variables for specific faults
- **Separation Comparison** — How well normal and fault scores are separated

---

## 📚 References | 参考文献

1. **Downs, J. J., & Vogel, E. F.** (1993). *A plant-wide industrial process control problem.* Computers & Chemical Engineering, 17(3), 245-255.
2. **Goodfellow, I., et al.** (2014). *Generative adversarial nets.* NeurIPS.
3. **Schlegl, T., et al.** (2017). *Unsupervised anomaly detection with generative adversarial networks to guide marker discovery.* IPMI.
4. **Bowman, A. W., & Azzalini, A.** (1997). *Applied smoothing techniques for data analysis: the kernel approach with S-Plus illustrations.* Oxford University Press.
5. **Chiang, L. H., Russell, E. L., & Braatz, R. D.** (2001). *Fault detection and diagnosis in industrial systems.* Springer.

---

## 📄 License | 许可证

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with 🏭 for industrial process monitoring research**

[Report Bug](https://github.com/Windyhhh/FGAN-Industrial-Fault-Detection/issues) · [Request Feature](https://github.com/Windyhhh/FGAN-Industrial-Fault-Detection/issues)

</div>
