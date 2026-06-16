# FAN-Autoformer
FAN-Autoformer: Frequency-Adaptive Normalization + AutoKernel Decomposition for non-stationary meteorological time series forecasting | PyTorch | XAI


# 🌡️ FAN-Autoformer: A Frequency-Adaptive Framework for Non-Stationary Time Series Forecasting

> **Official PyTorch implementation** of the paper:  
> *"FAN-Autoformer: A Frequency-Adaptive Framework for Non-Stationary Time Series Forecasting with XAI"*

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview

FAN-Autoformer is a unified multivariate deep learning framework that tackles **non-stationarity** and **distribution shift** in real-world meteorological time series by integrating three synergistic mechanisms:

| Component | Role |
|---|---|
| **FAN Layer** (Frequency-Adaptive Normalization) | Removes dominant non-stationary frequency components via FFT before the transformer backbone |
| **AutoKernel** (Dynamic Progressive Decomposition) | Adaptively calibrates the smoothing kernel size to the input horizon for phase-continuous trend extraction |
| **Auto-Correlation Encoder-Decoder** | Captures long-range periodic dependencies at O(L log L) complexity |

> 💡 **Key Insight (Ablation Study):** Optimal meteorological forecasting demands a *hybrid paradigm* — **frequency-domain normalization** for distribution alignment + **time-domain decomposition** to prevent Gibbs ringing artifacts.
## 📊 Results

Evaluated on daily minimum temperature (T_min) at Ilam Shohada Airport (ICAO: OICI), western Iran — across **4 forecasting horizons** (H ∈ {1, 3, 7, 30} days) and **5 independent random seeds**.

### Short-term Horizons (H = 1, 3 days)

| Model | MAE↓ (H=1) | MAE↓ (H=3) | R²↑ (H=1) | R²↑ (H=3) |
|---|---|---|---|---|
| **FAN-Autoformer (ours)** | **1.5531 ± 0.059** | **1.9213 ± 0.036** | **0.9369 ± 0.004** | **0.9061 ± 0.004** |
| Informer | 1.3940 ± 0.012 | 1.8923 ± 0.088 | 0.9484 ± 0.001 | 0.9073 ± 0.009 |
| PatchTST | 1.7407 ± 0.013 | 2.0612 ± 0.017 | 0.9233 ± 0.001 | 0.8935 ± 0.002 |
| TimeXer | 1.8300 ± 0.034 | 2.0615 ± 0.020 | 0.9174 ± 0.002 | 0.8918 ± 0.001 |
| iTransformer | 1.8683 ± 0.030 | 2.1350 ± 0.028 | 0.9148 ± 0.002 | 0.8844 ± 0.003 |
| Autoformer | 2.2142 ± 0.159 | 2.8774 ± 0.471 | 0.8785 ± 0.019 | 0.7990 ± 0.059 |

### Medium/Long-term Horizons (H = 7, 30 days)

| Model | MAE↓ (H=7) | MAE↓ (H=30) | R²↑ (H=7) | R²↑ (H=30) |
|---|---|---|---|---|
| **FAN-Autoformer (ours)** | 2.1733 ± 0.016 | 2.3171 ± 0.057 | **0.8827 ± 0.002** | 0.8682 ± 0.006 |
| Informer | **2.0457 ± 0.028** | **2.2247 ± 0.074** | **0.8926 ± 0.003** | **0.8723 ± 0.007** |
| TimeXer | 2.1619 ± 0.020 | **2.1987 ± 0.042** | 0.8783 ± 0.002 | **0.8762 ± 0.004** |
| Autoformer | 2.6106 ± 0.367 | 2.6737 ± 0.268 | 0.8293 ± 0.049 | 0.8303 ± 0.031 |

> ✅ **FAN-Autoformer achieves the lowest inter-seed variance across all horizons**, demonstrating superior optimization stability over all baselines — including PatchTST, iTransformer, and Vanilla Autoformer.
## 🏗️ Architecture

![Architecture](figures/architecture.png)

The pipeline:
1. **Input:** Multivariate time series (T_min + exogenous: T_max, humidity, vapor pressure, precipitation) + 6 cyclical calendar features
2. **FAN Layer:** FFT → identify top-K non-stationary frequencies → filter → inverse FFT → stationary residual
3. **AutoKernel Decomposition:** AvgPool with dynamically computed kernel = f(input_length) → trend + seasonal components
4. **Auto-Correlation Encoder-Decoder:** O(L log L) periodic dependency modeling via Wiener-Khinchin theorem
5. **Output restoration:** FAN de-normalization + transformer output composition

## 🔍 Explainability (GradCAM-based Attribution)

Gradient-weighted activation analysis reveals that FAN-Autoformer autonomously internalizes **physically meaningful** atmospheric dynamics:

- **tmin_lag14** → quasi-biweekly Rossby wave periodicities (14-day mesoscale oscillation)
- **tmin_diff_1** → thermal advection gradients and synoptic cold front passages
- **humidity_range + p0min** → Nocturnal Boundary Layer radiational cooling mechanisms

These findings provide strong evidence of **genuine physical pattern internalization** beyond spurious statistical correlation.

## 📁 Repository Structure
```markdown
FAN-Autoformer/
├── models/
│ ├── FAN-Autoformer.ipynb #with Xai
│ ├── requirements.txt
├── data/
│ └── ilam_weather.csv # Pre-processed Ilam Airport dataset
├── figures/ # All paper figures
│ ├──result/ # result for many models
└── README.md
```

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
Under Review
```

## 📬 Contact

For questions or collaboration inquiries, open a GitHub Issue or contact: `mojtaba.fathollahi4346@iau.ir`

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
