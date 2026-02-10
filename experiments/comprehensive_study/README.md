# Comprehensive GSSM Insurance Experimental Framework

## 🎯 Overview

This directory contains a **complete experimental validation** of the Insurance-GSSM methodology based on the theoretical framework established in the paper. All experiments are designed to rigorously link theoretical propositions to empirical results through systematic hypothesis testing.

## 📋 Experimental Structure

### Research Questions (RQs) Framework

Our experimental design is organized around **5 core research questions** that directly test the theoretical contributions:

| RQ | Question | Theory Link | Metrics |
|---|---|---|---|
| **RQ1** | Does domain adaptation improve forecasting? | Definition 3.1 (Continuous Dynamics) | R², MSE, RMSE, MAPE |
| **RQ2** | What is each component's contribution? | Theorem 3.2, Proposition 3.1 | Ablation R² differences |
| **RQ3** | Do components exhibit synergy? | Component interactions | Synergy scores |
| **RQ4** | Does cycle detection work empirically? | Theorem 3.2 (Spectral Sync) | FFT peak accuracy |
| **RQ5** | Does accuracy translate to business value? | Definition 3.1 (Solvency) | Combined Ratio, Profit |

## 🗂️ Directory Structure

```
comprehensive_study/
├── README.md                          # This file
├── EXPERIMENTAL_PROTOCOL.md           # Detailed protocols
├── scripts/                           # All experiment scripts
│   ├── rq1_baseline_comparison.py    # RQ1: 15 baseline methods
│   ├── rq2_ablation_study.py         # RQ2: 7 configurations
│   ├── rq3_synergy_analysis.py       # RQ3: Pairwise interactions
│   ├── rq4_cycle_validation.py       # RQ4: FFT spectral analysis
│   ├── rq5_business_impact.py        # RQ5: Financial translation
│   ├── generate_all_visualizations.py # Master visualization script
│   └── run_all_experiments.py        # Orchestrate all RQs
├── results/                           # Experimental outputs
│   ├── rq1_baseline_results.csv
│   ├── rq2_ablation_results.csv
│   ├── rq3_synergy_matrix.csv
│   ├── rq4_spectral_analysis.json
│   ├── rq5_financial_metrics.csv
│   └── statistical_validation.csv
├── visualizations/                    # All figures
│   ├── rq1_baseline_comparison.pdf   # 15-method bar chart
│   ├── rq1_horizon_heatmap.pdf       # Performance across horizons
│   ├── rq2_ablation_radar.pdf        # Component contributions
│   ├── rq2_waterfall_chart.pdf       # Cumulative gains
│   ├── rq3_synergy_heatmap.pdf       # Interaction matrix
│   ├── rq4_fft_analysis.pdf          # Time + frequency domain
│   ├── rq4_cycle_detection.pdf       # Detected vs. true cycles
│   ├── rq5_combined_ratio.pdf        # CR comparison
│   ├── rq5_profit_analysis.pdf       # Financial impact
│   └── supplementary/                 # Additional figures
└── analysis/                          # Analysis reports
    ├── rq1_analysis.md               # Detailed RQ1 findings
    ├── rq2_analysis.md               # Ablation insights
    ├── rq3_analysis.md               # Synergy interpretation
    ├── rq4_analysis.md               # Spectral validation
    ├── rq5_analysis.md               # Business implications
    └── COMPREHENSIVE_REPORT.md       # Master report
```

## 🔬 Experimental Methodology

### 1. Dataset Specification

**Synthetic Insurance Dataset** (following Wüthrich & Merz, 2023):

- **Volume**: 10,000 policies over 120 months (10 years)
- **Process**: Compound Poisson (freq ~ Poisson(λ), severity ~ LogNormal(μ, σ))
- **Structure**:
  - AR(2) autocorrelation: `y_t = 0.6*y_{t-1} + 0.3*y_{t-2} + ε_t`
  - 72-month market cycle: `exp(α*sin(2π*t/72))`
  - 12-month seasonality: `1 + 0.15*sin(2π*t/12)`
  - Heteroscedastic noise: `σ_t = σ_0 * (1 + 0.3*|sin(2π*t/24)|)`
- **Split**: 70% train (84m) / 15% val (18m) / 15% test (18m)

### 2. Baseline Methods (15 Total)

#### Classical Time Series (2)
1. **ARIMA** (Box et al., 2015) - Auto-regressive integrated moving average
2. **Prophet** (Taylor & Letham, 2018) - Additive model with trend/seasonality

#### Deep Recurrent (2)
3. **LSTM** (Hochreiter, 1997) - Long short-term memory
4. **GRU** (Cho et al., 2014) - Gated recurrent unit

#### Attention-Based (4)
5. **Transformer** (Vaswani et al., 2017) - Self-attention
6. **TFT** (Lim et al., 2021) - Temporal Fusion Transformer
7. **N-BEATS** (Oreshkin et al., 2019) - Neural basis expansion
8. **Informer** (Zhou et al., 2021) - Efficient attention

#### State-Space Models (2)
9. **Vanilla SSM (S4)** (Gu et al., 2021) - Structured state-space
10. **Mamba** (Gu & Dao, 2023) - Selective state-space

#### Bayesian / Probabilistic (2)
11. **MCMC-ARIMA** (Martin, 2021) - Bayesian ARIMA via MCMC
12. **Gaussian Process** (Rasmussen, 2006) - GP regression

#### Reinforcement Learning (2)
13. **PPO** (Schulman et al., 2017) - Proximal Policy Optimization
14. **DQN** (Mnih et al., 2015) - Deep Q-Network

#### Our Method
15. **Insurance-GSSM** (This work) - Generative SSM with Flow-Selectivity

### 3. Ablation Configurations (7 Total)

| Config | SSM Core | Flow-Selectivity (φ_FS) | FFT Cycles | Autocorr (r_AC) | Seasonal (τ_SE) | GFlowNet Policy |
|--------|----------|-------------------------|------------|-----------------|-----------------|-----------------|
| **Full Model** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **w/o Flow** | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| **w/o FFT** | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| **w/o Autocorr** | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| **w/o Seasonal** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| **w/o GFlowNet** | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |
| **Minimal SSM** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |

### 4. Evaluation Metrics

#### Forecasting Accuracy
- **R² Score**: Coefficient of determination
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **MAPE**: Mean absolute percentage error

#### Task-Specific
- **Poisson NLL**: Negative log-likelihood for frequency
- **Cross-Entropy**: For risk classification
- **Precision/Recall/F1**: Classification metrics

#### Business Impact
- **Combined Ratio (CR)**: (Losses + Expenses) / Premiums × 100%
- **Loss Ratio (LR)**: Incurred Losses / Earned Premiums × 100%
- **Pricing MAPE**: Pricing accuracy
- **Reserve Adequacy**: % of required reserves met
- **Annual Profit**: For $1B premium volume

#### Statistical Validation
- **Paired t-tests**: With Bonferroni correction (α=0.05/14)
- **Bootstrap CI**: 95% confidence intervals (1000 samples)
- **Wilcoxon signed-rank**: Non-parametric validation
- **Cohen's d**: Effect size measurement

## 🚀 Quick Start

### Run All Experiments

```bash
# 1. Run all RQ experiments sequentially
python scripts/run_all_experiments.py --mode full

# 2. Run specific RQ
python scripts/rq1_baseline_comparison.py
python scripts/rq2_ablation_study.py
python scripts/rq3_synergy_analysis.py
python scripts/rq4_cycle_validation.py
python scripts/rq5_business_impact.py

# 3. Generate all visualizations
python scripts/generate_all_visualizations.py

# 4. Generate comprehensive report
python scripts/generate_comprehensive_report.py
```

### Run Subset (Quick Validation)

```bash
# Run with reduced baselines and epochs
python scripts/run_all_experiments.py --mode quick --epochs 10 --baselines 5
```

### Parallel Execution

```bash
# Run RQs in parallel (requires 5 GPUs)
python scripts/run_all_experiments.py --mode parallel --devices 0,1,2,3,4
```

## 📊 Expected Results

Based on the theoretical framework, we hypothesize:

### RQ1: Baseline Comparison
- **Hypothesis**: GSSM achieves 24-67% improvement over best baseline
- **Expected R² (12m)**: 
  - ARIMA: 0.038, Prophet: 0.045, LSTM: 0.050, GRU: 0.053
  - Transformer: 0.058, TFT: 0.061, N-BEATS: 0.060
  - Vanilla SSM: 0.065, Mamba: 0.067
  - PPO: 0.042, DQN: 0.038
  - **GSSM: 0.072** (24.1% gain vs. best)

### RQ2: Ablation Study
- **Hypothesis**: Each component contributes positively
- **Expected gains**:
  - Full vs. Minimal: +44% (12m R²)
  - Flow-Selectivity: +8-12% (enables risk routing)
  - FFT Cycles: +5-7% (long-horizon improvement)
  - Autocorr: +10-15% (short-horizon improvement)
  - GFlowNet Policy: +15-20% (trajectory validity)

### RQ3: Synergy Analysis
- **Hypothesis**: Components exhibit super-additive effects
- **Expected synergies**:
  - Autocorr × FFT: +22% (complementary timescales)
  - FFT × Seasonal: +14% (frequency interaction)
  - Flow × GFlowNet: +18% (policy optimization)

### RQ4: Cycle Detection
- **Hypothesis**: FFT detects 72-month cycle with <5% error
- **Expected results**:
  - Detected period: 70-74 months
  - Bootstrap 95% CI: [68, 76]
  - SNR: >15 dB

### RQ5: Business Impact
- **Hypothesis**: Technical gains translate to $13M value
- **Expected CR**:
  - Manual Actuary: 102.5% (loss)
  - ARIMA: 101.8% (loss)
  - Vanilla SSM: 99.8% (+$2M)
  - **GSSM: 98.5%** (+$15M)

## 📈 Visualization Catalog

### Primary Figures (9)

1. **Baseline Comparison Bar Chart**: 15 methods, R² scores, color-coded by paradigm
2. **Horizon Performance Heatmap**: All methods × 4 horizons
3. **Ablation Radar Chart**: 7 configs × 5 metrics
4. **Waterfall Chart**: Cumulative component contributions
5. **Synergy Heatmap**: Pairwise interaction matrix (6×6)
6. **FFT Time-Frequency Plot**: Dual-panel spectral analysis
7. **Cycle Detection Validation**: Detected vs. true periods with CI
8. **Combined Ratio Comparison**: Financial profitability across methods
9. **Profit Analysis**: Annual value generation breakdown

### Supplementary Figures (12)

10. Learning curves (loss over epochs)
11. Error distributions (Q-Q plots)
12. Autocorrelation functions (ACF/PACF)
13. Power spectrum comparison
14. Flow gate activations over time
15. Policy entropy dynamics
16. Feature importance weights
17. Reserve adequacy analysis
18. Loss ratio forecasting
19. Statistical significance forest plot
20. Confusion matrices (risk classification)
21. Architecture diagram with information flow

## 📝 Analysis Reports

Each RQ generates a detailed Markdown report containing:

- **Hypothesis**: Theoretical prediction
- **Experimental Design**: Protocol and controls
- **Results**: Quantitative findings with statistics
- **Analysis**: Interpretation linking theory to data
- **Figures**: Cross-referenced visualizations
- **Discussion**: Implications and limitations
- **Answer**: Direct response to RQ

## 🔗 Theory-to-Experiment Linkage

### Mathematical Foundations → Empirical Validation

| Theoretical Component | Mathematical Form | Experimental Validation | Expected Result |
|----------------------|-------------------|------------------------|-----------------|
| **SSM Stability** | Theorem 3.1: Re(λ(A)) < 0 | Eigenvalue analysis of learned A | All eigenvalues in LHP |
| **Flow-Selectivity** | Proposition 3.1: E[y\|h] = Ch + Σ a·P(a\|h) | Policy entropy H(P_F) dynamics | High entropy in transitions |
| **Spectral Sync** | Theorem 3.2: D_KL(S_target \|\| S_PF) → 0 | FFT peak detection accuracy | 2.2% cycle error |
| **Autocorr Injection** | Eq. 5: r_AC = Σ w_k · Cov(τ_t, τ_{t-k}) | ACF/PACF of generated trajectories | Match ground truth ±0.05 |
| **Backward Policy** | Lemma 3.5: P_B ensures trajectory validity | Constraint violation rate | <1% invalid trajectories |

## ⚙️ Hyperparameters

### Model Architecture
- **SSM State Dimension (N)**: 64
- **Model Dimension (d_model)**: 256
- **Number of Layers**: 4
- **Flow Policy Hidden Dim**: 128
- **GFlowNet Action Space**: 50 discrete bins

### Training
- **Optimizer**: AdamW
- **Learning Rate**: 5e-5
- **Weight Decay**: 1e-4
- **Batch Size**: 32
- **Epochs**: 50 (full), 10 (quick)
- **Gradient Clip**: 0.5
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10)

### Loss Weights
- λ_claims = 0.5
- λ_freq = 0.2
- λ_risk = 0.1
- λ_price = 0.2
- β_spectral = 0.1 (FFT KL divergence weight)
- λ_autocorr = 0.05 (r_AC reward weight)

### Baseline-Specific
- **ARIMA**: auto.arima (p,d,q) selection
- **Prophet**: yearly, weekly seasonality
- **LSTM/GRU**: 2 layers, hidden=256
- **Transformer**: 4 heads, 4 layers
- **TFT**: Variable selection + attention
- **PPO**: Actor-Critic, clip_ratio=0.2
- **DQN**: ε-greedy, replay_buffer=10k

## 🎓 Statistical Rigor

### Multiple Testing Correction
- **Bonferroni**: α = 0.05/14 = 0.00357 for 14 pairwise comparisons
- **Holm-Bonferroni**: Sequential correction for ablations

### Effect Size Thresholds
- Cohen's d: 0.2 (small), 0.5 (medium), 0.8 (large), 1.2 (very large)

### Bootstrap Procedure
- Resamples: 1000
- Method: Percentile CI (2.5%, 97.5%)
- Stratified by horizon

### Significance Markers
- * : p < 0.05
- ** : p < 0.01
- *** : p < 0.001 (highly significant)

## 📦 Deliverables

Upon completion, this experiment generates:

1. **Results CSVs** (5 files): Machine-readable metrics
2. **Visualizations** (21 figures): Publication-quality PDFs + PNGs
3. **Analysis Reports** (6 Markdown files): Detailed interpretations
4. **Statistical Tables** (LaTeX): For paper integration
5. **Comprehensive Report** (1 master Markdown): Executive summary
6. **Code Archive** (ZIP): Reproducible scripts

## 🔄 Reproducibility

All experiments use:
- **Fixed Random Seeds**: 42, 123, 456 (3 runs per config)
- **Version Pinning**: PyTorch 2.0+, NumPy 1.24+
- **Environment**: Conda environment.yml provided
- **Hardware**: NVIDIA A100 (40GB) recommended, V100 minimum
- **Time Estimate**: ~48 hours full run, ~6 hours quick mode

## 📚 References

Key papers cited in experiments:

- **GSSM Theory**: This work (ICML 2026 submission)
- **S4/Mamba**: Gu et al. (2021, 2023)
- **GFlowNets**: Bengio et al. (2021)
- **Insurance Cycles**: Venezian (1985), Harrington et al. (2004)
- **Actuarial Modeling**: Wüthrich & Merz (2023)

## 🆘 Troubleshooting

### Common Issues

1. **OOM Error**: Reduce batch_size from 32 to 16
2. **NaN Loss**: Check gradient clipping, reduce lr to 1e-5
3. **Slow FFT**: Use scipy.fft instead of numpy.fft
4. **Missing Baselines**: Install via `pip install -r requirements_baselines.txt`

### Contact

For issues or questions about experiments:
- See `EXPERIMENTAL_PROTOCOL.md` for detailed protocols
- Check `analysis/*.md` for interpretation guidance
- Review `scripts/*.py` for implementation details

---

**Status**: 🚀 Ready for Full-Scale Execution  
**Last Updated**: February 8, 2026  
**Estimated Completion**: 48 hours (full mode) | 6 hours (quick mode)  
**Expected Output**: 21 figures, 5 result files, 6 analysis reports, 1 comprehensive report
