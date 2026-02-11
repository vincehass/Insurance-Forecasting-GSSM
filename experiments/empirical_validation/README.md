# Empirical Validation of Insurance-GSSM Components

**Author**: Nadhir Hassen  
**Email**: nadhir.hassen@mila.quebec  
**Date**: February 7, 2026

## Overview

This experiment folder contains comprehensive empirical validation of all Insurance-GSSM methodology components with research questions, statistical tests, comparative analysis, and publication-quality visualizations.

## Research Questions

### RQ1: Autocorrelation Temporal Dependencies
**Question**: Does the Autocorrelation Module (r_AC) effectively capture temporal claims dependencies compared to baseline temporal models?

**Hypothesis**: The autocorrelation operator captures long-range dependencies in claims patterns more effectively than RNN/LSTM baselines.

**Metrics**: ACF scores, PACF analysis, MSE, MAE, temporal consistency

### RQ2: Spectral Cycle Detection
**Question**: Can FFT-based Cycle Detection identify insurance market cycles (hard/soft phases) with statistical significance?

**Hypothesis**: The spectral cycle alignment accurately detects 72-month insurance cycles with KL divergence < 0.05.

**Metrics**: KL divergence, cycle detection accuracy, frequency domain analysis, phase alignment

### RQ3: Flow-Selectivity Information Routing
**Question**: Does the Flow-Selectivity Layer (φ_FS) improve pricing decisions through adaptive information gating?

**Hypothesis**: GFlowNet-based routing reduces pricing error by >15% vs. standard attention mechanisms.

**Metrics**: Pricing accuracy, gating entropy, feature importance, ablation performance

### RQ4: Seasonal Encoding Effectiveness
**Question**: Do sinusoidal seasonal embeddings (τ_SE) capture periodic patterns better than learned embeddings?

**Hypothesis**: Explicit temporal encoding improves forecasting accuracy for seasonal claims by >10%.

**Metrics**: Seasonal R², periodic pattern recovery, forecast accuracy on quarterly data

### RQ5: Multi-Horizon Forecasting Consistency
**Question**: Does the GSSM maintain forecast quality across multiple horizons (3, 6, 12, 24 months)?

**Hypothesis**: GSSM shows <5% accuracy degradation across horizons vs. >15% for baselines.

**Metrics**: MSE by horizon, MAPE consistency, forecast calibration

### RQ6: Component Synergy
**Question**: Do GSSM components work synergistically, or are improvements additive?

**Hypothesis**: Full GSSM outperforms sum of individual component improvements (synergy effect > 20%).

**Metrics**: Ablation study results, component interaction analysis

## Experimental Design

### Dataset Specification
- **Size**: 1,000 policies × 100 months (100,000 observations)
- **Features**: 50 (12 core + 24 lag + 8 temporal + 6 derived)
- **Targets**: 4 tasks (claims amount, frequency, risk, premium)
- **Horizons**: 4 (3, 6, 12, 24 months)
- **Splits**: 70% train / 15% validation / 15% test
- **Seeds**: 10 random seeds for statistical validation

### Baseline Methods
1. **ARIMA** - Classical time series
2. **Prophet** - Facebook's forecasting tool
3. **LSTM** - Recurrent neural network
4. **GRU** - Gated recurrent unit
5. **Transformer** - Self-attention architecture
6. **TFT** - Temporal Fusion Transformers
7. **N-BEATS** - Neural basis expansion
8. **Vanilla SSM (S4)** - Standard state-space model
9. **Mamba** - Selective SSM
10. **Insurance-GSSM (Full)** - Our method

### Ablation Configurations
1. **Full Model** - All components enabled
2. **No Autocorrelation** - Remove r_AC
3. **No Cycle Detection** - Remove FFT-based cycle detector
4. **No Flow-Selectivity** - Remove φ_FS gating
5. **No Seasonal Encoding** - Remove τ_SE
6. **Minimal SSM** - Only core SSM without insurance adaptations
7. **No Multi-Task** - Single task (claims only)

## Experimental Protocol

### 1. Data Generation
```bash
python scripts/01_generate_data.py \
    --n_policies 1000 \
    --n_months 100 \
    --output results/insurance_data.csv
```

### 2. Baseline Comparison (RQ1-RQ5)
```bash
python scripts/02_run_baselines.py \
    --data results/insurance_data.csv \
    --models all \
    --seeds 10 \
    --output results/baseline_results.json
```

### 3. Ablation Study (RQ6)
```bash
python scripts/03_run_ablation.py \
    --data results/insurance_data.csv \
    --config all \
    --seeds 10 \
    --output results/ablation_results.json
```

### 4. Component Analysis
```bash
python scripts/04_component_analysis.py \
    --results results/ \
    --output analysis/
```

### 5. Visualization Generation
```bash
python scripts/05_generate_figures.py \
    --results results/ \
    --figures figures/ \
    --format pdf
```

### 6. Statistical Validation
```bash
python scripts/06_statistical_tests.py \
    --results results/ \
    --tests all \
    --output tables/statistical_validation.tex
```

### 7. Generate Report
```bash
python scripts/07_generate_report.py \
    --results results/ \
    --figures figures/ \
    --tables tables/ \
    --output empirical_validation_report.tex
```

## Expected Outputs

### Tables
1. `baseline_comparison.tex` - Performance metrics across all baselines
2. `ablation_study.tex` - Component contribution analysis
3. `statistical_tests.tex` - T-tests, p-values, Cohen's d, confidence intervals
4. `multi_horizon_performance.tex` - Accuracy by forecast horizon
5. `component_synergy.tex` - Interaction effects analysis

### Figures
1. `autocorrelation_analysis.pdf` - ACF/PACF plots, temporal patterns
2. `cycle_detection_fft.pdf` - Frequency domain analysis, detected cycles
3. `flow_selectivity_gates.pdf` - Gating patterns, feature importance
4. `seasonal_patterns.pdf` - Seasonal decomposition, periodic effects
5. `multi_horizon_forecasting.pdf` - Forecast quality across horizons
6. `baseline_comparison.pdf` - Bar charts, radar plots
7. `ablation_waterfall.pdf` - Component contribution breakdown
8. `learning_curves.pdf` - Training dynamics
9. `error_analysis.pdf` - Residual plots, calibration curves
10. `component_synergy.pdf` - Interaction heatmap

### Analysis Reports
1. `rq1_autocorrelation.md` - Detailed RQ1 analysis
2. `rq2_cycle_detection.md` - Detailed RQ2 analysis
3. `rq3_flow_selectivity.md` - Detailed RQ3 analysis
4. `rq4_seasonal_encoding.md` - Detailed RQ4 analysis
5. `rq5_multi_horizon.md` - Detailed RQ5 analysis
6. `rq6_synergy.md` - Detailed RQ6 analysis

## Results Summary

Results will be populated after experiments are run. Expected improvements:

| Component | Baseline | Insurance-GSSM | Improvement |
|-----------|----------|----------------|-------------|
| Overall MSE | 0.145 | 0.092 | **36.6%** |
| Claims MAE | 127.3 | 98.4 | **22.7%** |
| Risk F1 | 0.823 | 0.911 | **10.7%** |
| 24m Horizon | 0.198 | 0.134 | **32.3%** |

## Theoretical Links

Each experiment explicitly links to theoretical propositions:

- **RQ1** → Definition 3.4 (Autocorrelation Operator)
- **RQ2** → Theorem 3.2 (Spectral Cycle Synchronization)
- **RQ3** → Proposition 3.1 (SSM as Flow-Selective Gate)
- **RQ4** → Definition 3.3 (Seasonal Temporal Encoding)
- **RQ5** → Definition 3.1 (Actuarial Dynamical System)
- **RQ6** → Theorem 3.3 (Component Synergy Bound)

## Citation

If you use this experimental framework, please cite:

```bibtex
@inproceedings{hassen2026insurance,
  title={Insurance-GSSM: Domain-Adapted State-Space Models for Multi-Horizon Insurance Forecasting},
  author={Hassen, Nadhir},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  organization={Mila - Quebec AI Institute}
}
```

## Contact

For questions about the experiments:
- **Email**: nadhir.hassen@mila.quebec
- **Institution**: Mila - Quebec AI Institute

## License

This research code is released under MIT License. See main repository for details.
