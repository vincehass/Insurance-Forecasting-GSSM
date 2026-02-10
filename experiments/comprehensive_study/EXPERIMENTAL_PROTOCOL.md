# Experimental Protocol: Complete GSSM Validation Framework

## Document Purpose

This protocol provides detailed experimental procedures for validating the Insurance-GSSM methodology based on the theoretical framework established in "Insurance-GSSM: Generative State-Space Models with Flow-Selective Policies for Multi-Horizon Actuarial Forecasting."

## Table of Contents

1. [Experimental Design Principles](#experimental-design-principles)
2. [Research Questions Framework](#research-questions-framework)
3. [Data Generation Protocol](#data-generation-protocol)
4. [Baseline Implementation Details](#baseline-implementation-details)
5. [Ablation Study Protocol](#ablation-study-protocol)
6. [Statistical Validation Procedures](#statistical-validation-procedures)
7. [Visualization Guidelines](#visualization-guidelines)
8. [Analysis and Reporting](#analysis-and-reporting)

---

## 1. Experimental Design Principles

### 1.1 Core Principles

**P1. Theory-Experiment Linkage**: Every experiment must explicitly link to a theoretical proposition from Section 3 of the paper.

**P2. Reproducibility**: All experiments use fixed random seeds (42, 123, 456) with complete parameter documentation.

**P3. Statistical Rigor**: Multiple testing correction (Bonferroni), effect size reporting (Cohen's d), confidence intervals (bootstrap).

**P4. Fair Comparison**: All baselines trained on identical data with comparable parameter budgets (~400K parameters).

**P5. Transparent Reporting**: Report both successes and failures, including negative results from ablations.

### 1.2 Hypotheses Testing Framework

Each research question follows this structure:

```
RQ[N]: [Question]
├── Theoretical Link: [Theorem/Proposition reference]
├── Hypothesis: [Specific testable prediction]
├── Experimental Design:
│   ├── Independent Variables: [What we manipulate]
│   ├── Dependent Variables: [What we measure]
│   ├── Controls: [What we hold constant]
│   └── Sample Size: [n policies, n horizons, n seeds]
├── Expected Results: [Quantitative predictions]
├── Analysis Plan: [Statistical tests, visualizations]
└── Decision Criteria: [Accept/reject thresholds]
```

---

## 2. Research Questions Framework

### RQ1: Does Domain Adaptation Improve Forecasting Accuracy?

**Theoretical Link**: Definition 3.1 (Actuarial Dynamical System) - Continuous-time LTI system vs. discrete recurrent architectures.

**Hypothesis**: Insurance-GSSM with continuous-time SSM backbone captures long-range dependencies (12+ months) better than discrete RNNs, attention mechanisms, and classical time series methods.

**Metrics**:
- **Primary**: R² coefficient at 12-month horizon
- **Secondary**: MSE, RMSE, MAE, MAPE at 3m, 6m, 12m, 24m

**Expected Results**:
- GSSM achieves R²(12m) = 0.072 ± 0.002
- Improvement over best baseline (Vanilla S4): +24.1% relative gain
- All comparisons p < 0.001 (Bonferroni corrected)

**Statistical Tests**:
- Paired t-tests (GSSM vs. each baseline)
- Bonferroni correction: α = 0.05/14 = 0.00357
- Cohen's d effect sizes (target: d > 0.8 "large")
- Bootstrap 95% CI (1000 resamples)

**Decision Criteria**:
- **Accept H1** if: R²(GSSM) > R²(best_baseline) AND p < 0.00357 AND d > 0.5
- **Reject H1** if: R²(GSSM) ≤ R²(best_baseline) OR p ≥ 0.05

---

### RQ2: What is Each Component's Contribution?

**Theoretical Link**: 
- Theorem 3.2 (Spectral Cycle Synchronization) → FFT component
- Proposition 3.1 (Flow-Selective Gate) → φ_FS component
- Definition 3.4 (Autocorrelation Operator) → r_AC component

**Hypothesis**: Each domain-specific component (Flow-Selectivity, FFT Cycles, Autocorr, GFlowNet, Seasonal) contributes positively, with synergistic interactions.

**Ablation Configurations** (7 total):
1. Full Model (baseline)
2. w/o Flow-Selectivity
3. w/o FFT Cycles
4. w/o Autocorr
5. w/o Seasonal
6. w/o GFlowNet Policy
7. Minimal SSM (all ablations)

**Expected Component Impacts** (12m R²):
- Flow-Selectivity: -0.008 (enables risk routing)
- FFT Cycles: -0.005 (long-horizon cycles)
- Autocorr: -0.007 (short-horizon persistence)
- Seasonal: -0.003 (annual patterns)
- GFlowNet: -0.012 (trajectory validity)
- **Total Domain Gain**: Full vs. Minimal = +44%

**Statistical Tests**:
- Paired t-tests: Full vs. each ablation
- ANOVA: Overall effect of ablations
- Post-hoc: Tukey HSD for pairwise comparisons

---

### RQ3: Do Components Exhibit Synergy?

**Theoretical Link**: Component interactions through shared SSM hidden state h_t.

**Hypothesis**: Integrated architecture exhibits super-additive effects beyond simple component stacking.

**Synergy Calculation**:
```
Synergy(A, B) = R²_Full - (R²_{w/o A} + R²_{w/o B} - R²_Minimal)
```

**Expected Synergies**:
- Autocorr × FFT: +0.008 (+22.1%) - complementary timescales
- FFT × Seasonal: +0.005 (+13.8%) - frequency interaction
- Flow × GFlowNet: +0.012 (+18%) - policy optimization

**Analysis**:
- Interaction heatmap (6×6 pairwise)
- Mechanistic interpretation per interaction
- Total synergy: sum of positive interactions

---

### RQ4: Does Cycle Detection Work Empirically?

**Theoretical Link**: Theorem 3.2 implies S_PF → S_target via frequency-domain KL minimization.

**Hypothesis**: FFT-based cycle detection identifies embedded 72-month market cycle with <5% error.

**Ground Truth**: Synthetic data with deterministic 72-month sinusoidal cycle.

**Detection Protocol**:
1. Generate 10 independent trajectories from GSSM policy
2. Compute Power Spectral Density (PSD) via FFT
3. Identify dominant frequency: f* = argmax_k P_k
4. Convert to period: T_detected = N / f*
5. Bootstrap 95% CI (1000 samples)

**Expected Results**:
- Detected period: 70-74 months (2-3% error)
- Bootstrap 95% CI: [68, 76]
- SNR (peak / background): >15 dB
- Validation: Wilcoxon test against uniform spectrum (p < 0.001)

---

### RQ5: Does Accuracy Translate to Business Value?

**Theoretical Link**: Definition 3.1 (Solvency Metric) - Combined Ratio as actuarial KPI.

**Hypothesis**: Technical forecasting improvements translate to measurable financial gains for insurers.

**Business Metrics**:
- **Combined Ratio (CR)**: (Losses + Expenses) / Premiums × 100%
- **Loss Ratio (LR)**: Incurred Losses / Earned Premiums × 100%
- **Pricing MAPE**: Accuracy of premium predictions
- **Reserve Adequacy**: % of required reserves met
- **Annual Profit**: For $1B premium volume

**Expected Results**:
- GSSM CR: 98.5% (1.5% profit margin)
- Vanilla SSM CR: 99.8% (0.2% profit)
- Traditional methods: CR > 100% (underwriting loss)
- **Value Generation**: $13M annual gain vs. best baseline

**Sensitivity Analysis**:
- Vary premium volume: $500M, $1B, $2B
- Vary expense ratio: 30%, 35%, 40%
- Monte Carlo: 10,000 simulations

---

## 3. Data Generation Protocol

### 3.1 Collective Risk Model

Following Wüthrich & Merz (2023), we generate synthetic insurance data via:

```python
# Compound Poisson Process
for t in range(T):
    N_t = Poisson(λ_t)  # Number of claims
    for i in range(N_t):
        X_ti = LogNormal(μ, σ)  # Claim severity
    Y_t = sum(X_ti)  # Total claims
```

### 3.2 Embedded Structure

**AR(2) Autocorrelation**:
```
Y_t = μ_t + 0.6*(Y_{t-1} - μ_{t-1}) + 0.3*(Y_{t-2} - μ_{t-2}) + ε_t
ε_t ~ N(0, σ_t²)
```

**72-Month Market Cycle**:
```
μ_t = μ_base * exp(α * sin(2π * t / 72 + φ))
α = 0.15  # Cycle amplitude
φ ~ Uniform(0, 2π)  # Random phase
```

**12-Month Seasonality**:
```
s_t = 1 + 0.15 * sin(2π * t / 12)
Y_t = Y_t * s_t
```

**Heteroscedastic Noise**:
```
σ_t = σ_base * (1 + 0.3 * |sin(2π * t / 24)|)
```

### 3.3 Dataset Specification

- **Policies**: N = 10,000
- **Duration**: T = 120 months (10 years)
- **Features**: d_x = 20 (claims history, policy attributes, economic indicators, seasonal)
- **Train/Val/Test**: 70% / 15% / 15% (84m / 18m / 18m)
- **Seeds**: 42, 123, 456 (3 independent datasets)

### 3.4 Validation Checks

Before experiments, verify:
1. **Autocorrelation**: ACF(lag=1) ≈ 0.6 ± 0.05
2. **Cycle**: FFT peak at 72 ± 3 months
3. **Seasonality**: Spectral peak at 12 months
4. **Stationarity**: ADF test p > 0.05 (non-stationary)
5. **Heteroscedasticity**: Breusch-Pagan test p < 0.05

---

## 4. Baseline Implementation Details

### 4.1 Parameter Budget Equivalence

All models constrained to ~400K parameters for fair comparison:

| Method | Architecture | Parameters |
|--------|--------------|------------|
| LSTM | 2 layers × 256 hidden | 395K |
| GRU | 2 layers × 256 hidden | 394K |
| Transformer | 4 layers × 256 d_model, 4 heads | 402K |
| Vanilla SSM | N=64, d_model=256, 4 layers | 398K |
| Insurance-GSSM | N=64, d_model=256, 4 layers + domain components | 405K |

### 4.2 Training Protocol (All Methods)

**Optimizer**: AdamW(lr=5e-5, weight_decay=1e-4)
**Scheduler**: CosineAnnealingWarmRestarts(T_0=10)
**Batch Size**: 32
**Epochs**: 50 (full) / 10 (quick)
**Gradient Clipping**: 0.5
**Early Stopping**: Patience=10, monitor=val_loss

**Loss Functions**:
- Claims amount: MSE
- Claims frequency: Poisson NLL
- Risk classification: Cross-Entropy
- Premium pricing: MSE + L1 (combined)

### 4.3 Baseline-Specific Details

**ARIMA**:
- Order selection: auto.arima (AIC criterion)
- Max p, d, q: 5, 2, 5
- Seasonal: Yes (period=12)

**Prophet**:
- Yearly seasonality: True
- Changepoint prior scale: 0.05
- Seasonality mode: multiplicative

**PPO (Reinforcement Learning)**:
- Actor-Critic architecture
- Clip ratio: 0.2
- GAE λ: 0.95
- Entropy coefficient: 0.01
- **Reward**: R(τ) = -MSE(forecast, truth) + λ * r_AC(τ)

**DQN**:
- ε-greedy: ε_start=1.0, ε_end=0.01, decay=0.995
- Replay buffer: 10,000
- Target network update: every 100 steps
- **Action space**: 50 discretized forecast values

---

## 5. Ablation Study Protocol

### 5.1 Component Isolation

Each ablation removes exactly ONE component while keeping all others:

**Example: w/o Flow-Selectivity**
```python
model = InsuranceGSSM(
    use_flow=False,        # ← ABLATED
    use_cycle=True,
    use_autocorr=True,
    use_seasonal=True,
    use_gflownet=True
)
```

### 5.2 Minimal SSM Baseline

The "Minimal SSM" configuration removes ALL domain components, serving as the vanilla S4 baseline:

```python
model = InsuranceGSSM(
    use_flow=False,
    use_cycle=False,
    use_autocorr=False,
    use_seasonal=False,
    use_gflownet=False
)
# Equivalent to vanilla S4
```

### 5.3 Interaction Analysis

**Synergy Score**:
```
S(A, B) = R²_Full - (R²_{w/o A} + R²_{w/o B} - R²_Minimal)
```

**Interpretation**:
- S > 0: Positive synergy (super-additive)
- S = 0: Independent components (additive)
- S < 0: Interference (sub-additive)

---

## 6. Statistical Validation Procedures

### 6.1 Multiple Testing Correction

**Bonferroni Correction**:
- Total comparisons: 14 (GSSM vs. 14 baselines)
- Adjusted α: 0.05 / 14 = 0.00357
- Reject H0 if p < 0.00357

**Holm-Bonferroni** (for ablations, more powerful):
1. Order p-values: p_1 ≤ p_2 ≤ ... ≤ p_7
2. Compare p_i to α / (8 - i)
3. Reject H0 for all j ≤ i where p_i < α / (8 - i)

### 6.2 Effect Size Thresholds

**Cohen's d**:
```
d = (mean_GSSM - mean_baseline) / pooled_std
```

**Interpretation**:
- d < 0.2: Negligible
- 0.2 ≤ d < 0.5: Small
- 0.5 ≤ d < 0.8: Medium
- d ≥ 0.8: Large
- d ≥ 1.2: Very large

**Target**: All key comparisons should achieve d > 0.8 (large).

### 6.3 Bootstrap Confidence Intervals

**Procedure**:
1. Resample test set with replacement (n=1500)
2. Compute metric (e.g., R²) on resample
3. Repeat 1000 times
4. CI = [2.5th percentile, 97.5th percentile]

**Reporting**:
- Point estimate: mean
- Uncertainty: 95% CI
- Example: R² = 0.072 [0.068, 0.076]

### 6.4 Non-Parametric Tests

**Wilcoxon Signed-Rank**:
- Use when normality assumption violated
- Paired test: GSSM vs. baseline on same test samples
- H0: Median difference = 0
- Report: W statistic, p-value

---

## 7. Visualization Guidelines

### 7.1 Publication Standards

All figures must meet:
- **Resolution**: 300 DPI minimum
- **Format**: PDF (vector) + PNG (raster)
- **Font**: Serif (Times, Palatino), size 10-12pt
- **Colors**: Colorblind-safe palette (use ColorBrewer)
- **Captions**: 100-200 words with panel descriptions

### 7.2 Required Figures

**RQ1 (3 figures)**:
1. Baseline comparison bar chart (15 methods)
2. Multi-horizon heatmap (methods × horizons)
3. Relative gains line plot (horizons)

**RQ2 (4 figures)**:
4. Ablation radar chart (7 configs)
5. Waterfall chart (cumulative gains)
6. Component impact bar chart (per horizon)
7. Domain gain visualization (Full vs. Minimal)

**RQ3 (2 figures)**:
8. Synergy heatmap (6×6 interactions)
9. Network diagram (component dependencies)

**RQ4 (3 figures)**:
10. FFT time-frequency dual panel
11. Cycle detection validation (detected vs. true)
12. Spectral comparison (GSSM vs. baselines)

**RQ5 (3 figures)**:
13. Combined Ratio bar chart
14. Profit analysis (volume sensitivity)
15. Loss Ratio forecasting over time

**Supplementary (6+ figures)**:
16. Learning curves (all methods)
17. Error distributions (Q-Q plots)
18. ACF/PACF plots (generated trajectories)
19. Architecture diagram (GSSM components)
20. Confusion matrices (risk classification)
21. Statistical significance forest plot

---

## 8. Analysis and Reporting

### 8.1 Analysis Structure (per RQ)

Each RQ analysis follows:

```markdown
## RQ[N]: [Question]

### Hypothesis
[Specific testable prediction with theoretical link]

### Experimental Design
- **Independent Variables**: [...]
- **Dependent Variables**: [...]
- **Sample Size**: n=[...]
- **Statistical Power**: 1-β=[...]

### Results
[Present data: tables, figures with cross-references]

### Statistical Analysis
- **Test Used**: [...]
- **Results**: t([df]) = [...], p = [...], d = [...]
- **Interpretation**: [Accept/reject hypothesis]

### Mechanistic Interpretation
[Link results to theoretical propositions]

### Insurance Implications
[Translate to actuarial practice]

### Limitations
[Caveats, assumptions, boundary conditions]

### Answer to RQ[N]
[Direct yes/no answer with quantitative summary]
```

### 8.2 Reporting Standards

**Tables**:
- Mean ± std (across seeds)
- Significance markers: * p<0.05, ** p<0.01, *** p<0.001
- Effect sizes in separate column
- Sample sizes in caption

**Figures**:
- Error bars: 95% CI (bootstrap)
- Legend: clear method names, no abbreviations
- Axes: units explicitly labeled
- Reference lines: theoretical values, break-even points

**Text**:
- Report exact p-values (not "p<0.05")
- Report confidence intervals for all estimates
- Discuss practical significance, not just statistical
- Link every finding to theory (proposition/theorem number)

---

## 9. Quality Assurance Checklist

Before finalizing experiments, verify:

### Data Quality
- [ ] Train/val/test splits verified
- [ ] No data leakage between splits
- [ ] Embedded cycle/autocorrelation validated
- [ ] Class balance checked (for classification)
- [ ] Missing values handled appropriately

### Implementation
- [ ] All baselines use identical data
- [ ] Parameter budgets equivalent (~400K)
- [ ] Random seeds fixed and documented
- [ ] Gradient clipping applied consistently
- [ ] Early stopping criteria consistent

### Statistical Validity
- [ ] Multiple testing correction applied
- [ ] Effect sizes reported
- [ ] Confidence intervals computed
- [ ] Assumptions checked (normality, homoscedasticity)
- [ ] Sample size adequate (power analysis)

### Reproducibility
- [ ] Requirements.txt / environment.yml provided
- [ ] Hyperparameters documented
- [ ] Code commented and readable
- [ ] Seeds documented
- [ ] Hardware specifications noted

### Reporting
- [ ] All RQs answered directly
- [ ] Theory-experiment links explicit
- [ ] Limitations acknowledged
- [ ] Negative results reported
- [ ] Figures meet publication standards

---

## 10. Troubleshooting Guide

### Issue: NaN losses during training
**Diagnosis**: Numerical instability in SSM discretization or FFT
**Solution**:
1. Reduce learning rate: 5e-5 → 1e-5
2. Increase gradient clipping: 0.5 → 0.3
3. Add epsilon to denominators: 1e-8 → 1e-6
4. Clamp FFT outputs: [-1e6, 1e6]

### Issue: Baseline underperforming expectations
**Diagnosis**: Hyperparameter mismatch or insufficient training
**Solution**:
1. Verify parameter count matches GSSM
2. Increase epochs if not converged
3. Tune learning rate per method
4. Check for implementation bugs (autograd, shapes)

### Issue: Ablation shows unexpected improvement
**Diagnosis**: Component interaction or overfit
**Solution**:
1. Verify ablation correctly removes component
2. Check for compensatory effects (other components adapting)
3. Increase validation set for robust estimate
4. Investigate per-horizon effects (may benefit some horizons)

---

## Contact and Version Control

**Protocol Version**: 1.0  
**Last Updated**: February 8, 2026  
**Author**: Nadhir Hassen  
**Email**: nadhir.hassen@mila.quebec

**Changelog**:
- v1.0 (2026-02-08): Initial protocol based on paper methodology

---

**Document Status**: ✅ APPROVED FOR USE  
**Ethics Review**: Not required (synthetic data only)  
**Preregistration**: OSF [TBD]
