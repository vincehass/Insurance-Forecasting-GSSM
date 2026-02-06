# Insurance Policy Pricing and Claims Forecasting with GSSM

## 🎯 Overview

This project integrates the **Graph-based Selective Synthesis Methodology (GSSM)** for insurance policy pricing and claims forecasting. We adapt the GSSM architecture, originally developed for EEG time-series forecasting and molecular generation, to tackle complex insurance problems including policy pricing, claims prediction, and risk assessment.

## 📋 Problem Definition

### The Insurance Challenge

Insurance companies face critical forecasting challenges:

1. **Policy Pricing**: Determining optimal premium prices based on historical claims data, customer demographics, and risk factors
2. **Claims Forecasting**: Predicting future claims amounts and frequencies for portfolio risk management
3. **Risk Assessment**: Evaluating long-term risk exposure across different policy types and customer segments
4. **Portfolio Optimization**: Balancing risk and profitability across diverse insurance products

### Why GSSM for Insurance?

The GSSM methodology is uniquely suited for insurance forecasting:

- **Long-Horizon Forecasting**: Insurance requires predictions spanning months to years
- **Non-Stationary Patterns**: Insurance claims exhibit evolving patterns influenced by economic conditions, weather events, and societal changes
- **Temporal Dependencies**: Historical claims patterns influence future risk assessment
- **Sequential Decision-Making**: Policy pricing involves sequential decisions based on evolving market conditions
- **Autocorrelation Modeling**: Insurance claims often exhibit seasonal and cyclical patterns that GSSM's r_AC component can capture

## 🏗️ Architecture Integration

### GSSM Components Adapted for Insurance

1. **State-Space Models (SSM)**

   - **Original Purpose**: Compress historical molecular generation sequences
   - **Insurance Application**: Compress multi-year insurance history including claims, policy changes, and market conditions
   - **Benefit**: Efficient representation of long policy histories

2. **Flow-Selectivity Mechanism**

   - **Original Purpose**: History-aware action selection for molecular synthesis
   - **Insurance Application**: Risk-aware pricing decisions based on complete customer history
   - **Benefit**: Context-sensitive premium adjustments

3. **Autocorrelation Intrinsic Reward (r_AC)**

   - **Original Purpose**: Discover periodic structures in EEG signals
   - **Insurance Application**: Identify seasonal patterns in claims (winter storms, summer travel accidents)
   - **Benefit**: 26.8% performance contribution (most critical component)

4. **FFT Learning**

   - **Original Purpose**: Frequency-domain analysis for EEG bands
   - **Insurance Application**: Spectral analysis of claims cycles (monthly, quarterly, annual patterns)
   - **Benefit**: Enhanced feature extraction from temporal patterns

5. **KL-Frequency Divergence (L_KL-Freq)**
   - **Original Purpose**: Regularization in frequency domain
   - **Insurance Application**: Prevent overfitting to historical claims spikes
   - **Benefit**: Robust generalization to future market conditions

## 📊 Dataset and Task Specifications

### Insurance Dataset

**Primary Dataset**: Insurance Company Claims & Policy Data

- **Temporal Resolution**: Monthly observations
- **Forecast Horizon**: 12-24 months ahead
- **Feature Types**:
  - **Policy Features**: Coverage type, premium amount, deductible, policy duration
  - **Customer Demographics**: Age, location, occupation, credit score
  - **Claims History**: Claim frequency, severity, timing, type
  - **External Factors**: Economic indicators, weather data, local risk factors

**Data Structure**:

```
- Sequence Length: 60 time steps (5 years of history)
- Prediction Horizon: 12-24 steps (1-2 years forecast)
- Feature Dimensions: ~50 features per time step
- Signal Type: Non-stationary time series with seasonal components
```

### Tasks

1. **Task 1: Claims Amount Forecasting**
   - Predict total claims amount for next 12 months
   - Metrics: MSE, MAE, RMSE, R²
2. **Task 2: Claims Frequency Prediction**
   - Forecast number of claims per month
   - Metrics: Count accuracy, Poisson deviance
3. **Task 3: Policy Pricing Optimization**

   - Determine optimal premium pricing
   - Metrics: Revenue optimization, loss ratio

4. **Task 4: Risk Score Forecasting**
   - Predict evolving risk scores for policy holders
   - Metrics: Classification accuracy, AUC-ROC

## 🔬 Experimental Setup

### Baseline Methods

We compare GSSM against state-of-the-art time-series forecasting methods:

1. **PatchTST**: Patching-based transformer for time series
2. **SparseTSF**: Sparse transformer for efficient forecasting
3. **FEDformer**: Frequency Enhanced Decomposed Transformer
4. **Informer**: Attention-based long-sequence forecasting
5. **Traditional Methods**: ARIMA, Prophet, LightGBM

### Evaluation Metrics

**Primary Metrics**:

- **MSE (Mean Squared Error)**: Forecast accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **MAPE (Mean Absolute Percentage Error)**: Relative error
- **R² Score**: Explained variance

**Business Metrics**:

- **Loss Ratio**: (Predicted Claims) / (Premium Revenue)
- **Combined Ratio**: Loss ratio + expense ratio
- **Profit Margin**: Revenue optimization efficiency

**Statistical Validation**:

- **Confidence Intervals**: 95% CI for all metrics
- **Hypothesis Testing**: t-tests against baselines
- **Effect Size**: Cohen's d for practical significance

## 📁 Project Structure

```
insurance_forecasting_gssm/
├── README.md                          # This file
├── Insurance_Policy_Analysis.md       # Detailed problem analysis
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
│
├── src/
│   ├── gssm/                         # Core GSSM implementation
│   │   ├── __init__.py
│   │   ├── gssm_model.py            # Main GSSM architecture
│   │   ├── flow_selectivity.py     # Flow-Selectivity layer
│   │   ├── state_space_layer.py    # SSM layers
│   │   ├── gssm_trainer.py         # Training framework
│   │   └── insurance_gssm.py       # Insurance-specific adaptations
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── insurance_dataset.py     # Dataset loader
│   │   ├── preprocessing.py         # Data preprocessing
│   │   └── augmentation.py          # Data augmentation
│   │
│   ├── models/                       # Model definitions
│   │   ├── __init__.py
│   │   ├── baselines.py             # Baseline implementations
│   │   └── ensemble.py              # Ensemble methods
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics
│       ├── visualization.py         # Plotting functions
│       └── config.py                # Configuration management
│
├── experiments/                      # Experiment scripts
│   ├── train_insurance_gssm.py      # Main training script
│   ├── evaluate_baselines.py       # Baseline comparison
│   ├── ablation_study.py            # Component ablation
│   └── hyperparameter_search.py    # Hyperparameter tuning
│
├── results/                          # Experimental results
│   ├── figures/                     # Generated figures
│   ├── checkpoints/                 # Model checkpoints
│   └── logs/                        # Training logs
│
└── docs/                             # Documentation
    ├── insurance_problem.md         # Problem formulation
    ├── architecture.md              # Architecture details
    └── results_analysis.md          # Results interpretation
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository (from parent GSSM directory)
cd insurance_forecasting_gssm

# Create virtual environment
python -m venv insurance_env
source insurance_env/bin/activate  # On Windows: insurance_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Preparation

```python
# Prepare insurance dataset
python src/data/preprocessing.py --input data/raw_claims.csv --output data/processed/

# Generate synthetic data for testing
python src/data/generate_synthetic.py --output data/synthetic/
```

### Training GSSM on Insurance Data

```python
# Train GSSM model
python experiments/train_insurance_gssm.py \
    --data_path data/processed/ \
    --model_config configs/gssm_insurance.yaml \
    --output_dir results/gssm_baseline/ \
    --epochs 100 \
    --batch_size 32
```

### Evaluation

```python
# Evaluate trained model
python experiments/evaluate_model.py \
    --checkpoint results/gssm_baseline/best_model.pt \
    --test_data data/processed/test.csv \
    --output results/evaluation/

# Compare with baselines
python experiments/compare_baselines.py \
    --models gssm,patchtst,sparsetst,fedformer,informer \
    --output results/comparison/
```

### Ablation Study

```python
# Run ablation study
python experiments/ablation_study.py \
    --base_model results/gssm_baseline/best_model.pt \
    --components r_AC,flow_selectivity,fft_learning,kl_freq \
    --output results/ablation/
```

## 📈 Expected Performance

Based on GSSM's EEG forecasting results, we expect:

### Forecasting Accuracy

- **13-20% improvement** over PatchTST in claims amount prediction
- **20-25% improvement** over traditional ARIMA methods
- **Superior performance** on long-horizon forecasts (12+ months)

### Component Contributions

- **r_AC (Autocorrelation)**: ~25% performance contribution (seasonal patterns)
- **Flow-Selectivity**: ~10% contribution (risk-aware decisions)
- **FFT Learning**: ~5% contribution (cycle detection)
- **KL-Freq Regularization**: ~3% contribution (generalization)

### Business Impact

- **Improved Loss Ratio**: 5-10% reduction through better claims prediction
- **Pricing Optimization**: 3-7% revenue increase through dynamic pricing
- **Risk Management**: 15-20% better early detection of high-risk policies

## 🔍 Key Innovations for Insurance

### 1. Temporal Risk Encoding

```python
# Encode evolving risk profiles over time
risk_history = encode_temporal_risk(
    claims_history=claims_data,
    policy_changes=policy_updates,
    external_factors=market_conditions
)
```

### 2. Autocorrelation-Based Seasonality

```python
# Discover seasonal claims patterns
seasonal_patterns = compute_autocorrelation_reward(
    claims_sequence=monthly_claims,
    lag_range=(1, 24)  # Monthly and annual cycles
)
```

### 3. Flow-Based Pricing Decisions

```python
# History-aware premium pricing
optimal_premium = flow_selectivity_pricing(
    customer_history=compressed_history,
    current_risk=current_risk_factors,
    market_conditions=market_state
)
```

### 4. Multi-Horizon Forecasting

```python
# Simultaneous short and long-term predictions
forecasts = gssm_forecast(
    history=policy_history,
    horizons=[3, 6, 12, 24]  # months
)
```

## 📊 Evaluation Framework

### Statistical Rigor

1. **Train/Validation/Test Split**: 60%/20%/20% temporal split
2. **Cross-Validation**: 5-fold time-series cross-validation
3. **Significance Testing**: p < 0.001 for all claims
4. **Effect Size**: Cohen's d > 0.8 (large effect)
5. **Confidence Intervals**: 95% CI reported for all metrics

### Visualization

Generate comprehensive analysis figures:

```bash
python experiments/generate_figures.py \
    --results_dir results/ \
    --output_dir results/figures/
```

**Generated Figures**:

1. **Claims Forecasting Performance**: MSE/MAE comparison across methods
2. **Ablation Study Analysis**: Component contribution breakdown
3. **Temporal Pattern Visualization**: Claims prediction over time
4. **Seasonal Decomposition**: r_AC captured patterns
5. **Business Metrics Dashboard**: Loss ratio, profit margins, ROI

## 🎓 Research Contributions

### Novel Adaptations

1. **Insurance-Specific State Space**: Tailored SSM for insurance temporal dynamics
2. **Risk-Aware Flow-Selectivity**: Pricing decisions conditioned on risk history
3. **Seasonal Autocorrelation**: Enhanced r_AC for insurance cyclical patterns
4. **Multi-Task Learning**: Simultaneous claims and pricing prediction

### Expected Publications

- **Primary Paper**: "GSSM for Insurance Forecasting: Long-Horizon Claims Prediction with State-Space Models"
- **Workshop Paper**: "Autocorrelation-Based Seasonality Detection in Insurance Time Series"
- **Technical Report**: "Comparative Analysis of Deep Learning Methods for Insurance Forecasting"

## 🤝 Integration with Intrinsic-GFlowNet

This project builds upon the core GSSM methodology from the Intrinsic-GFlowNet repository, adapting:

- ✅ **State-Space Layers**: For temporal compression
- ✅ **Flow-Selectivity**: For decision-making
- ✅ **GFlowNet Training**: For structured prediction
- ✅ **Intrinsic Rewards**: For pattern discovery

**Key Differences**:

- **Domain**: Molecular generation → Insurance forecasting
- **Objective**: Synthesis reward → Forecast accuracy
- **Actions**: Molecular additions → Pricing decisions
- **Sequences**: SMILES strings → Time series

## 📝 License

This project inherits the license from the parent GSSM repository.

## 📧 Contact

For questions and collaborations:

- **Project Lead**: Insurance GSSM Research Team
- **Email**: insurance-gssm@research.ai
- **GitHub**: [Link to repository]

## 🙏 Acknowledgments

- Original GSSM architecture for EEG forecasting
- Intrinsic-GFlowNet methodology
- Insurance domain expertise from industry partners

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Status**: Active Development
