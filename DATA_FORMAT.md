# Insurance Data Format Documentation

## 📊 Data Structure Overview

The insurance forecasting dataset consists of time-series data for multiple insurance policies, with monthly observations containing claims history, policy features, customer demographics, and external factors.

---

## 🗂️ CSV File Format

### Raw Data Structure

The input CSV file has the following columns:

```csv
policy_id,date,claims_amount,claims_count,risk_level,age,premium,location_risk,month,quarter,unemployment_rate,gdp_growth
0,2015-01-01,327.45,0,0,45,1234.56,0.87,0,0,5.23,2.67
0,2015-02-01,412.89,1,0,45,1234.56,0.87,1,1,5.18,2.71
0,2015-03-01,589.23,0,1,45,1234.56,0.87,2,2,5.12,2.69
...
```

### Column Descriptions

| Column                | Type     | Description                             | Example      |
| --------------------- | -------- | --------------------------------------- | ------------ |
| **policy_id**         | int      | Unique policy identifier                | 0, 1, 2, ... |
| **date**              | datetime | Month of observation                    | 2015-01-01   |
| **claims_amount**     | float    | Total claims amount ($)                 | 327.45       |
| **claims_count**      | int      | Number of claims                        | 0, 1, 2      |
| **risk_level**        | int      | Risk category (0=Low, 1=Medium, 2=High) | 0, 1, 2      |
| **age**               | int      | Customer age                            | 25-75        |
| **premium**           | float    | Monthly premium ($)                     | 500-2000     |
| **location_risk**     | float    | Location risk factor                    | 0.5-1.5      |
| **month**             | int      | Month of year (0-11)                    | 0-11         |
| **quarter**           | int      | Quarter (0-3)                           | 0-3          |
| **unemployment_rate** | float    | Local unemployment (%)                  | 4.5-5.5      |
| **gdp_growth**        | float    | GDP growth rate (%)                     | 2.2-2.8      |

---

## 📈 Example Data Records

### Sample Policy Timeline (Policy ID: 42)

```
Policy ID: 42
Customer Age: 52
Base Premium: $1,567.23
Location Risk: 1.12 (slightly elevated)

Month    Date         Claims ($)  Claims Count  Risk Level  Notes
-----    ----         ----------  ------------  ----------  -----
Jan-15   2015-01-01   234.56      0            Low         Normal month
Feb-15   2015-02-01   512.34      1            Medium      Minor claim
Mar-15   2015-03-01   189.67      0            Low         Normal
Apr-15   2015-04-01   3,456.78    1            High        ⚠️ Large claim (accident)
May-15   2015-05-01   298.45      0            Low         Back to normal
Jun-15   2015-06-01   645.23      1            Medium      Summer incident
...
```

### Seasonal Pattern Example

```
Policy exhibits clear seasonal pattern:
- Winter (Dec-Feb): Higher claims (avg $678) - weather-related
- Spring (Mar-May): Moderate claims (avg $423)
- Summer (Jun-Aug): Elevated claims (avg $589) - travel accidents
- Fall (Sep-Nov): Lower claims (avg $312)

Annual Pattern: Claims = 300 + 200*sin(2π*month/12) + trend + noise
```

---

## 🔢 Tensor Shapes in Model

### Input Shape (History)

```python
# Single sample
history: torch.Tensor [60, 50]
         # 60 months of history
         # 50 features per month

# Batched
history: torch.Tensor [batch_size, 60, 50]
         # Example: [32, 60, 50] for batch_size=32
```

### Target Shapes (Multi-Horizon Forecasts)

```python
# Claims amount targets (one per horizon)
targets = {
    '3m':  torch.Tensor [batch_size, 1]  # 3-month ahead
    '6m':  torch.Tensor [batch_size, 1]  # 6-month ahead
    '12m': torch.Tensor [batch_size, 1]  # 12-month ahead
    '24m': torch.Tensor [batch_size, 1]  # 24-month ahead
}

# Claims frequency targets
frequencies = {
    '3m':  torch.LongTensor [batch_size, 1]  # Count
    '6m':  torch.LongTensor [batch_size, 1]
    '12m': torch.LongTensor [batch_size, 1]
    '24m': torch.LongTensor [batch_size, 1]
}

# Risk classification
risk: torch.LongTensor [batch_size, 1]  # 0, 1, or 2
```

---

## 📊 Data Statistics (Synthetic Dataset)

### Dataset Size

```
Number of Policies: 1,000
Months per Policy: 100
Total Records: 100,000
Time Range: Jan 2015 - Apr 2023 (100 months)
```

### Claims Amount Distribution

```
Mean:    $567.34
Median:  $489.23
Std Dev: $421.56
Min:     $0.00
Max:     $6,234.89
P25:     $312.45
P75:     $723.67
P95:     $1,456.78
```

### Claims Frequency Distribution

```
0 claims: 82.3% of months
1 claim:  15.4% of months
2 claims:  2.1% of months
3+ claims: 0.2% of months

Average claims per month: 0.21
```

### Risk Level Distribution

```
Low Risk (0):    45.2% of records
Medium Risk (1): 38.7% of records
High Risk (2):   16.1% of records
```

---

## 🎯 Data Processing Pipeline

### 1. Raw CSV → Preprocessed Features

```python
# Input: raw_data.csv
# Output: processed features with sliding windows

Raw Data:
┌─────────────┬──────────┬──────────────┬───────┐
│ policy_id   │ date     │ claims_amt   │ ...   │
├─────────────┼──────────┼──────────────┼───────┤
│ 0           │ 2015-01  │ 327.45       │ ...   │
│ 0           │ 2015-02  │ 412.89       │ ...   │
│ ...         │ ...      │ ...          │ ...   │
└─────────────┴──────────┴──────────────┴───────┘

↓ Preprocessing (normalization, feature engineering)

Processed Features:
┌──────────────────────────────────────────────────┐
│ [60 months × 50 features] normalized history    │
├──────────────────────────────────────────────────┤
│ Month 1: [age, premium, claims, risk, ...]      │
│ Month 2: [age, premium, claims, risk, ...]      │
│ ...                                              │
│ Month 60: [age, premium, claims, risk, ...]     │
└──────────────────────────────────────────────────┘
```

### 2. Sliding Window Creation

```python
# For each policy, create overlapping windows

Policy Timeline: [100 months total]
├─────────────────────┬─────────┐
│ History (60 months) │ Future  │
├─────────────────────┴─────────┤
│ Window 1: Month 1-60 → Predict 61-84
│ Window 2: Month 2-61 → Predict 62-85
│ Window 3: Month 3-62 → Predict 63-86
│ ...
└──────────────────────────────────
```

### 3. Feature Normalization

```python
# Z-score normalization for each feature
normalized_value = (value - mean) / std

Example:
Raw claims_amount: $567.34
Mean: $500.00
Std: $200.00
Normalized: (567.34 - 500.00) / 200.00 = 0.337
```

---

## 📝 Example: Complete Data Sample

### Input Sample

```python
{
    'history': torch.Tensor([
        # Month 1 (60 months ago)
        [0.34, -0.56, 0.12, ..., 0.89],  # 50 features

        # Month 2 (59 months ago)
        [0.21, -0.48, 0.34, ..., 0.76],

        # ...

        # Month 60 (current)
        [0.45, -0.32, 0.67, ..., 0.94]
    ]),  # Shape: [60, 50]

    'targets': {
        '3m':  torch.Tensor([0.234]),   # Normalized claims 3 months ahead
        '6m':  torch.Tensor([0.456]),   # 6 months ahead
        '12m': torch.Tensor([0.678]),   # 12 months ahead
        '24m': torch.Tensor([0.789])    # 24 months ahead
    },

    'frequencies': {
        '3m':  torch.LongTensor([1]),   # Expected 1 claim
        '6m':  torch.LongTensor([0]),   # Expected 0 claims
        '12m': torch.LongTensor([2]),   # Expected 2 claims
        '24m': torch.LongTensor([1])    # Expected 1 claim
    },

    'risk': torch.LongTensor([1])  # Medium risk
}
```

### Model Output

```python
{
    'claims_amount_3m':  0.245,  # Predicted $589.23
    'claims_amount_6m':  0.467,  # Predicted $612.45
    'claims_amount_12m': 0.689,  # Predicted $734.56
    'claims_amount_24m': 0.801,  # Predicted $856.78

    'claims_count_3m':  1.23,  # Expected ~1 claim
    'claims_count_6m':  0.87,  # Expected ~1 claim
    'claims_count_12m': 2.14,  # Expected ~2 claims
    'claims_count_24m': 1.56,  # Expected ~2 claims

    'risk_probs': [0.12, 0.76, 0.12],  # [Low, Medium, High]
    'risk_prediction': 1,  # Medium risk

    'pricing_probs': [0.05, 0.08, 0.45, 0.25, 0.12, 0.03, 0.02, 0.00],
    'pricing_action': 2,  # Maintain premium

    'loss_ratio': 0.68  # Predicted 68% loss ratio
}
```

---

## 🔍 Feature Details (50 Features)

### Core Features (12)

1. **claims_amount** - Historical claims ($)
2. **claims_count** - Number of claims
3. **age** - Customer age
4. **premium** - Monthly premium
5. **location_risk** - Geographic risk factor
6. **month** - Month of year (0-11)
7. **quarter** - Quarter (0-3)
8. **unemployment_rate** - Local unemployment
9. **gdp_growth** - Economic growth
10. **risk_level** - Historical risk category
11. **month_sin** - Cyclic encoding sin(2π\*month/12)
12. **month_cos** - Cyclic encoding cos(2π\*month/12)

### Lag Features (24)

For claims_amount and claims_count:

- lag_1, lag_3, lag_6, lag_12 (last 1, 3, 6, 12 months)
- rolling_mean_3, rolling_mean_6, rolling_mean_12
- rolling_std_3, rolling_std_6, rolling_std_12

### Temporal Features (8)

- **quarter_sin** - Cyclic quarter encoding
- **quarter_cos** - Cyclic quarter encoding
- **day_of_week** - Day of week (if daily data)
- **day_of_year** - Day of year
- **year** - Year number
- **days_since_policy_start** - Policy age
- **days_until_renewal** - Days to renewal
- **season** - Season indicator

### Derived Features (6)

- **claims_per_premium** - Claims/Premium ratio
- **risk_score** - Computed risk score
- **trend** - Linear trend component
- **seasonal_component** - Extracted seasonal pattern
- **residual** - Detrended, deseasonalized residual
- **volatility** - Claims volatility measure

**Total: 50 features per time step**

---

## 💾 Data Loading Example

```python
from data.insurance_dataset import create_dataloaders, create_synthetic_dataset

# Generate synthetic data
create_synthetic_dataset(
    output_path='data/synthetic_insurance.csv',
    num_policies=1000,
    num_months=100
)

# Load data
train_loader, val_loader, test_loader = create_dataloaders(
    data_path='data/synthetic_insurance.csv',
    batch_size=32,
    history_length=60,
    forecast_horizons=[3, 6, 12, 24]
)

# Inspect a batch
batch = next(iter(train_loader))
print(f"History shape: {batch['history'].shape}")          # [32, 60, 50]
print(f"Target 12m shape: {batch['targets']['12m'].shape}") # [32, 1]
print(f"Risk shape: {batch['risk'].shape}")                # [32, 1]
```

---

## 📊 Visualization Examples

### Temporal Pattern

```
Claims Amount Over Time (Policy ID: 123)
$2000 ┤                            ╭╮
      │                          ╭╯╰╮
$1500 ┤                        ╭╯   ╰╮
      │           ╭╮         ╭╯      ╰╮
$1000 ┤         ╭╯╰╮      ╭╯         ╰╮
      │       ╭╯   ╰╮   ╭╯            ╰╮
 $500 ┤     ╭╯      ╰╮╭╯               ╰
      │   ╭╯         ╰╯
    0 ┼───┴─────────────────────────────→
      Jan  Apr  Jul  Oct  Jan  Apr  Jul  Oct

Clear seasonal pattern with annual cycle
```

### Risk Distribution

```
Risk Level Distribution
Low (0)    ████████████████████████████████████ 45.2%
Medium (1) ██████████████████████████████ 38.7%
High (2)   ████████████████ 16.1%
```

---

## 🎯 Key Takeaways

1. **Time-Series Nature**: Data is sequential with temporal dependencies
2. **Multi-Horizon**: Predict 3, 6, 12, 24 months ahead simultaneously
3. **Rich Features**: 50 features including temporal, economic, and customer data
4. **Seasonal Patterns**: Strong annual cycles in claims data
5. **Sparse Events**: Most months have 0 claims, occasional large claims
6. **Multi-Task**: Predict amounts, frequencies, and risk levels together

---

**Generated**: February 2026  
**Version**: 1.0  
**For**: Insurance-Forecasting-GSSM Repository
