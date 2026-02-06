"""
Insurance Dataset for Time-Series Forecasting
============================================

Handles loading and batching of insurance policy and claims data
for multi-horizon forecasting tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class InsuranceDataset(Dataset):
    """
    Insurance time-series dataset for GSSM training.
    
    Each sample contains:
    - Historical features (60 months of policy and claims data)
    - Target forecasts (claims amounts and frequencies at multiple horizons)
    - Risk labels
    """
    
    def __init__(
        self,
        data_path: str,
        history_length: int = 60,  # 5 years
        forecast_horizons: List[int] = [3, 6, 12, 24],
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'claims_amount',
        frequency_column: str = 'claims_count',
        risk_column: str = 'risk_level',
        normalize: bool = True,
        train: bool = True
    ):
        """
        Initialize Insurance Dataset.
        
        Args:
            data_path: Path to CSV data file
            history_length: Length of historical sequence
            forecast_horizons: List of forecast horizons in months
            feature_columns: List of feature column names
            target_column: Name of claims amount column
            frequency_column: Name of claims count column
            risk_column: Name of risk level column
            normalize: Whether to normalize features
            train: Whether this is training data (for normalization fitting)
        """
        self.data_path = Path(data_path)
        self.history_length = history_length
        self.forecast_horizons = sorted(forecast_horizons)
        self.max_horizon = max(forecast_horizons)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.frequency_column = frequency_column
        self.risk_column = risk_column
        self.normalize = normalize
        self.train = train
        
        # Load data
        self.data = self._load_data()
        
        # Compute normalization statistics
        if normalize and train:
            self._compute_normalization_stats()
        
        # Create samples
        self.samples = self._create_samples()
    
    def _load_data(self) -> pd.DataFrame:
        """Load insurance data from CSV."""
        if self.data_path.suffix == '.csv':
            data = pd.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            data = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Ensure time column is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')
        
        return data
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        if self.feature_columns is None:
            # Use all numeric columns except targets
            exclude_cols = [self.target_column, self.frequency_column, 
                          self.risk_column, 'date', 'policy_id']
            self.feature_columns = [c for c in self.data.columns 
                                   if c not in exclude_cols and 
                                   pd.api.types.is_numeric_dtype(self.data[c])]
        
        feature_data = self.data[self.feature_columns].values
        self.feature_mean = np.nanmean(feature_data, axis=0)
        self.feature_std = np.nanstd(feature_data, axis=0) + 1e-8
        
        # Normalize target for loss scaling
        target_data = self.data[self.target_column].values
        self.target_mean = np.nanmean(target_data)
        self.target_std = np.nanstd(target_data) + 1e-8
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using computed statistics."""
        if not self.normalize:
            return features
        return (features - self.feature_mean) / self.feature_std
    
    def _normalize_target(self, target: float) -> float:
        """Normalize target using computed statistics."""
        if not self.normalize:
            return target
        return (target - self.target_mean) / self.target_std
    
    def _denormalize_target(self, normalized_target: float) -> float:
        """Denormalize target for evaluation."""
        if not self.normalize:
            return normalized_target
        return normalized_target * self.target_std + self.target_mean
    
    def _create_samples(self) -> List[Dict]:
        """
        Create samples from time-series data.
        
        Each sample contains:
        - history: [history_length, num_features]
        - targets: Dict of forecast targets for each horizon
        - frequencies: Dict of claim counts for each horizon
        - risk: Risk level (0=low, 1=medium, 2=high)
        """
        samples = []
        
        # Group by policy if policy_id exists
        if 'policy_id' in self.data.columns:
            grouped = self.data.groupby('policy_id')
        else:
            grouped = [(0, self.data)]
        
        for policy_id, policy_data in grouped:
            policy_data = policy_data.sort_values('date') if 'date' in policy_data.columns else policy_data
            
            # Extract features
            features = policy_data[self.feature_columns].values
            targets = policy_data[self.target_column].values
            frequencies = policy_data[self.frequency_column].values
            risks = policy_data[self.risk_column].values if self.risk_column in policy_data.columns else np.zeros(len(policy_data))
            
            # Create sliding windows
            max_start = len(features) - self.history_length - self.max_horizon
            
            for start_idx in range(max(0, max_start)):
                end_idx = start_idx + self.history_length
                
                # Historical features
                history = features[start_idx:end_idx]
                
                # Skip if too many NaNs
                if np.isnan(history).sum() / history.size > 0.1:
                    continue
                
                # Normalize features
                history = self._normalize_features(history)
                
                # Target forecasts for each horizon
                sample_targets = {}
                sample_frequencies = {}
                
                for horizon in self.forecast_horizons:
                    target_idx = end_idx + horizon - 1
                    if target_idx < len(targets):
                        sample_targets[f'{horizon}m'] = self._normalize_target(targets[target_idx])
                        sample_frequencies[f'{horizon}m'] = frequencies[target_idx]
                    else:
                        # Pad with last available value
                        sample_targets[f'{horizon}m'] = self._normalize_target(targets[-1])
                        sample_frequencies[f'{horizon}m'] = frequencies[-1]
                
                # Risk level (use median risk in history)
                risk_level = int(np.median(risks[start_idx:end_idx]))
                
                samples.append({
                    'history': history,
                    'targets': sample_targets,
                    'frequencies': sample_frequencies,
                    'risk': risk_level,
                    'policy_id': policy_id
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert to tensors
        history = torch.FloatTensor(sample['history'])
        
        targets = {
            k: torch.FloatTensor([v]) for k, v in sample['targets'].items()
        }
        
        frequencies = {
            k: torch.LongTensor([int(v)]) for k, v in sample['frequencies'].items()
        }
        
        risk = torch.LongTensor([sample['risk']])
        
        return {
            'history': history,
            'targets': targets,
            'frequencies': frequencies,
            'risk': risk
        }
    
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics for use in other datasets."""
        return {
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }
    
    def set_normalization_stats(self, stats: Dict[str, np.ndarray]):
        """Set normalization statistics from training set."""
        self.feature_mean = stats['feature_mean']
        self.feature_std = stats['feature_std']
        self.target_mean = stats['target_mean']
        self.target_std = stats['target_std']


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching insurance samples.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    # Stack histories
    histories = torch.stack([item['history'] for item in batch])
    
    # Stack targets for each horizon
    targets = {}
    for horizon_key in batch[0]['targets'].keys():
        targets[horizon_key] = torch.stack([item['targets'][horizon_key] for item in batch])
    
    # Stack frequencies
    frequencies = {}
    for horizon_key in batch[0]['frequencies'].keys():
        frequencies[horizon_key] = torch.stack([item['frequencies'][horizon_key] for item in batch])
    
    # Stack risks
    risks = torch.stack([item['risk'] for item in batch])
    
    return {
        'history': histories,
        'targets': targets,
        'frequencies': frequencies,
        'risk': risks
    }


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    history_length: int = 60,
    forecast_horizons: List[int] = [3, 6, 12, 24],
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        history_length: Length of historical sequence
        forecast_horizons: List of forecast horizons
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Train, validation, and test dataloaders
    """
    # Create full dataset
    full_dataset = InsuranceDataset(
        data_path=data_path,
        history_length=history_length,
        forecast_horizons=forecast_horizons,
        normalize=True,
        train=True
    )
    
    # Get normalization stats
    norm_stats = full_dataset.get_normalization_stats()
    
    # Split dataset
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_dataset(
    output_path: str,
    num_policies: int = 1000,
    num_months: int = 100,
    seed: int = 42
):
    """
    Generate synthetic insurance dataset for testing.
    
    Args:
        output_path: Path to save synthetic data
        num_policies: Number of insurance policies
        num_months: Number of months of data per policy
        seed: Random seed
    """
    np.random.seed(seed)
    
    data_list = []
    
    for policy_id in range(num_policies):
        # Generate policy characteristics
        age = np.random.randint(25, 75)
        location_risk = np.random.uniform(0.5, 1.5)
        base_premium = np.random.uniform(500, 2000)
        
        # Generate time series
        for month in range(num_months):
            date = pd.Timestamp('2015-01-01') + pd.DateOffset(months=month)
            
            # Seasonal component (annual)
            seasonal = np.sin(2 * np.pi * month / 12) * 200
            
            # Trend
            trend = month * 2
            
            # Random noise
            noise = np.random.normal(0, 100)
            
            # Claims amount (with occasional large claims)
            base_claims = 300 + seasonal + trend + noise
            if np.random.random() < 0.05:  # 5% chance of large claim
                base_claims += np.random.uniform(1000, 5000)
            claims_amount = max(0, base_claims * location_risk)
            
            # Claims count (Poisson)
            claims_count = np.random.poisson(0.2)
            
            # Risk level
            if claims_amount < 500:
                risk_level = 0  # Low
            elif claims_amount < 1500:
                risk_level = 1  # Medium
            else:
                risk_level = 2  # High
            
            # Create record
            record = {
                'policy_id': policy_id,
                'date': date,
                'claims_amount': claims_amount,
                'claims_count': claims_count,
                'risk_level': risk_level,
                'age': age,
                'premium': base_premium,
                'location_risk': location_risk,
                'month': month % 12,
                'quarter': month % 4,
                'unemployment_rate': 5.0 + np.random.normal(0, 0.5),
                'gdp_growth': 2.5 + np.random.normal(0, 0.3)
            }
            
            data_list.append(record)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_list)
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic dataset created: {output_path}")
    print(f"   - {num_policies} policies")
    print(f"   - {num_months} months per policy")
    print(f"   - Total records: {len(df)}")
