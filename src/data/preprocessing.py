"""
Insurance Data Preprocessing Module
==================================

Handles data cleaning, feature engineering, and transformations
for insurance forecasting tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class InsurancePreprocessor:
    """
    Preprocessor for insurance claims and policy data.
    
    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Feature engineering
    - Temporal encoding
    - Normalization
    """
    
    def __init__(
        self,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 3.0,
        missing_strategy: str = 'forward_fill',
        add_temporal_features: bool = True,
        add_lag_features: bool = True,
        lag_periods: List[int] = [1, 3, 6, 12]
    ):
        """
        Initialize preprocessor.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            missing_strategy: Strategy for missing values ('forward_fill', 'mean', 'median')
            add_temporal_features: Whether to add temporal features
            add_lag_features: Whether to add lag features
            lag_periods: Lag periods for features
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_strategy = missing_strategy
        self.add_temporal_features = add_temporal_features
        self.add_lag_features = add_lag_features
        self.lag_periods = lag_periods
        
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'InsurancePreprocessor':
        """Fit preprocessor on training data."""
        # Fit scaler on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.scaler.fit(df[numeric_cols].fillna(0))
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Add temporal features
        if self.add_temporal_features:
            df = self._add_temporal_features(df)
        
        # Add lag features
        if self.add_lag_features:
            df = self._add_lag_features(df)
        
        # Normalize
        if self.fitted:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.scaler.transform(df[numeric_cols].fillna(0))
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(df).transform(df)
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        if self.missing_strategy == 'forward_fill':
            return df.fillna(method='ffill').fillna(method='bfill')
        elif self.missing_strategy == 'mean':
            return df.fillna(df.mean())
        elif self.missing_strategy == 'median':
            return df.fillna(df.median())
        else:
            return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
                df[col] = df[col].clip(lower, upper)
            
            elif self.outlier_method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(
                    mean - self.outlier_threshold * std,
                    mean + self.outlier_threshold * std
                )
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Cyclic encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series."""
        if 'policy_id' in df.columns:
            grouped = df.groupby('policy_id')
            
            for col in ['claims_amount', 'claims_count']:
                if col in df.columns:
                    for lag in self.lag_periods:
                        df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
                        df[f'{col}_rolling_mean_{lag}'] = grouped[col].rolling(lag, min_periods=1).mean().values
                        df[f'{col}_rolling_std_{lag}'] = grouped[col].rolling(lag, min_periods=1).std().values
        
        return df


def load_and_preprocess(
    data_path: str,
    train: bool = True,
    preprocessor: Optional[InsurancePreprocessor] = None
) -> Tuple[pd.DataFrame, InsurancePreprocessor]:
    """
    Load and preprocess insurance data.
    
    Args:
        data_path: Path to data file
        train: Whether this is training data
        preprocessor: Pre-fitted preprocessor (for validation/test)
        
    Returns:
        Preprocessed DataFrame and preprocessor
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize or use provided preprocessor
    if preprocessor is None:
        preprocessor = InsurancePreprocessor()
        if train:
            df = preprocessor.fit_transform(df)
        else:
            raise ValueError("Preprocessor must be provided for non-training data")
    else:
        df = preprocessor.transform(df)
    
    return df, preprocessor
