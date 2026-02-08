"""
Evaluation Metrics for Insurance Forecasting
==========================================

Comprehensive metrics for assessing insurance forecasting performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional


class InsuranceMetrics:
    """
    Comprehensive metrics for insurance forecasting evaluation.
    
    Includes:
    - Forecasting accuracy metrics (MSE, MAE, RMSE, MAPE, R²)
    - Business metrics (Loss Ratio, Profit Margin)
    - Statistical validation (Confidence Intervals, Hypothesis Tests)
    - Distribution metrics (Quantile Loss, CRPS)
    """
    
    def __init__(self):
        pass
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all forecasting metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        # Median Absolute Error
        metrics['medae'] = np.median(np.abs(y_true - y_pred))
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        return metrics
    
    def compute_business_metrics(
        self,
        claims_pred: np.ndarray,
        claims_true: np.ndarray,
        premiums: np.ndarray,
        expenses: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute business-relevant metrics.
        
        Args:
            claims_pred: Predicted claims amounts
            claims_true: True claims amounts
            premiums: Premium amounts
            expenses: Expense amounts
            
        Returns:
            Dictionary of business metrics
        """
        metrics = {}
        
        # Loss Ratio (predicted)
        metrics['loss_ratio_pred'] = np.sum(claims_pred) / np.sum(premiums)
        
        # Loss Ratio (actual)
        metrics['loss_ratio_true'] = np.sum(claims_true) / np.sum(premiums)
        
        # Loss Ratio Error
        metrics['loss_ratio_error'] = abs(metrics['loss_ratio_pred'] - metrics['loss_ratio_true'])
        
        # Combined Ratio (if expenses provided)
        if expenses is not None:
            metrics['combined_ratio_pred'] = (np.sum(claims_pred) + np.sum(expenses)) / np.sum(premiums)
            metrics['combined_ratio_true'] = (np.sum(claims_true) + np.sum(expenses)) / np.sum(premiums)
        
        # Profit Margin
        metrics['profit_margin_pred'] = (np.sum(premiums) - np.sum(claims_pred)) / np.sum(premiums) * 100
        metrics['profit_margin_true'] = (np.sum(premiums) - np.sum(claims_true)) / np.sum(premiums) * 100
        
        return metrics
    
    def compute_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for metrics using bootstrap.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary of (lower, upper) confidence intervals
        """
        n_samples = len(y_true)
        bootstrap_mse = []
        bootstrap_mae = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metrics
            bootstrap_mse.append(mean_squared_error(y_true_boot, y_pred_boot))
            bootstrap_mae.append(mean_absolute_error(y_true_boot, y_pred_boot))
        
        # Compute confidence intervals
        alpha = (1 - confidence) / 2
        
        ci = {}
        ci['mse'] = (
            np.percentile(bootstrap_mse, alpha * 100),
            np.percentile(bootstrap_mse, (1 - alpha) * 100)
        )
        ci['mae'] = (
            np.percentile(bootstrap_mae, alpha * 100),
            np.percentile(bootstrap_mae, (1 - alpha) * 100)
        )
        
        return ci
    
    def hypothesis_test(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        test_type: str = 'paired_t'
    ) -> Dict[str, float]:
        """
        Perform hypothesis testing between two models.
        
        Args:
            errors_model1: Errors from model 1
            errors_model2: Errors from model 2
            test_type: Type of test ('paired_t', 'wilcoxon')
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        if test_type == 'paired_t':
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(errors_model1, errors_model2)
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            
            # Cohen's d (effect size)
            mean_diff = np.mean(errors_model1 - errors_model2)
            std_diff = np.std(errors_model1 - errors_model2)
            results['cohens_d'] = mean_diff / std_diff if std_diff > 0 else 0
            
        elif test_type == 'wilcoxon':
            # Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(errors_model1, errors_model2)
            results['statistic'] = statistic
            results['p_value'] = p_value
        
        # Significance level
        results['significant_001'] = results['p_value'] < 0.001
        results['significant_005'] = results['p_value'] < 0.01
        results['significant_01'] = results['p_value'] < 0.05
        
        return results
    
    def quantile_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, float]:
        """
        Compute quantile loss for probabilistic forecasting.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values (assumed to be quantiles)
            quantiles: List of quantiles
            
        Returns:
            Dictionary of quantile losses
        """
        losses = {}
        
        for q in quantiles:
            errors = y_true - y_pred
            losses[f'quantile_{q}'] = np.mean(
                np.maximum(q * errors, (q - 1) * errors)
            )
        
        return losses
    
    def compute_temporal_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time_windows: List[Tuple[int, int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy metrics for different time windows.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            time_windows: List of (start, end) time window tuples
            
        Returns:
            Dictionary of metrics per time window
        """
        temporal_metrics = {}
        
        for start, end in time_windows:
            window_name = f'{start}-{end}'
            y_true_window = y_true[start:end]
            y_pred_window = y_pred[start:end]
            
            temporal_metrics[window_name] = self.compute_metrics(
                y_true_window,
                y_pred_window
            )
        
        return temporal_metrics


def compare_models(
    model_predictions: Dict[str, np.ndarray],
    ground_truth: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models and create results table.
    
    Args:
        model_predictions: Dictionary of {model_name: predictions}
        ground_truth: Ground truth values
        
    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    
    metrics_calc = InsuranceMetrics()
    results = []
    
    for model_name, predictions in model_predictions.items():
        metrics = metrics_calc.compute_metrics(ground_truth, predictions)
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df[['model', 'mse', 'mae', 'rmse', 'r2', 'mape']]
    df = df.sort_values('mse')
    
    return df
