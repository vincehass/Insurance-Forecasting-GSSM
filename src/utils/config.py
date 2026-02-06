"""
Configuration management utilities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path}")
    
    print(f"✅ Config saved: {save_path}")


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for Insurance GSSM."""
    return {
        'model': {
            'd_model': 256,
            'd_state': 64,
            'num_layers': 6,
            'dropout': 0.1,
            'max_history_length': 60,
            'forecast_horizons': [3, 6, 12, 24],
            'num_pricing_actions': 8,
            'use_seasonal_encoding': True,
            'use_insurance_autocorrelation': True,
            'use_cycle_detection': True,
            'seasonal_lags': [12, 24]
        },
        'training': {
            'batch_size': 32,
            'epochs': 150,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'val_split': 0.2,
            'test_split': 0.1
        },
        'data': {
            'history_length': 60,
            'forecast_horizons': [3, 6, 12, 24],
            'normalize': True,
            'add_temporal_features': True,
            'add_lag_features': True
        }
    }
