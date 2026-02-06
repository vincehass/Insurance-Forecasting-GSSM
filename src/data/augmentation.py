"""
Data Augmentation for Insurance Time Series
=========================================

Augmentation techniques to increase training data diversity.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


class InsuranceDataAugmentation:
    """
    Data augmentation for insurance time series.
    
    Techniques:
    - Jittering: Add random noise
    - Scaling: Random amplitude scaling
    - Time warping: Non-linear time distortion
    - Window slicing: Random sub-sequences
    - Magnitude warping: Smooth distortion of magnitudes
    """
    
    def __init__(
        self,
        jitter_sigma: float = 0.03,
        scaling_sigma: float = 0.1,
        time_warp_sigma: float = 0.2,
        magnitude_warp_sigma: float = 0.2,
        augment_prob: float = 0.5
    ):
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.augment_prob = augment_prob
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        if np.random.random() < self.augment_prob:
            # Randomly select augmentation
            aug_type = np.random.choice(['jitter', 'scaling', 'magnitude_warp'])
            
            if aug_type == 'jitter':
                return self.jitter(x)
            elif aug_type == 'scaling':
                return self.scaling(x)
            elif aug_type == 'magnitude_warp':
                return self.magnitude_warp(x)
        
        return x
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add random noise."""
        noise = torch.randn_like(x) * self.jitter_sigma
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling."""
        scale = 1.0 + torch.randn(1).item() * self.scaling_sigma
        return x * scale
    
    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth magnitude distortion."""
        length = x.shape[0]
        warp = torch.randn(length) * self.magnitude_warp_sigma
        smooth_warp = torch.nn.functional.avg_pool1d(
            warp.unsqueeze(0).unsqueeze(0), 
            kernel_size=5, 
            stride=1, 
            padding=2
        ).squeeze()
        return x * (1 + smooth_warp.unsqueeze(-1))
