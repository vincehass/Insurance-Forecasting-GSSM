"""
Insurance GSSM: GSSM Adapted for Insurance Forecasting
=====================================================

This module adapts the GSSM architecture for insurance policy pricing
and claims forecasting. Key adaptations:

1. Temporal encoding for monthly insurance data
2. Autocorrelation reward for seasonal claims patterns
3. Flow-selectivity for premium pricing decisions
4. FFT learning for insurance cycle detection
5. Multi-horizon forecasting (3, 6, 12, 24 months)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from torch.fft import fft, ifft, fftfreq

from .gssm_model import GSSMModel
from .flow_selectivity import FlowSelectivityLayer
from .state_space_layer import StateSpaceLayer


class InsuranceGSSM(GSSMModel):
    """
    GSSM model adapted for insurance forecasting tasks.
    
    This model inherits from the base GSSM and adds insurance-specific components:
    - Seasonal autocorrelation discovery
    - Multi-horizon forecasting heads
    - Business metric optimization
    - Risk-aware pricing decisions
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        d_state: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_history_length: int = 60,  # 5 years of monthly data
        forecast_horizons: List[int] = [3, 6, 12, 24],  # Forecast horizons in months
        num_pricing_actions: int = 8,  # Premium adjustment actions
        use_seasonal_encoding: bool = True,
        use_insurance_autocorrelation: bool = True,
        use_cycle_detection: bool = True,
        seasonal_lags: List[int] = [12, 24],  # Monthly, Annual cycles
        **kwargs
    ):
        """
        Initialize Insurance GSSM model.
        
        Args:
            num_features: Number of input features per time step
            d_model: Model embedding dimension
            d_state: SSM compressed state dimension
            num_layers: Number of SSM layers
            dropout: Dropout probability
            max_history_length: Maximum policy history length (months)
            forecast_horizons: List of forecast horizons in months
            num_pricing_actions: Number of premium pricing actions
            use_seasonal_encoding: Whether to use seasonal time encoding
            use_insurance_autocorrelation: Whether to use insurance-specific r_AC
            use_cycle_detection: Whether to use FFT cycle detection
            seasonal_lags: List of seasonal lags for autocorrelation (months)
        """
        # Initialize base GSSM with vocabulary size set to num_features
        super().__init__(
            vocab_size=num_features,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_history_length,
            **kwargs
        )
        
        self.num_features = num_features
        self.max_history_length = max_history_length
        self.forecast_horizons = sorted(forecast_horizons)
        self.num_pricing_actions = num_pricing_actions
        self.use_seasonal_encoding = use_seasonal_encoding
        self.use_insurance_autocorrelation = use_insurance_autocorrelation
        self.use_cycle_detection = use_cycle_detection
        self.seasonal_lags = seasonal_lags
        
        # Replace embedding with projection for continuous features
        self.embedding = nn.Linear(num_features, d_model)
        
        # Seasonal time encoding
        if use_seasonal_encoding:
            self.seasonal_encoder = SeasonalEncoder(d_model)
        
        # Multi-horizon forecasting heads
        self.forecast_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Predict claims amount
            )
            for h in forecast_horizons
        })
        
        # Claims frequency prediction heads
        self.frequency_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)  # Predict claim count (Poisson)
            )
            for h in forecast_horizons
        })
        
        # Risk score prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # Low, Medium, High risk
        )
        
        # Premium pricing head (replaces flow-selectivity num_actions)
        self.pricing_head = FlowSelectivityLayer(
            d_model=d_model,
            d_state=d_state,
            num_actions=num_pricing_actions,
            dropout=dropout,
            temperature=self.temperature,
            use_entropy_regularization=self.use_entropy_regularization
        )
        
        # Insurance-specific autocorrelation module
        if use_insurance_autocorrelation:
            self.autocorr_module = InsuranceAutocorrelationModule(
                d_model=d_model,
                max_lag=max(seasonal_lags),
                seasonal_lags=seasonal_lags
            )
        
        # FFT-based cycle detection module
        if use_cycle_detection:
            self.cycle_detector = InsuranceCycleDetector(
                d_model=d_model,
                sampling_rate=1.0/30.0,  # Monthly (1/30 days)
                relevant_cycles=[1/12, 1/4, 1/2, 1]  # Monthly, quarterly, biannual, annual
            )
        
        # Business metrics head
        self.loss_ratio_head = nn.Linear(d_model, 1)
        
        # Initialize new parameters
        self._initialize_insurance_parameters()
    
    def _initialize_insurance_parameters(self):
        """Initialize insurance-specific parameters."""
        # Initialize forecast heads
        for heads in [self.forecast_heads, self.frequency_heads]:
            for head in heads.values():
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
        
        # Initialize risk head
        for layer in self.risk_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _get_embeddings(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Get input embeddings from continuous features.
        
        Args:
            input_features: Input features [batch, length, num_features]
            
        Returns:
            Embedded inputs [batch, length, d_model]
        """
        batch_size, seq_length, _ = input_features.shape
        
        # Project features to embedding space
        embeddings = self.embedding(input_features)
        
        # Add positional encoding
        position_embeddings = self.positional_encoding[:seq_length].unsqueeze(0)
        embeddings = embeddings + position_embeddings
        
        # Add seasonal encoding if enabled
        if self.use_seasonal_encoding:
            # Extract time indices (assuming last feature is month)
            seasonal_embeddings = self.seasonal_encoder(seq_length, embeddings.device)
            embeddings = embeddings + seasonal_embeddings.unsqueeze(0)
        
        return embeddings
    
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        initial_states: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
        return_extras: bool = False,
        compute_business_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Insurance GSSM model.
        
        Args:
            input_features: Input features [batch, length, num_features]
            attention_mask: Attention mask [batch, length]
            initial_states: Initial states for SSM layers
            return_states: Whether to return final states
            return_extras: Whether to return extra outputs
            compute_business_metrics: Whether to compute business metrics
            
        Returns:
            Dictionary of outputs including forecasts
        """
        batch_size, seq_length, _ = input_features.shape
        
        # Get embeddings
        embeddings = self._get_embeddings(input_features)
        
        # Apply SSM layers
        ssm_output, final_states = self._apply_ssm_layers(embeddings, initial_states)
        
        # Apply layer normalization
        ssm_output = self.layer_norm(ssm_output)
        
        # Get compressed history state (from last SSM layer)
        history_state = final_states[-1]
        
        # Get the last time step representation for forecasting
        last_representation = ssm_output[:, -1, :]
        
        # Multi-horizon forecasting
        forecasts = {}
        for horizon in self.forecast_horizons:
            head = self.forecast_heads[f'horizon_{horizon}']
            forecasts[f'claims_amount_{horizon}m'] = head(last_representation).squeeze(-1)
        
        # Claims frequency forecasting
        frequencies = {}
        for horizon in self.forecast_horizons:
            head = self.frequency_heads[f'horizon_{horizon}']
            # Use exponential activation for counts
            frequencies[f'claims_count_{horizon}m'] = torch.exp(
                head(last_representation).squeeze(-1)
            )
        
        # Risk score prediction
        risk_logits = self.risk_head(last_representation)
        risk_probs = F.softmax(risk_logits, dim=-1)
        
        # Premium pricing decision using flow-selectivity
        pricing_probs, pricing_extras = self.pricing_head(
            history_state,
            last_representation,
            return_extras=return_extras
        )
        
        # Prepare outputs
        outputs = {
            **forecasts,
            **frequencies,
            'risk_probs': risk_probs,
            'risk_logits': risk_logits,
            'pricing_probs': pricing_probs,
            'ssm_output': ssm_output,
            'history_state': history_state,
            'last_representation': last_representation
        }
        
        # Add autocorrelation reward if enabled
        if self.use_insurance_autocorrelation and return_extras:
            # Extract claims history (assuming it's in the input features)
            autocorr_reward = self.autocorr_module(input_features, ssm_output)
            outputs['autocorr_reward'] = autocorr_reward
        
        # Add cycle detection if enabled
        if self.use_cycle_detection and return_extras:
            cycle_features = self.cycle_detector(input_features, ssm_output)
            outputs['cycle_features'] = cycle_features
        
        # Add business metrics if requested
        if compute_business_metrics:
            loss_ratio_pred = torch.sigmoid(self.loss_ratio_head(last_representation))
            outputs['loss_ratio'] = loss_ratio_pred.squeeze(-1)
        
        # Add states if requested
        if return_states:
            outputs['final_states'] = final_states
        
        # Add pricing extras if requested
        if return_extras:
            outputs['pricing_extras'] = pricing_extras
        
        return outputs
    
    def forecast(
        self,
        input_features: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate forecasts for specified horizon.
        
        Args:
            input_features: Input features [batch, length, num_features]
            horizon: Forecast horizon in months (None for all horizons)
            
        Returns:
            Dictionary of forecasts
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_features, compute_business_metrics=True)
            
            if horizon is not None:
                # Return only specified horizon
                return {
                    'claims_amount': outputs[f'claims_amount_{horizon}m'],
                    'claims_count': outputs[f'claims_count_{horizon}m'],
                    'risk_probs': outputs['risk_probs'],
                    'loss_ratio': outputs['loss_ratio']
                }
            else:
                # Return all horizons
                return outputs
    
    def recommend_pricing(
        self,
        input_features: torch.Tensor,
        action_meanings: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recommend premium pricing action.
        
        Args:
            input_features: Input features [batch, length, num_features]
            action_meanings: List of action descriptions
            
        Returns:
            Pricing recommendation dictionary
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(input_features)
            
            pricing_probs = outputs['pricing_probs']
            recommended_action = torch.argmax(pricing_probs, dim=-1)
            confidence = pricing_probs.max(dim=-1)[0]
            
            if action_meanings is None:
                action_meanings = [
                    "Decrease 10%",
                    "Decrease 5%",
                    "Maintain",
                    "Increase 5%",
                    "Increase 10%",
                    "Increase 15%",
                    "Increase 20%",
                    "Manual Review"
                ]
            
            return {
                'action_id': recommended_action,
                'action_name': [action_meanings[a] for a in recommended_action.cpu().numpy()],
                'confidence': confidence,
                'action_probs': pricing_probs,
                'risk_level': outputs['risk_probs'].argmax(dim=-1)
            }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        stats = super().get_model_statistics()
        
        # Add insurance-specific stats
        stats.update({
            'num_features': self.num_features,
            'forecast_horizons': self.forecast_horizons,
            'num_pricing_actions': self.num_pricing_actions,
            'seasonal_encoding': self.use_seasonal_encoding,
            'insurance_autocorrelation': self.use_insurance_autocorrelation,
            'cycle_detection': self.use_cycle_detection,
            'seasonal_lags': self.seasonal_lags
        })
        
        return stats


class SeasonalEncoder(nn.Module):
    """
    Seasonal time encoding for insurance data.
    
    Encodes monthly cyclical patterns using sine/cosine transformations.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable seasonal components
        self.monthly_weight = nn.Parameter(torch.randn(d_model))
        self.quarterly_weight = nn.Parameter(torch.randn(d_model))
        self.annual_weight = nn.Parameter(torch.randn(d_model))
    
    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Generate seasonal encodings.
        
        Args:
            seq_length: Sequence length
            device: Device to create tensor on
            
        Returns:
            Seasonal encodings [seq_length, d_model]
        """
        months = torch.arange(seq_length, device=device, dtype=torch.float32)
        
        # Monthly cycle (12 months)
        monthly_sin = torch.sin(2 * np.pi * months / 12)
        monthly_cos = torch.cos(2 * np.pi * months / 12)
        
        # Quarterly cycle (3 months)
        quarterly_sin = torch.sin(2 * np.pi * months / 3)
        quarterly_cos = torch.cos(2 * np.pi * months / 3)
        
        # Annual phase
        annual_sin = torch.sin(2 * np.pi * months / 12)
        annual_cos = torch.cos(2 * np.pi * months / 12)
        
        # Combine with learnable weights
        seasonal_encoding = (
            monthly_sin.unsqueeze(-1) * self.monthly_weight * 0.5 +
            monthly_cos.unsqueeze(-1) * self.monthly_weight * 0.5 +
            quarterly_sin.unsqueeze(-1) * self.quarterly_weight * 0.3 +
            quarterly_cos.unsqueeze(-1) * self.quarterly_weight * 0.3 +
            annual_sin.unsqueeze(-1) * self.annual_weight * 0.2 +
            annual_cos.unsqueeze(-1) * self.annual_weight * 0.2
        )
        
        return seasonal_encoding


class InsuranceAutocorrelationModule(nn.Module):
    """
    Insurance-specific autocorrelation intrinsic reward module.
    
    Discovers seasonal patterns in insurance claims with focus on:
    - Monthly patterns (billing cycles)
    - Annual patterns (seasonal weather, holidays)
    - Multi-year patterns (economic cycles)
    """
    
    def __init__(
        self,
        d_model: int,
        max_lag: int = 24,
        seasonal_lags: List[int] = [12, 24]
    ):
        super().__init__()
        self.d_model = d_model
        self.max_lag = max_lag
        self.seasonal_lags = seasonal_lags
        
        # Learnable importance weights for different lags
        self.lag_weights = nn.Parameter(torch.ones(max_lag))
        
        # Seasonal lag importance
        self.seasonal_importance = nn.Parameter(torch.ones(len(seasonal_lags)))
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute autocorrelation reward for insurance patterns.
        
        Args:
            input_sequence: Input features [batch, length, num_features]
            hidden_states: SSM hidden states [batch, length, d_model]
            
        Returns:
            Autocorrelation reward scalar
        """
        batch_size, seq_length, _ = input_sequence.shape
        
        # Extract claims-related features (assuming first feature is claims amount)
        claims_sequence = input_sequence[:, :, 0]  # [batch, length]
        
        # Compute autocorrelation
        rewards = []
        for b in range(batch_size):
            seq = claims_sequence[b]
            
            # Compute autocorrelation function
            seq_normalized = (seq - seq.mean()) / (seq.std() + 1e-8)
            
            autocorr = []
            for lag in range(1, min(self.max_lag + 1, seq_length)):
                if lag < seq_length:
                    corr = torch.corrcoef(torch.stack([
                        seq_normalized[:-lag],
                        seq_normalized[lag:]
                    ]))[0, 1]
                    autocorr.append(corr)
                else:
                    autocorr.append(torch.tensor(0.0, device=seq.device))
            
            autocorr = torch.stack(autocorr)
            
            # Apply learnable lag weights
            lag_weights_normalized = F.softmax(self.lag_weights[:len(autocorr)], dim=0)
            weighted_autocorr = (autocorr * lag_weights_normalized).sum()
            
            # Extra weight for seasonal lags
            seasonal_reward = 0.0
            for i, lag in enumerate(self.seasonal_lags):
                if lag <= len(autocorr):
                    seasonal_reward += autocorr[lag - 1] * self.seasonal_importance[i]
            
            total_reward = weighted_autocorr + seasonal_reward
            rewards.append(total_reward)
        
        return torch.stack(rewards).mean()


class InsuranceCycleDetector(nn.Module):
    """
    FFT-based cycle detection for insurance claims patterns.
    
    Detects and extracts features from:
    - Monthly claim submission patterns
    - Quarterly business cycles
    - Seasonal weather patterns
    - Annual renewal cycles
    """
    
    def __init__(
        self,
        d_model: int,
        sampling_rate: float = 1.0/30.0,
        relevant_cycles: List[float] = [1/12, 1/4, 1/2, 1]
    ):
        super().__init__()
        self.d_model = d_model
        self.sampling_rate = sampling_rate
        self.relevant_cycles = relevant_cycles
        
        # Learnable cycle importance weights
        self.cycle_weights = nn.Parameter(torch.ones(len(relevant_cycles)))
        
        # Feature projection
        self.cycle_projection = nn.Linear(len(relevant_cycles) * 2, d_model)
    
    def forward(
        self,
        input_sequence: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract cycle features from claims sequence.
        
        Args:
            input_sequence: Input features [batch, length, num_features]
            hidden_states: SSM hidden states [batch, length, d_model]
            
        Returns:
            Cycle features [batch, d_model]
        """
        batch_size, seq_length, _ = input_sequence.shape
        
        # Extract claims sequence
        claims_sequence = input_sequence[:, :, 0]  # [batch, length]
        
        # FFT analysis
        fft_claims = fft(claims_sequence, dim=1)
        freqs = fftfreq(seq_length, self.sampling_rate)
        
        # Extract energy at relevant frequencies
        cycle_features = []
        for cycle_freq in self.relevant_cycles:
            # Find closest frequency bin
            freq_idx = torch.argmin(torch.abs(freqs - cycle_freq))
            
            # Extract magnitude and phase
            magnitude = torch.abs(fft_claims[:, freq_idx])
            phase = torch.angle(fft_claims[:, freq_idx])
            
            cycle_features.append(magnitude)
            cycle_features.append(phase)
        
        # Stack and weight
        cycle_features = torch.stack(cycle_features, dim=-1)  # [batch, len(cycles)*2]
        
        # Apply learnable weights (repeat for magnitude and phase)
        weights = torch.repeat_interleave(self.cycle_weights, 2)
        weighted_features = cycle_features * weights.unsqueeze(0)
        
        # Project to model dimension
        projected_features = self.cycle_projection(weighted_features)
        
        return projected_features
