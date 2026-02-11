"""
Data Generation for Empirical Validation
Generates synthetic insurance data with known properties for controlled experiments
Author: Nadhir Hassen (nadhir.hassen@mila.quebec)
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import json

np.random.seed(42)


def generate_insurance_data(n_policies=1000, n_months=100, output_path=None):
    """
    Generate synthetic insurance data with embedded patterns for validation
    
    Properties:
    - 72-month insurance cycle (hard/soft market phases)
    - 12-month seasonal pattern
    - AR(2) temporal autocorrelation
    - Heteroscedastic noise
    - Multiple risk levels
    """
    print(f"Generating {n_policies} policies × {n_months} months...")
    
    data = []
    
    for policy_id in range(n_policies):
        # Policy characteristics
        age = np.random.randint(25, 75)
        risk_level = np.random.choice([0, 1, 2], p=[0.45, 0.38, 0.17])
        location_risk = np.random.uniform(0.5, 1.5)
        base_premium = np.random.uniform(500, 2000)
        
        # Initialize AR process
        ar_state = np.zeros(2)
        
        for month in range(n_months):
            # Time features
            t = month / n_months
            month_of_year = month % 12
            quarter = month % 4
            
            # 72-month insurance cycle (hard/soft market)
            cycle_component = 0.3 * np.sin(2 * np.pi * month / 72)
            
            # 12-month seasonal pattern
            seasonal_component = 0.2 * np.sin(2 * np.pi * month / 12)
            
            # AR(2) autocorrelation
            ar_component = 0.7 * ar_state[0] - 0.2 * ar_state[1] + np.random.normal(0, 0.1)
            ar_state[1] = ar_state[0]
            ar_state[0] = ar_component
            
            # Risk-based component
            risk_component = (risk_level * 0.3)
            
            # Economic indicators (trending)
            unemployment_rate = 4.5 + 0.5 * np.sin(2 * np.pi * month / 60) + np.random.normal(0, 0.1)
            gdp_growth = 2.5 + 0.3 * np.cos(2 * np.pi * month / 48) + np.random.normal(0, 0.1)
            
            # Claims amount (normalized to [0, 1] range)
            base_claims = (
                0.5 +  # baseline
                cycle_component +
                seasonal_component +
                ar_component +
                risk_component +
                location_risk * 0.1 +
                np.random.normal(0, 0.1 * (1 + risk_level * 0.5))  # heteroscedastic noise
            )
            claims_amount = np.clip(base_claims, 0, 2) * 1000  # scale to dollars
            
            # Claims frequency (Poisson process)
            lambda_claims = np.exp(
                -2.5 +
                risk_level * 0.5 +
                cycle_component +
                seasonal_component
            )
            claims_count = np.random.poisson(lambda_claims)
            
            # Premium adjustment
            premium = base_premium * (1 + risk_level * 0.2 + location_risk * 0.1)
            
            data.append({
                'policy_id': policy_id,
                'month': month,
                'date': pd.Timestamp('2015-01-01') + pd.DateOffset(months=month),
                'claims_amount': round(claims_amount, 2),
                'claims_count': int(claims_count),
                'risk_level': int(risk_level),
                'age': int(age),
                'premium': round(premium, 2),
                'location_risk': round(location_risk, 3),
                'month_of_year': int(month_of_year),
                'quarter': int(quarter),
                'unemployment_rate': round(unemployment_rate, 3),
                'gdp_growth': round(gdp_growth, 3),
                'cycle_phase': round(cycle_component, 3),
                'seasonal_phase': round(seasonal_component, 3)
            })
    
    df = pd.DataFrame(data)
    
    # Calculate statistics
    stats = {
        'n_policies': n_policies,
        'n_months': n_months,
        'total_observations': len(df),
        'claims_amount': {
            'mean': float(df['claims_amount'].mean()),
            'std': float(df['claims_amount'].std()),
            'min': float(df['claims_amount'].min()),
            'max': float(df['claims_amount'].max())
        },
        'claims_count': {
            'mean': float(df['claims_count'].mean()),
            'zeros': int((df['claims_count'] == 0).sum()),
            'max': int(df['claims_count'].max())
        },
        'risk_distribution': {
            'low': int((df['risk_level'] == 0).sum()),
            'medium': int((df['risk_level'] == 1).sum()),
            'high': int((df['risk_level'] == 2).sum())
        }
    }
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        df.to_csv(output_path, index=False)
        print(f"✓ Data saved to {output_path}")
        
        # Save statistics
        stats_path = output_path.parent / 'data_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to {stats_path}")
    
    print("\nData Generation Summary:")
    print(f"  Total observations: {stats['total_observations']:,}")
    print(f"  Claims amount mean: ${stats['claims_amount']['mean']:.2f}")
    print(f"  Claims frequency mean: {stats['claims_count']['mean']:.3f}")
    print(f"  Zero claims: {100 * stats['claims_count']['zeros'] / stats['total_observations']:.1f}%")
    print(f"  Risk distribution: Low={stats['risk_distribution']['low']}, Med={stats['risk_distribution']['medium']}, High={stats['risk_distribution']['high']}")
    
    return df, stats


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic insurance data')
    parser.add_argument('--n_policies', type=int, default=1000, help='Number of policies')
    parser.add_argument('--n_months', type=int, default=100, help='Number of months per policy')
    parser.add_argument('--output', type=str, default='results/insurance_data.csv', help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    df, stats = generate_insurance_data(
        n_policies=args.n_policies,
        n_months=args.n_months,
        output_path=args.output
    )
    
    print("\n✓ Data generation complete!")


if __name__ == '__main__':
    main()
