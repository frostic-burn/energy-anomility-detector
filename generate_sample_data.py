"""
Generate and save sample energy consumption data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(n_samples=2000, seed=42):
    """
    Generate realistic energy consumption data
    
    Parameters:
    -----------
    n_samples : int
        Number of hourly samples to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Sample energy data
    """
    np.random.seed(seed)
    
    # Create date range (hourly data)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate realistic energy consumption pattern
    base_consumption = 50
    
    # Hourly pattern (peak during day, low at night)
    hourly_pattern = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    
    # Weekly pattern (higher weekdays, lower weekends)
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
    
    # Seasonal pattern
    seasonal_pattern = 15 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365))
    
    # Random noise
    noise = np.random.normal(0, 3, n_samples)
    
    energy = base_consumption + hourly_pattern + weekly_pattern + seasonal_pattern + noise
    
    # Add anomalies (sudden spikes and drops)
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    energy[anomaly_indices] = energy[anomaly_indices] * np.random.uniform(
        0.2, 2.5, len(anomaly_indices)
    )
    
    # Ensure no negative values
    energy = np.maximum(0, energy)
    
    # Generate temperature data (correlated with season)
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                  np.random.normal(0, 2, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': energy,
        'temperature': temperature
    })
    
    return df


def main():
    """Generate and save sample data"""
    print("Generating sample energy data...")
    
    df = generate_sample_data(n_samples=2000)
    
    # Save to data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_file = os.path.join(data_dir, 'sample_energy_data.csv')
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample data generated: {output_file}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Preview:")
    print(df.head(10))
    print(f"\n   Statistics:")
    print(df.describe())


if __name__ == '__main__':
    main()
