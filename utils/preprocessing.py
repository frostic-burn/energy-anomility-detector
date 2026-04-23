"""
Data preprocessing utilities for energy consumption data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta


def handle_missing_values(df, method='interpolate'):
    """
    Handle missing values in the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        'interpolate', 'forward_fill', or 'mean'
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if method == 'interpolate':
        # Only interpolate numeric columns
        df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(method='linear', limit_direction='both')
    elif method == 'forward_fill':
        # Forward fill for all columns, then back fill
        df_copy = df_copy.ffill().bfill()
    elif method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    return df_copy


def standardize_data(df, numeric_cols=None, scaler_type='standard'):
    """
    Standardize/normalize numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numeric_cols : list
        Columns to standardize
    scaler_type : str
        'standard' for StandardScaler, 'minmax' for MinMaxScaler
    
    Returns:
    --------
    pd.DataFrame, object
        Standardized dataframe and the scaler object
    """
    df_copy = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    
    return df_copy, scaler


def extract_temporal_features(df, datetime_col):
    """
    Extract temporal features from datetime column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional temporal features
    """
    df_copy = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_copy[datetime_col]):
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    
    df_copy['hour'] = df_copy[datetime_col].dt.hour
    df_copy['day'] = df_copy[datetime_col].dt.day
    df_copy['weekday'] = df_copy[datetime_col].dt.dayofweek
    df_copy['month'] = df_copy[datetime_col].dt.month
    df_copy['day_of_year'] = df_copy[datetime_col].dt.dayofyear
    
    return df_copy


def calculate_rolling_features(df, target_col, window_sizes=[7, 24]):
    """
    Calculate rolling mean and standard deviation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Column to calculate rolling features for
    window_sizes : list
        Window sizes for rolling calculations
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional rolling features
    """
    df_copy = df.copy()
    
    for window in window_sizes:
        df_copy[f'{target_col}_rolling_mean_{window}'] = df_copy[target_col].rolling(
            window=window, min_periods=1
        ).mean()
        df_copy[f'{target_col}_rolling_std_{window}'] = df_copy[target_col].rolling(
            window=window, min_periods=1
        ).std()
    
    return df_copy.fillna(0)


def preprocess_pipeline(df, datetime_col, target_col, 
                       handle_missing=True, standardize=True,
                       extract_temporal=True, calculate_rolling=True):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Name of datetime column
    target_col : str
        Name of target/energy column
    handle_missing : bool
        Whether to handle missing values
    standardize : bool
        Whether to standardize data
    extract_temporal : bool
        Whether to extract temporal features
    calculate_rolling : bool
        Whether to calculate rolling features
    
    Returns:
    --------
    pd.DataFrame, dict
        Processed dataframe and metadata dictionary
    """
    df_processed = df.copy()
    metadata = {}
    
    # Handle missing values
    if handle_missing:
        df_processed = handle_missing_values(df_processed)
        metadata['missing_handled'] = True
    
    # Extract temporal features
    if extract_temporal:
        df_processed = extract_temporal_features(df_processed, datetime_col)
        metadata['temporal_extracted'] = True
    
    # Calculate rolling features
    if calculate_rolling:
        df_processed = calculate_rolling_features(df_processed, target_col)
        metadata['rolling_calculated'] = True
    
    # Standardize data
    scaler = None
    if standardize:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        # Don't standardize hour, day, weekday as they're categorical-like
        cols_to_scale = [col for col in numeric_cols if col not in ['hour', 'day', 'weekday', 'month', 'day_of_year']]
        if cols_to_scale:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
        metadata['standardized'] = True
        metadata['scaler'] = scaler
    
    return df_processed, metadata


def create_sample_data(n_samples=1000, seed=42):
    """
    Create sample energy consumption data for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed
    
    Returns:
    --------
    pd.DataFrame
        Sample energy data
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')
    
    # Generate realistic energy consumption pattern
    base_consumption = 50
    hourly_pattern = 30 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
    noise = np.random.normal(0, 3, n_samples)
    
    energy = base_consumption + hourly_pattern + weekly_pattern + noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    energy[anomaly_indices] = energy[anomaly_indices] * np.random.uniform(0.2, 2.5, len(anomaly_indices))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': np.maximum(0, energy),
        'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + np.random.normal(0, 2, n_samples)
    })
    
    return df
