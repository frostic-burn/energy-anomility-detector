"""
Prediction and post-processing utilities
"""

import pandas as pd
import numpy as np


def add_predictions_to_dataframe(df, predictions, scores, anomaly_col='anomaly', scores_col='anomaly_score'):
    """
    Add predictions and scores to dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    predictions : np.ndarray
        Predictions (-1 for anomaly, 1 for normal)
    scores : np.ndarray
        Anomaly scores (0-1)
    anomaly_col : str
        Name of anomaly column in result
    scores_col : str
        Name of scores column in result
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions and scores
    """
    df_result = df.copy()
    
    # Convert predictions: -1 -> 1 (anomaly), 1 -> 0 (normal)
    df_result[anomaly_col] = (predictions == -1).astype(int)
    df_result[scores_col] = scores
    
    return df_result


def get_anomaly_statistics(df, anomaly_col='anomaly', target_col='energy_consumption'):
    """
    Get statistics about detected anomalies
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with anomaly labels
    anomaly_col : str
        Name of anomaly column
    target_col : str
        Name of target/energy column
    
    Returns:
    --------
    dict
        Anomaly statistics
    """
    total_samples = len(df)
    num_anomalies = (df[anomaly_col] == 1).sum()
    anomaly_percentage = (num_anomalies / total_samples) * 100
    
    normal_data = df[df[anomaly_col] == 0][target_col]
    anomaly_data = df[df[anomaly_col] == 1][target_col]
    
    stats = {
        'total_samples': total_samples,
        'num_anomalies': num_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'normal_mean': normal_data.mean() if len(normal_data) > 0 else 0,
        'normal_std': normal_data.std() if len(normal_data) > 0 else 0,
        'anomaly_mean': anomaly_data.mean() if len(anomaly_data) > 0 else 0,
        'anomaly_std': anomaly_data.std() if len(anomaly_data) > 0 else 0,
        'total_consumption': df[target_col].sum(),
        'avg_consumption': df[target_col].mean()
    }
    
    return stats


def get_anomaly_details(df, anomaly_col='anomaly', scores_col='anomaly_score', 
                        datetime_col=None, top_n=10):
    """
    Get detailed information about top anomalies
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with anomaly labels and scores
    anomaly_col : str
        Name of anomaly column
    scores_col : str
        Name of scores column
    datetime_col : str, optional
        Name of datetime column
    top_n : int
        Number of top anomalies to return
    
    Returns:
    --------
    pd.DataFrame
        Top anomalies sorted by score
    """
    anomalies_df = df[df[anomaly_col] == 1].copy()
    top_anomalies = anomalies_df.nlargest(top_n, scores_col)
    
    return top_anomalies


def get_anomaly_explanation(df, idx, anomaly_col='anomaly', scores_col='anomaly_score',
                            target_col='energy_consumption'):
    """
    Get explanation for why a point is flagged as anomaly
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with data
    idx : int
        Index of the point
    anomaly_col : str
        Name of anomaly column
    scores_col : str
        Name of scores column
    target_col : str
        Name of target column
    
    Returns:
    --------
    dict
        Explanation details
    """
    # Support both positional and label-based indices from UI selections.
    if idx in df.index:
        row = df.loc[idx]
    else:
        row = df.iloc[idx]
    is_anomaly = row[anomaly_col] == 1
    score = row[scores_col]
    value = row[target_col]
    
    # Calculate deviation from mean
    mean_value = df[target_col].mean()
    std_value = df[target_col].std()
    z_score = (value - mean_value) / (std_value + 1e-10)
    
    # Calculate deviation from rolling mean if available
    rolling_mean_col = None
    for col in df.columns:
        if 'rolling_mean' in col:
            rolling_mean_col = col
            break
    
    explanation = {
        'is_anomaly': is_anomaly,
        'anomaly_score': score,
        'value': value,
        'z_score': z_score,
        'mean_value': mean_value,
        'std_value': std_value,
        'deviation_from_mean': value - mean_value,
        'deviation_percent': ((value - mean_value) / (mean_value + 1e-10)) * 100
    }
    
    if rolling_mean_col is not None:
        rolling_mean = row[rolling_mean_col]
        explanation['rolling_mean'] = rolling_mean
        explanation['deviation_from_rolling_mean'] = value - rolling_mean
    
    return explanation


def filter_anomalies_by_threshold(df, scores_col='anomaly_score', threshold=0.5,
                                  anomaly_col='anomaly'):
    """
    Filter anomalies by custom threshold
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with anomaly scores
    scores_col : str
        Name of scores column
    threshold : float
        Custom threshold (0-1)
    anomaly_col : str
        Name of anomaly column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with updated anomaly labels
    """
    df_filtered = df.copy()
    df_filtered[anomaly_col] = (df_filtered[scores_col] > threshold).astype(int)
    
    return df_filtered


def get_temporal_anomaly_distribution(df, datetime_col, anomaly_col='anomaly'):
    """
    Get distribution of anomalies over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with anomalies
    datetime_col : str
        Name of datetime column
    anomaly_col : str
        Name of anomaly column
    
    Returns:
    --------
    pd.DataFrame
        Anomaly counts by date
    """
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df_copy = df.copy()
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    else:
        df_copy = df.copy()
    
    daily_anomalies = df_copy.groupby(df_copy[datetime_col].dt.date)[anomaly_col].sum()
    
    return daily_anomalies


def get_hourly_anomaly_distribution(df, datetime_col, anomaly_col='anomaly'):
    """
    Get distribution of anomalies by hour
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with anomalies
    datetime_col : str
        Name of datetime column
    anomaly_col : str
        Name of anomaly column
    
    Returns:
    --------
    pd.Series
        Anomaly counts by hour
    """
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df_copy = df.copy()
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    else:
        df_copy = df.copy()
    
    hourly_anomalies = df_copy.groupby(df_copy[datetime_col].dt.hour)[anomaly_col].sum()
    
    return hourly_anomalies
