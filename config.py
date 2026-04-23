"""
Configuration file for the Smart Energy Anomaly Detection System
"""

# Model Configuration
MODEL_CONFIG = {
    'isolation_forest': {
        'contamination': 0.05,
        'random_state': 42,
        'n_estimators': 100
    },
    'one_class_svm': {
        'kernel': 'rbf',
        'gamma': 'auto',
        'nu': 0.05
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'handle_missing_values': True,
    'missing_value_method': 'interpolate',  # 'interpolate', 'forward_fill', 'mean'
    'standardize': True,
    'scaler_type': 'standard',  # 'standard', 'minmax'
    'extract_temporal': True,
    'calculate_rolling': True,
    'rolling_windows': [7, 24]  # hours
}

# Feature Configuration
FEATURES = {
    'temporal': ['hour', 'day', 'weekday', 'month', 'day_of_year'],
    'rolling': ['rolling_mean_7', 'rolling_mean_24', 'rolling_std_7', 'rolling_std_24']
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'color_normal': '#1f77b4',
    'color_anomaly': '#ff0000',
    'color_heatmap': 'RdBu',
    'template': 'plotly_white',
    'chart_height': 400,
    'chart_width': 'full'
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Data Configuration
DATA_CONFIG = {
    'sample_size': 2000,
    'datetime_format': '%Y-%m-%d %H:%M:%S',
    'null_threshold': 0.5  # Maximum % of missing values allowed
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'cache_enabled': True,
    'cache_ttl': 3600,  # seconds
    'max_rows_display': 1000
}
