"""
Model training utilities for anomaly detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle


class AnomalyDetectionModel:
    """
    Base class for anomaly detection
    """
    
    def __init__(self, model_name='isolation_forest', contamination=0.05, **kwargs):
        """
        Initialize anomaly detection model
        
        Parameters:
        -----------
        model_name : str
            'isolation_forest' or 'one_class_svm'
        contamination : float
            Expected proportion of anomalies
        **kwargs : dict
            Additional parameters for the model
        """
        self.model_name = model_name
        self.contamination = contamination
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.kwargs = kwargs
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying model"""
        if self.model_name == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                **self.kwargs
            )
        elif self.model_name == 'one_class_svm':
            nu = self.contamination
            self.model = OneClassSVM(
                nu=nu,
                kernel='rbf',
                gamma='auto',
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def train(self, X):
        """
        Train the anomaly detection model
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training data
        
        Returns:
        --------
        self : AnomalyDetectionModel
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # Standardize if not already done
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Data to predict
        
        Returns:
        --------
        np.ndarray
            Predictions: -1 for anomaly, 1 for normal
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_scores(self, X):
        """
        Get anomaly scores
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Data to predict
        
        Returns:
        --------
        np.ndarray
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        
        if self.model_name == 'isolation_forest':
            # Get negative of decision function (negative_outlier_factor)
            scores = -self.model.decision_function(X_scaled)
        else:  # one_class_svm
            # Get decision function
            scores = -self.model.decision_function(X_scaled)
        
        # Normalize scores to 0-1
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        return scores
    
    def fit_predict(self, X):
        """
        Fit model and predict in one step
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Training data
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        self.train(X)
        return self.predict(X)
    
    def set_contamination(self, contamination):
        """
        Update contamination parameter and reinitialize model
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies
        """
        self.contamination = contamination
        self._initialize_model()
        self.is_trained = False
    
    def get_model_info(self):
        """
        Get model information
        
        Returns:
        --------
        dict
            Model information
        """
        return {
            'model_name': self.model_name,
            'contamination': self.contamination,
            'is_trained': self.is_trained,
            'features': self.feature_names if self.feature_names else 'Not set'
        }


def train_model(X, model_name='isolation_forest', contamination=0.05):
    """
    Train an anomaly detection model
    
    Parameters:
    -----------
    X : np.ndarray or pd.DataFrame
        Training data
    model_name : str
        Model type
    contamination : float
        Expected proportion of anomalies
    
    Returns:
    --------
    AnomalyDetectionModel
        Trained model
    """
    model = AnomalyDetectionModel(model_name=model_name, contamination=contamination)
    model.train(X)
    return model


def save_model(model, filepath):
    """
    Save trained model to file
    
    Parameters:
    -----------
    model : AnomalyDetectionModel
        Model to save
    filepath : str
        Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """
    Load model from file
    
    Parameters:
    -----------
    filepath : str
        Path to model file
    
    Returns:
    --------
    AnomalyDetectionModel
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
