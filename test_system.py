"""
Test script to verify the Smart Energy Anomaly Detection System works correctly
Run this before deploying to ensure all components are functional
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    try:
        from model.train import AnomalyDetectionModel, train_model
        from model.predict import add_predictions_to_dataframe, get_anomaly_statistics
        from utils.preprocessing import preprocess_pipeline, create_sample_data
        from utils.visualization import plot_timeseries_anomalies
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_sample_data():
    """Test sample data generation"""
    print("🧪 Testing sample data generation...")
    try:
        from utils.preprocessing import create_sample_data
        df = create_sample_data(n_samples=100)
        
        assert len(df) == 100, "Wrong number of samples"
        assert 'timestamp' in df.columns, "Missing timestamp column"
        assert 'energy_consumption' in df.columns, "Missing energy_consumption column"
        
        print(f"   ✅ Generated {len(df)} samples with columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("🧪 Testing preprocessing...")
    try:
        from utils.preprocessing import create_sample_data, preprocess_pipeline
        
        df = create_sample_data(n_samples=100)
        df_processed, metadata = preprocess_pipeline(
            df,
            datetime_col='timestamp',
            target_col='energy_consumption'
        )
        
        assert len(df_processed) > 0, "Preprocessing failed"
        assert 'hour' in df_processed.columns, "Temporal features not extracted"
        print(f"   ✅ Preprocessing successful, {len(df_processed.columns)} features created")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_model_training():
    """Test model training"""
    print("🧪 Testing model training...")
    try:
        from model.train import train_model
        from utils.preprocessing import create_sample_data, preprocess_pipeline
        
        df = create_sample_data(n_samples=100)
        df_processed, _ = preprocess_pipeline(df, 'timestamp', 'energy_consumption')
        
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[feature_cols].values
        
        # Test Isolation Forest
        model_if = train_model(X, model_name='isolation_forest', contamination=0.1)
        assert model_if.is_trained, "Isolation Forest not trained"
        print("   ✅ Isolation Forest trained successfully")
        
        # Test One-Class SVM
        model_svm = train_model(X, model_name='one_class_svm', contamination=0.1)
        assert model_svm.is_trained, "One-Class SVM not trained"
        print("   ✅ One-Class SVM trained successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_predictions():
    """Test predictions and scores"""
    print("🧪 Testing predictions...")
    try:
        from model.train import train_model
        from model.predict import add_predictions_to_dataframe, get_anomaly_statistics
        from utils.preprocessing import create_sample_data, preprocess_pipeline
        
        df = create_sample_data(n_samples=100)
        df_processed, _ = preprocess_pipeline(df, 'timestamp', 'energy_consumption')
        
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[feature_cols].values
        
        model = train_model(X, contamination=0.1)
        
        predictions = model.predict(X)
        scores = model.predict_scores(X)
        
        assert len(predictions) == len(X), "Wrong prediction shape"
        assert len(scores) == len(X), "Wrong scores shape"
        
        # Add to dataframe
        df_with_pred = add_predictions_to_dataframe(df, predictions, scores)
        assert 'anomaly' in df_with_pred.columns, "Anomaly column not added"
        assert 'anomaly_score' in df_with_pred.columns, "Anomaly score column not added"
        
        # Get statistics
        stats = get_anomaly_statistics(df_with_pred)
        assert stats['num_anomalies'] >= 0, "Invalid anomaly count"
        
        print(f"   ✅ Predictions successful")
        print(f"      - Total samples: {stats['total_samples']}")
        print(f"      - Anomalies detected: {stats['num_anomalies']}")
        print(f"      - Anomaly %: {stats['anomaly_percentage']:.1f}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_visualizations():
    """Test visualization functions"""
    print("🧪 Testing visualizations...")
    try:
        from model.train import train_model
        from model.predict import add_predictions_to_dataframe
        from utils.preprocessing import create_sample_data, preprocess_pipeline
        from utils.visualization import (
            plot_timeseries_anomalies, plot_histogram,
            plot_boxplot, plot_anomaly_scores
        )
        
        df = create_sample_data(n_samples=100)
        df_processed, _ = preprocess_pipeline(df, 'timestamp', 'energy_consumption')
        
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[feature_cols].values
        
        model = train_model(X, contamination=0.1)
        predictions = model.predict(X)
        scores = model.predict_scores(X)
        
        df_with_pred = add_predictions_to_dataframe(df, predictions, scores)
        
        # Test each visualization
        fig1 = plot_timeseries_anomalies(df_with_pred, 'timestamp', 'energy_consumption', 'anomaly')
        assert fig1 is not None, "Time series plot failed"
        
        fig2 = plot_histogram(df_with_pred, 'energy_consumption')
        assert fig2 is not None, "Histogram plot failed"
        
        fig3 = plot_boxplot(df_with_pred, 'energy_consumption')
        assert fig3 is not None, "Box plot failed"
        
        fig4 = plot_anomaly_scores(df_with_pred, 'anomaly_score', threshold=0.5)
        assert fig4 is not None, "Anomaly scores plot failed"
        
        print("   ✅ All visualizations working")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_end_to_end():
    """Complete end-to-end test"""
    print("🧪 Running end-to-end test...")
    try:
        from model.train import train_model
        from model.predict import add_predictions_to_dataframe, get_anomaly_statistics, get_anomaly_details
        from utils.preprocessing import create_sample_data, preprocess_pipeline
        
        # Step 1: Generate data
        df = create_sample_data(n_samples=200)
        
        # Step 2: Preprocess
        df_processed, metadata = preprocess_pipeline(df, 'timestamp', 'energy_consumption')
        
        # Step 3: Prepare features
        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[feature_cols].values
        
        # Step 4: Train model
        model = train_model(X, model_name='isolation_forest', contamination=0.05)
        
        # Step 5: Make predictions
        predictions = model.predict(X)
        scores = model.predict_scores(X)
        
        # Step 6: Add to dataframe
        df_final = add_predictions_to_dataframe(df, predictions, scores)
        
        # Step 7: Get statistics
        stats = get_anomaly_statistics(df_final)
        
        # Step 8: Get top anomalies
        if stats['num_anomalies'] > 0:
            top_anomalies = get_anomaly_details(df_final, top_n=5)
            assert len(top_anomalies) > 0, "No top anomalies found"
        
        print("   ✅ End-to-end test successful")
        print(f"      Data: {len(df)} → {len(df_final)} samples")
        print(f"      Features engineered: {len(feature_cols)}")
        print(f"      Model: Isolation Forest")
        print(f"      Anomalies detected: {stats['num_anomalies']} ({stats['anomaly_percentage']:.1f}%)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 SMART ENERGY ANOMALY DETECTION - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Sample Data", test_sample_data),
        ("Preprocessing", test_preprocessing),
        ("Model Training", test_model_training),
        ("Predictions", test_predictions),
        ("Visualizations", test_visualizations),
        ("End-to-End", test_end_to_end)
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
        print()
    
    # Summary
    print("="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {name}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to deploy.\n")
        return True
    else:
        print("⚠️  Some tests failed. Please review the errors above.\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
