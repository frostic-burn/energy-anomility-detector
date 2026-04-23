"""
Smart Energy Anomaly Detection System - Streamlit Application
A production-quality web interface for anomaly detection in energy consumption data
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go

# Import custom modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.train import AnomalyDetectionModel, train_model
from model.predict import (
    add_predictions_to_dataframe, get_anomaly_statistics, 
    get_anomaly_details, get_anomaly_explanation, 
    filter_anomalies_by_threshold, get_temporal_anomaly_distribution,
    get_hourly_anomaly_distribution
)
from utils.preprocessing import (
    preprocess_pipeline, create_sample_data, 
    extract_temporal_features, calculate_rolling_features
)
from utils.visualization import (
    plot_timeseries_anomalies, plot_histogram, plot_boxplot,
    plot_heatmap, plot_correlation_matrix, plot_anomaly_scores,
    create_dashboard_summary
)


# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Energy Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===========================
# CUSTOM STYLES (defined early so it can be called after session state init)
# ===========================
def inject_custom_styles():
    """Apply a distinct, modern visual style to the app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

        :root {
            --ink: #0f172a;
            --muted: #475569;
            --accent: #d97706;
            --accent-2: #0f766e;
            --card: rgba(255, 255, 255, 0.80);
            --line: rgba(15, 23, 42, 0.12);
            --sidebar-start: #f1f5f9;
            --sidebar-end: #fffdf7;
            --app-grad-1: rgba(13, 148, 136, 0.13);
            --app-grad-2: rgba(217, 119, 6, 0.13);
            --app-bg-start: #fffdf7;
            --app-bg-end: #f8fafc;
            --hero-overlay: rgba(255, 255, 255, 0.7);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --ink: #e2e8f0;
                --muted: #cbd5e1;
                --card: rgba(15, 23, 42, 0.72);
                --line: rgba(148, 163, 184, 0.25);
                --sidebar-start: #0b1222;
                --sidebar-end: #111827;
                --app-grad-1: rgba(20, 184, 166, 0.18);
                --app-grad-2: rgba(245, 158, 11, 0.14);
                --app-bg-start: #020617;
                --app-bg-end: #0f172a;
                --hero-overlay: rgba(2, 6, 23, 0.48);
            }
        }

        .stApp {
            background:
                radial-gradient(1200px 650px at 100% -5%, var(--app-grad-1), transparent 60%),
                radial-gradient(1000px 600px at -15% 20%, var(--app-grad-2), transparent 55%),
                linear-gradient(180deg, var(--app-bg-start) 0%, var(--app-bg-end) 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        h1, h2, h3 {
            letter-spacing: -0.015em;
            color: var(--ink);
        }

        p, span, label, div, li {
            color: inherit;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stText"] {
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-start) 0%, var(--sidebar-end) 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(8px);
        }

        .hero {
            padding: 1.2rem 1.2rem 1rem;
            border-radius: 20px;
            border: 1px solid rgba(15, 23, 42, 0.10);
            background:
                linear-gradient(120deg, rgba(15, 118, 110, 0.12), rgba(217, 119, 6, 0.15)),
                var(--hero-overlay);
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
            animation: fadeIn 550ms ease-in;
        }

        .hero h1 {
            font-size: clamp(1.4rem, 1.3rem + 1vw, 2.2rem);
            margin: 0;
            font-weight: 700;
        }

        .hero p {
            margin: 0.4rem 0 0;
            color: var(--muted);
        }

        .insight-chip {
            display: inline-block;
            margin: 0.35rem 0.3rem 0 0;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.12);
            background: rgba(255, 255, 255, 0.72);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem;
        }

        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] * {
            color: var(--ink) !important;
        }

        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"] * {
            color: var(--ink) !important;
        }

        .stAlert p,
        .stAlert div {
            color: var(--ink) !important;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .hero {
                padding: 1rem;
                border-radius: 14px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===========================
# SESSION STATE INITIALIZATION
# ===========================
def initialize_session_state():
    """Initialize all session state variables"""
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    if 'scores' not in st.session_state:
        st.session_state.scores = None
    
    if 'df_with_predictions' not in st.session_state:
        st.session_state.df_with_predictions = None
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = True
    
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Dashboard"
    
    if 'datetime_col' not in st.session_state:
        st.session_state.datetime_col = None
    
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    if 'anomaly_threshold' not in st.session_state:
        st.session_state.anomaly_threshold = 0.5


initialize_session_state()
inject_custom_styles()


# ===========================
# CACHED FUNCTIONS
# ===========================
@st.cache_data
def load_sample_data():
    """Load sample energy data"""
    return create_sample_data(n_samples=2000)


@st.cache_resource
def preprocess_data_cached(df_hash, datetime_col, target_col):
    """Cache preprocessed data"""
    # This is a placeholder - actual preprocessing happens in sidebar
    return None


# inject_custom_styles is defined above (before session state init)


def get_probable_causes(explanation, timestamp_value=None):
    """Generate actionable, human-readable probable causes for an anomaly."""
    causes = []

    z_score = abs(float(explanation.get('z_score', 0)))
    if z_score >= 3:
        causes.append("Sudden extreme spike/drop relative to baseline (|z| >= 3).")
    elif z_score >= 2:
        causes.append("Notable deviation from normal operating range (|z| >= 2).")

    deviation_pct = abs(float(explanation.get('deviation_percent', 0)))
    if deviation_pct >= 40:
        causes.append("Large percentage shift from average demand suggests unusual appliance behavior.")

    rolling_mean = explanation.get('rolling_mean')
    if rolling_mean is not None and float(rolling_mean) != 0:
        roll_delta = abs(float(explanation.get('value', 0)) - float(rolling_mean)) / (abs(float(rolling_mean)) + 1e-10)
        if roll_delta >= 0.35:
            causes.append("Sharp break from short-term trend; possible transient load or outage-recovery event.")

    if timestamp_value is not None:
        ts = pd.to_datetime(timestamp_value, errors='coerce')
        if pd.notna(ts):
            if ts.hour in [0, 1, 2, 3, 4, 5]:
                causes.append("Off-hours anomaly; check unattended equipment, timers, or overnight HVAC behavior.")
            if ts.weekday() >= 5:
                causes.append("Weekend pattern mismatch; verify occupancy schedule or holiday mode settings.")

    if not causes:
        causes.append("Multi-feature pattern differs from learned baseline; inspect weather, occupancy, and recent device changes.")

    return causes[:4]


# ===========================
# SIDEBAR CONFIGURATION
# ===========================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Data Upload Section
    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Upload CSV", "Use Sample Data"],
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type="csv",
            help="CSV must have timestamp and energy consumption columns"
        )
    
    # Load data based on selection
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.df = None
    else:
        st.session_state.df = load_sample_data()
        st.info("📊 Using sample data (2000 hours of energy consumption)")
    
    # Column Selection
    if st.session_state.df is not None:
        st.markdown("### 📋 Column Mapping")
        
        cols = st.session_state.df.columns.tolist()
        
        st.session_state.datetime_col = st.selectbox(
            "Select timestamp column:",
            cols,
            index=0 if len(cols) > 0 else None,
            key="datetime_select"
        )
        
        st.session_state.target_col = st.selectbox(
            "Select energy consumption column:",
            cols,
            index=1 if len(cols) > 1 else 0,
            key="target_select"
        )
    
    # Model Configuration
    st.markdown("### 🤖 Model Settings")
    
    model_type = st.selectbox(
        "Select model:",
        ["Isolation Forest", "One-Class SVM"],
        help="Choose anomaly detection algorithm"
    )
    
    contamination = st.slider(
        "Contamination ratio:",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalies in data"
    )
    
    # Threshold Configuration
    st.markdown("### 🎯 Anomaly Threshold")
    st.session_state.anomaly_threshold = st.slider(
        "Score threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Anomaly score threshold (0-1)"
    )
    
    # Feature Selection
    st.markdown("### ✨ Feature Options")
    extract_temporal = st.checkbox(
        "Extract temporal features",
        value=True,
        help="Add hour, day, weekday features"
    )
    calculate_rolling = st.checkbox(
        "Calculate rolling features",
        value=True,
        help="Add rolling mean and std"
    )
    
    # Date Range Filter
    st.markdown("### 📅 Date Range Filter")
    if st.session_state.df is not None and st.session_state.datetime_col:
        try:
            df_datetime = pd.to_datetime(st.session_state.df[st.session_state.datetime_col])
            min_date = df_datetime.min().date()
            max_date = df_datetime.max().date()
            
            date_range = st.date_input(
                "Select date range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        except:
            date_range = None
    else:
        date_range = None
    
    # Train Model Button
    st.markdown("### 🚀 Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Train Model", use_container_width=True):
            if st.session_state.df is None:
                st.error("Please load data first!")
            elif st.session_state.datetime_col is None or st.session_state.target_col is None:
                st.error("Please select datetime and target columns!")
            else:
                with st.spinner("Training model..."):
                    try:
                        # Preprocess data
                        df_processed, metadata = preprocess_pipeline(
                            st.session_state.df,
                            st.session_state.datetime_col,
                            st.session_state.target_col,
                            handle_missing=True,
                            standardize=True,
                            extract_temporal=extract_temporal,
                            calculate_rolling=calculate_rolling
                        )
                        
                        st.session_state.df_processed = df_processed
                        
                        # Select features for training
                        feature_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                        X_train = df_processed[feature_cols].values
                        
                        # Train model
                        model_name = 'isolation_forest' if model_type == 'Isolation Forest' else 'one_class_svm'
                        st.session_state.model = train_model(X_train, model_name=model_name, contamination=contamination)
                        
                        # Get predictions
                        predictions = st.session_state.model.predict(X_train)
                        scores = st.session_state.model.predict_scores(X_train)
                        
                        st.session_state.predictions = predictions
                        st.session_state.scores = scores
                        
                        # Add to dataframe
                        st.session_state.df_with_predictions = add_predictions_to_dataframe(
                            st.session_state.df,
                            predictions,
                            scores
                        )
                        
                        st.session_state.model_trained = True
                        st.success(f"✅ Model trained successfully with {model_type}!")
                    
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


# ===========================
# MAIN CONTENT AREA
# ===========================
st.markdown(
    """
    <div class="hero">
        <h1>Smart Energy Consumption Anomaly Detection</h1>
        <p>Monitor daily patterns, surface anomalies early, and explain likely causes with minimal false alarms.</p>
        <span class="insight-chip">Timeline Intelligence</span>
        <span class="insight-chip">Household-Aware Profiles</span>
        <span class="insight-chip">Actionable Root-Cause Hints</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.df is None:
    st.warning("📊 Please load data using the sidebar to get started!")
else:
    # KPI Cards at the top
    if st.session_state.model_trained and st.session_state.df_with_predictions is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        stats = get_anomaly_statistics(
            st.session_state.df_with_predictions,
            anomaly_col='anomaly',
            target_col=st.session_state.target_col
        )
        
        with col1:
            st.metric(
                "Total Consumption",
                f"{stats['total_consumption']:.0f} kWh",
                delta=f"Avg: {stats['avg_consumption']:.2f} kWh"
            )
        
        with col2:
            st.metric(
                "Anomalies Detected",
                int(stats['num_anomalies']),
                delta=f"{stats['anomaly_percentage']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Average Usage",
                f"{stats['avg_consumption']:.2f} kWh",
                delta=f"±{stats['normal_std']:.2f}"
            )
        
        with col4:
            st.metric(
                "Data Points",
                f"{stats['total_samples']:,}",
                delta=f"Normal: {stats['total_samples'] - int(stats['num_anomalies'])}"
            )

        st.markdown("### Operational Snapshot")
        profile_df = st.session_state.df_with_predictions.copy()
        profile_df[st.session_state.datetime_col] = pd.to_datetime(
            profile_df[st.session_state.datetime_col], errors='coerce'
        )
        anomaly_rows = profile_df[profile_df['anomaly'] == 1]
        if len(anomaly_rows) > 0 and profile_df[st.session_state.datetime_col].notna().any():
            peak_hour = int(anomaly_rows[st.session_state.datetime_col].dt.hour.value_counts().idxmax())
            weekend_share = (
                (anomaly_rows[st.session_state.datetime_col].dt.weekday >= 5).mean() * 100
            )
            severe_share = (anomaly_rows['anomaly_score'] >= 0.8).mean() * 100
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Peak Anomaly Hour", f"{peak_hour:02d}:00")
            with s2:
                st.metric("Weekend Anomalies", f"{weekend_share:.1f}%")
            with s3:
                st.metric("Severe Score Share", f"{severe_share:.1f}%")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "📁 Data Explorer", "📈 Visualization", "⚡ Live Simulation", "📤 Results"]
    )
    
    # ===========================
    # TAB 1: DASHBOARD
    # ===========================
    with tab1:
        st.markdown("## Dashboard Overview")
        
        if st.session_state.model_trained and st.session_state.df_with_predictions is not None:
            # Filter data by date range if specified
            df_display = st.session_state.df_with_predictions.copy()
            
            if date_range and len(date_range) == 2:
                try:
                    df_datetime = pd.to_datetime(df_display[st.session_state.datetime_col])
                    mask = (df_datetime.dt.date >= date_range[0]) & (df_datetime.dt.date <= date_range[1])
                    df_display = df_display[mask]
                except:
                    pass
            
            # Time Series Chart
            st.markdown("### 📈 Energy Consumption with Anomalies")
            try:
                fig_ts = plot_timeseries_anomalies(
                    df_display,
                    st.session_state.datetime_col,
                    st.session_state.target_col,
                    'anomaly',
                    title="Energy Consumption Over Time"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting time series: {e}")
            
            # Anomaly Statistics
            st.markdown("### 📊 Anomaly Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    daily_anomalies = get_temporal_anomaly_distribution(
                        df_display,
                        st.session_state.datetime_col,
                        'anomaly'
                    )
                    
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Bar(
                        x=daily_anomalies.index,
                        y=daily_anomalies.values,
                        marker=dict(color='#ef553b'),
                        name='Anomalies'
                    ))
                    fig_daily.update_layout(
                        title="Anomalies per Day",
                        xaxis_title="Date",
                        yaxis_title="Count",
                        height=400,
                        showlegend=False,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot daily anomalies: {e}")
            
            with col2:
                try:
                    hourly_anomalies = get_hourly_anomaly_distribution(
                        df_display,
                        st.session_state.datetime_col,
                        'anomaly'
                    )
                    
                    fig_hourly = go.Figure()
                    fig_hourly.add_trace(go.Bar(
                        x=hourly_anomalies.index,
                        y=hourly_anomalies.values,
                        marker=dict(color='#636EFA'),
                        name='Anomalies'
                    ))
                    fig_hourly.update_layout(
                        title="Anomalies by Hour",
                        xaxis_title="Hour of Day",
                        yaxis_title="Count",
                        height=400,
                        showlegend=False,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot hourly anomalies: {e}")
            
            # Alert Box
            num_anomalies = (df_display['anomaly'] == 1).sum()
            if num_anomalies > 0:
                st.warning(
                    f"⚠️ **{num_anomalies} anomalies detected** in the selected period. "
                    f"Check the Results tab for detailed analysis."
                )
            else:
                st.success("✅ No anomalies detected in the selected period.")
        
        else:
            st.info("👈 Please train the model first using the sidebar.")
    
    # ===========================
    # TAB 2: DATA EXPLORER
    # ===========================
    with tab2:
        st.markdown("## Data Explorer")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📋 Dataset Preview")
        
        with col2:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        # Display dataframe
        if st.session_state.df_with_predictions is not None:
            df_display = st.session_state.df_with_predictions.copy()
        else:
            df_display = st.session_state.df.copy()
        
        # Check if dataframe is empty
        if len(df_display) == 0:
            st.warning("No data available. Please check your filters or upload data.")
        else:
            # Filtering options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rows_to_show = st.slider(
                    "Rows to display:",
                    min_value=10,
                    max_value=len(df_display),
                    value=min(50, len(df_display)),
                    step=10
                )
            
            with col2:
                if 'anomaly' in df_display.columns:
                    anomaly_filter = st.multiselect(
                        "Filter by anomaly:",
                        [0, 1],
                        default=[0, 1],
                        format_func=lambda x: "Normal" if x == 0 else "Anomaly"
                    )
                    df_display = df_display[df_display['anomaly'].isin(anomaly_filter)]
            
            # Check again after anomaly filter
            if len(df_display) == 0:
                st.warning("No data matches the selected anomaly filter.")
            else:
                with col3:
                    sort_col = st.selectbox(
                        "Sort by:",
                        df_display.columns.tolist(),
                        index=0
                    )
                
                df_display = df_display.sort_values(by=sort_col).head(rows_to_show)
                
                # Display with styling
                st.dataframe(df_display, use_container_width=True)
                
                # Summary Statistics
                if show_stats:
                    st.markdown("### 📊 Summary Statistics")
                    
                    numeric_cols = df_display.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        stats_df = df_display[numeric_cols].describe().T
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        st.info("No numeric columns available for statistics.")
                
                # Download button
                st.markdown("### 💾 Download Data")
                
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="energy_data_with_predictions.csv",
                    mime="text/csv"
                )
    
    # ===========================
    # TAB 3: VISUALIZATION
    # ===========================
    with tab3:
        st.markdown("## Advanced Visualizations")
        
        if st.session_state.model_trained and st.session_state.df_with_predictions is not None:
            df_display = st.session_state.df_with_predictions.copy()
            
            # Visualization selection
            viz_type = st.selectbox(
                "Select visualization:",
                [
                    "Histogram",
                    "Box Plot",
                    "Energy by Hour (Heatmap)",
                    "Energy by Weekday (Heatmap)",
                    "Anomaly Scores Distribution",
                    "Correlation Matrix"
                ]
            )
            
            if viz_type == "Histogram":
                st.markdown("### Distribution of Energy Consumption")
                try:
                    fig = plot_histogram(
                        df_display,
                        st.session_state.target_col,
                        title="Energy Consumption Distribution",
                        nbins=40
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            elif viz_type == "Box Plot":
                st.markdown("### Box Plot Analysis")
                try:
                    fig = plot_boxplot(
                        df_display,
                        st.session_state.target_col,
                        title="Energy Consumption Box Plot"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            elif viz_type == "Energy by Hour (Heatmap)":
                st.markdown("### Hourly Energy Consumption Heatmap")
                try:
                    if 'hour' in df_display.columns and 'day' in df_display.columns:
                        fig = plot_heatmap(
                            df_display,
                            x_col='hour',
                            y_col='day',
                            z_col=st.session_state.target_col,
                            title="Average Energy by Hour and Day"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Temporal features not extracted. Please enable in sidebar.")
                except Exception as e:
                    st.error(f"Error: {e}")
            
            elif viz_type == "Energy by Weekday (Heatmap)":
                st.markdown("### Weekday Energy Consumption Heatmap")
                try:
                    if 'weekday' in df_display.columns and 'hour' in df_display.columns:
                        fig = plot_heatmap(
                            df_display,
                            x_col='hour',
                            y_col='weekday',
                            z_col=st.session_state.target_col,
                            title="Average Energy by Weekday and Hour"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Temporal features not extracted. Please enable in sidebar.")
                except Exception as e:
                    st.error(f"Error: {e}")
            
            elif viz_type == "Anomaly Scores Distribution":
                st.markdown("### Anomaly Scores Distribution")
                try:
                    fig = plot_anomaly_scores(
                        df_display,
                        'anomaly_score',
                        threshold=st.session_state.anomaly_threshold,
                        title="Distribution of Anomaly Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            
            elif viz_type == "Correlation Matrix":
                st.markdown("### Feature Correlation Matrix")
                try:
                    numeric_df = df_display.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        fig = plot_correlation_matrix(numeric_df)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough numeric features for correlation matrix.")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:
            st.info("👈 Please train the model first using the sidebar.")
    
    # ===========================
    # TAB 4: LIVE SIMULATION
    # ===========================
    with tab4:
        st.markdown("## Live Streaming Simulation")
        st.markdown(
            "Simulate real-time energy data and detect anomalies continuously"
        )
        
        if st.session_state.model_trained:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_samples = st.number_input(
                    "Samples to generate:",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
            
            with col2:
                update_frequency = st.slider(
                    "Update interval (seconds):",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                )
            
            with col3:
                anomaly_injection = st.slider(
                    "Anomaly injection rate:",
                    min_value=0.0,
                    max_value=0.3,
                    value=0.05,
                    step=0.01
                )
            
            # Simulation controls
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start Simulation", use_container_width=True):
                    st.session_state.simulation_running = True
            
            with col2:
                if st.button("⏹️ Stop Simulation", use_container_width=True):
                    st.session_state.simulation_running = False
            
            if st.session_state.simulation_running:
                st.markdown("### 📊 Live Data Stream")
                
                # Placeholder for live chart
                chart_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                # Generate streaming data
                try:
                    # Get last features from training data for context
                    last_data = st.session_state.df_processed.iloc[-1:]
                    feature_cols = st.session_state.df_processed.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    
                    # Initialize streaming data
                    stream_data = []
                    
                    for i in range(num_samples):
                        if not st.session_state.simulation_running:
                            break
                        
                        # Generate synthetic observation
                        last_values = st.session_state.df_processed[feature_cols].iloc[-1].values
                        noise = np.random.normal(0, 0.1, len(feature_cols))
                        
                        # Inject anomalies
                        if np.random.random() < anomaly_injection:
                            new_obs = last_values * np.random.uniform(0.3, 2.0, len(feature_cols))
                        else:
                            new_obs = last_values + noise
                        
                        # Predict
                        prediction = st.session_state.model.predict(new_obs.reshape(1, -1))[0]
                        score = st.session_state.model.predict_scores(new_obs.reshape(1, -1))[0]
                        
                        stream_data.append({
                            'timestamp': datetime.now() - timedelta(seconds=(num_samples - i) * update_frequency),
                            'value': new_obs[0],
                            'anomaly': 1 if prediction == -1 else 0,
                            'score': score
                        })
                        
                        # Update chart every 5 samples
                        if i % 5 == 0 or i == num_samples - 1:
                            stream_df = pd.DataFrame(stream_data)
                            
                            # Plot
                            fig = go.Figure()
                            
                            normal = stream_df[stream_df['anomaly'] == 0]
                            anomalies = stream_df[stream_df['anomaly'] == 1]
                            
                            fig.add_trace(go.Scatter(
                                x=normal['timestamp'],
                                y=normal['value'],
                                mode='lines+markers',
                                name='Normal',
                                line=dict(color='blue')
                            ))
                            
                            if len(anomalies) > 0:
                                fig.add_trace(go.Scatter(
                                    x=anomalies['timestamp'],
                                    y=anomalies['value'],
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(color='red', size=10)
                                ))
                            
                            fig.update_layout(
                                title="Live Energy Stream",
                                xaxis_title="Time",
                                yaxis_title="Energy",
                                height=400,
                                showlegend=True,
                                template='plotly_white'
                            )
                            
                            with chart_placeholder.container():
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Stats
                            anomaly_count = (stream_df['anomaly'] == 1).sum()
                            avg_score = stream_df['score'].mean()
                            
                            with stats_placeholder.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Samples", len(stream_df))
                                with col2:
                                    st.metric("Anomalies", anomaly_count)
                                with col3:
                                    st.metric("Avg Score", f"{avg_score:.3f}")
                        
                        time.sleep(update_frequency)
                    
                    st.success("✅ Simulation completed!")
                
                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    st.session_state.simulation_running = False
        
        else:
            st.info("👈 Please train the model first using the sidebar.")
    
    # ===========================
    # TAB 5: RESULTS & ANALYSIS
    # ===========================
    with tab5:
        st.markdown("## Anomaly Detection Results")
        
        if st.session_state.model_trained and st.session_state.df_with_predictions is not None:
            df_results = st.session_state.df_with_predictions.copy()
            
            # Filter by anomaly threshold
            df_results = filter_anomalies_by_threshold(
                df_results,
                scores_col='anomaly_score',
                threshold=st.session_state.anomaly_threshold,
                anomaly_col='anomaly'
            )
            
            # Get anomalies
            anomalies_df = df_results[df_results['anomaly'] == 1].copy()
            
            st.markdown(f"### 🚨 Detected Anomalies ({len(anomalies_df)} found)")
            
            if len(anomalies_df) > 0:
                # Display top anomalies
                top_n = st.slider(
                    "Show top N anomalies:",
                    min_value=1,
                    max_value=min(len(anomalies_df), 50),
                    value=min(10, len(anomalies_df)),
                    step=1
                )
                
                display_cols = [st.session_state.datetime_col, st.session_state.target_col, 
                               'anomaly_score']
                
                top_anomalies = anomalies_df.nlargest(top_n, 'anomaly_score')[display_cols]
                
                st.dataframe(top_anomalies, use_container_width=True)
                
                # Anomaly Details
                st.markdown("### 📊 Anomaly Analysis")
                
                if st.checkbox("Show detailed explanation"):
                    selected_idx = st.number_input(
                        "Select anomaly index:",
                        min_value=0,
                        max_value=len(anomalies_df) - 1,
                        value=0
                    )
                    
                    anomaly_idx = anomalies_df.index[selected_idx]
                    explanation = get_anomaly_explanation(
                        df_results,
                        anomaly_idx,
                        anomaly_col='anomaly',
                        scores_col='anomaly_score',
                        target_col=st.session_state.target_col
                    )
                    
                    st.markdown(f"**Selected Anomaly**: Row {anomaly_idx}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Anomaly Score", f"{explanation['anomaly_score']:.3f}")
                        st.metric("Actual Value", f"{explanation['value']:.2f}")
                        st.metric("Mean Value", f"{explanation['mean_value']:.2f}")
                        st.metric("Std Dev", f"{explanation['std_value']:.2f}")
                    
                    with col2:
                        st.metric("Z-Score", f"{explanation['z_score']:.2f}")
                        st.metric("Deviation", f"{explanation['deviation_from_mean']:.2f}")
                        st.metric("Deviation %", f"{explanation['deviation_percent']:.1f}%")
                        if 'rolling_mean' in explanation:
                            st.metric("Rolling Mean", f"{explanation['rolling_mean']:.2f}")

                    timestamp_value = None
                    if st.session_state.datetime_col in df_results.columns:
                        timestamp_value = df_results.loc[anomaly_idx, st.session_state.datetime_col]

                    probable_causes = get_probable_causes(explanation, timestamp_value=timestamp_value)
                    st.markdown("#### Probable Causes")
                    for cause in probable_causes:
                        st.markdown(f"- {cause}")
            
            else:
                st.success("✅ No anomalies detected with current threshold!")
            
            # Export Results
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export all predictions
                csv_all = df_results.to_csv(index=False)
                st.download_button(
                    label="Download All Results (CSV)",
                    data=csv_all,
                    file_name="all_predictions.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export anomalies only
                if len(anomalies_df) > 0:
                    csv_anomalies = anomalies_df.to_csv(index=False)
                    st.download_button(
                        label="Download Anomalies Only (CSV)",
                        data=csv_anomalies,
                        file_name="detected_anomalies.csv",
                        mime="text/csv"
                    )
        
        else:
            st.info("👈 Please train the model first using the sidebar.")


# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown(
    "💡 **Smart Energy Anomaly Detection System** | "
    "Built with Python, Streamlit, and Scikit-Learn | "
    "© 2024"
)
