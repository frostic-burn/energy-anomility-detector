"""
Visualization utilities using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_timeseries_anomalies(df, datetime_col, target_col, anomaly_col, title="Energy Consumption with Anomalies"):
    """
    Create interactive time-series plot with anomaly highlighting
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Datetime column name
    target_col : str
        Target/energy column name
    anomaly_col : str
        Column with anomaly labels (0 or 1)
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    normal_data = df[df[anomaly_col] == 0]
    anomaly_data = df[df[anomaly_col] == 1]
    
    fig = go.Figure()
    
    # Normal points
    fig.add_trace(go.Scatter(
        x=normal_data[datetime_col],
        y=normal_data[target_col],
        mode='lines+markers',
        name='Normal',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Time:</b> %{x}<br><b>Energy:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Anomalies
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data[datetime_col],
            y=anomaly_data[target_col],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8),
            hovertemplate='<b>Time:</b> %{x}<br><b>Energy:</b> %{y:.2f}<br><b>ANOMALY</b><extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=target_col,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_histogram(df, col, title="Distribution", nbins=30):
    """
    Create histogram of energy consumption
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column to plot
    title : str
        Plot title
    nbins : int
        Number of bins
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[col],
        nbinsx=nbins,
        marker=dict(color='#1f77b4'),
        name='Distribution',
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=col,
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_boxplot(df, col, title="Box Plot", group_col=None):
    """
    Create box plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    col : str
        Column to plot
    title : str
        Plot title
    group_col : str, optional
        Column for grouping
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    if group_col is None:
        fig.add_trace(go.Box(
            y=df[col],
            name=col,
            marker=dict(color='#1f77b4'),
            hovertemplate='<b>Value:</b> %{y:.2f}<extra></extra>'
        ))
    else:
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][col]
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                hovertemplate='<b>Value:</b> %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        yaxis_title=col,
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def plot_heatmap(df, x_col, y_col, z_col, title="Heatmap"):
    """
    Create heatmap (e.g., energy by hour vs day of week)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_col : str
        X-axis column
    y_col : str
        Y-axis column
    z_col : str
        Value column
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    pivot_df = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu',
        hovertemplate='<b>%{y}</b> at <b>%{x}</b><br>Value: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_correlation_matrix(df, title="Correlation Matrix"):
    """
    Create correlation matrix heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (should contain only numeric columns)
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_anomaly_scores(df, scores_col, threshold=None, title="Anomaly Scores Distribution"):
    """
    Create anomaly scores distribution plot
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    scores_col : str
        Column with anomaly scores
    threshold : float, optional
        Threshold line
    title : str
        Plot title
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[scores_col],
        nbinsx=50,
        marker=dict(color='#1f77b4'),
        name='Scores',
        hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))
    
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            name=f"Threshold: {threshold:.3f}",
            annotation_text=f"Threshold: {threshold:.3f}",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Anomaly Score',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def create_dashboard_summary(df, datetime_col, target_col, anomaly_col):
    """
    Create a summary dashboard with multiple subplots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Datetime column name
    target_col : str
        Target/energy column name
    anomaly_col : str
        Anomaly label column name
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Time Series", "Distribution", "Box Plot", "Anomalies Over Time"),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "box"}, {"type": "bar"}]]
    )
    
    # Time series
    fig.add_trace(
        go.Scatter(x=df[datetime_col], y=df[target_col], name="Energy",
                   mode='lines', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[target_col], name="Distribution",
                     marker=dict(color='green'), nbinsx=30),
        row=1, col=2
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[target_col], name="Box Plot",
               marker=dict(color='orange')),
        row=2, col=1
    )
    
    # Anomalies over time
    anomaly_counts = df.groupby(df[datetime_col].dt.date)[anomaly_col].sum()
    fig.add_trace(
        go.Bar(x=anomaly_counts.index, y=anomaly_counts.values,
               name="Anomalies", marker=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text=target_col, row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Energy", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, template='plotly_white')
    
    return fig
