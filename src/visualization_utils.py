"""Visualization utilities for the anomaly detection dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .dashboard_config import (
    COLORS, CHART_HEIGHT, FEATURE_NAMES, VIZ_CONFIG,
    ANOMALY_SCORE_THRESHOLD
)


def plot_anomaly_timeline(df: pd.DataFrame, show_threshold: bool = True) -> go.Figure:
    """Create timeline chart showing anomaly scores over time.
    
    Args:
        df: DataFrame with 'timestamp', 'anomaly_score', 'alert_type' columns
        show_threshold: Whether to show the anomaly threshold line
        
    Returns:
        Plotly figure
    """
    if df.empty:
        # Return empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS["text"])
        )
        fig.update_layout(
            height=CHART_HEIGHT,
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"])
        )
        return fig
    
    # Filter only ML anomalies (they have scores)
    ml_df = df[df['alert_type'] == 'ML'].copy()
    
    if ml_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ML anomalies to plot (only rule-based alerts)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
        fig.update_layout(
            height=CHART_HEIGHT,
            plot_bgcolor=COLORS["background"],
            paper_bgcolor=COLORS["background"],
            font=dict(color=COLORS["text"])
        )
        return fig
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add anomaly points
    fig.add_trace(go.Scatter(
        x=ml_df['timestamp'],
        y=ml_df['anomaly_score'],
        mode='markers+lines',
        name='Anomaly Score',
        marker=dict(
            size=8,
            color=ml_df['anomaly_score'],
            colorscale='RdYlGn',
            reversescale=True,
            showscale=True,
            colorbar=dict(title="Score"),
            line=dict(width=1, color='white')
        ),
        line=dict(width=1, color=COLORS["info"]),
        text=ml_df['ip_address'],
        hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add threshold line
    if show_threshold:
        fig.add_hline(
            y=ANOMALY_SCORE_THRESHOLD,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text=f"Threshold ({ANOMALY_SCORE_THRESHOLD})",
            annotation_position="right"
        )
    
    fig.update_layout(
        title="Anomaly Score Timeline",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        height=CHART_HEIGHT,
        hovermode='closest',
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor='#2d2d2d'),
        yaxis=dict(gridcolor='#2d2d2d')
    )
    
    return fig


def plot_score_distribution(df: pd.DataFrame) -> go.Figure:
    """Create histogram of anomaly score distribution.
    
    Args:
        df: DataFrame with 'anomaly_score' column
        
    Returns:
        Plotly figure
    """
    ml_df = df[df['alert_type'] == 'ML']
    
    if ml_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ML anomaly scores available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
    else:
        fig = go.Figure(data=[go.Histogram(
            x=ml_df['anomaly_score'],
            nbinsx=30,
            marker_color=COLORS["primary"],
            opacity=0.7
        )])
        
        # Add threshold line
        fig.add_vline(
            x=ANOMALY_SCORE_THRESHOLD,
            line_dash="dash",
            line_color=COLORS["warning"],
            annotation_text="Threshold"
        )
    
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        height=CHART_HEIGHT,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor='#2d2d2d'),
        yaxis=dict(gridcolor='#2d2d2d')
    )
    
    return fig


def plot_anomaly_rate_over_time(df: pd.DataFrame, interval: str = '1H') -> go.Figure:
    """Plot anomaly detection rate over time.
    
    Args:
        df: DataFrame with 'timestamp' column
        interval: Time interval for aggregation ('5min', '1H', '1D', etc.)
        
    Returns:
        Plotly figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
    else:
        # Resample to count anomalies per interval
        df_copy = df.set_index('timestamp')
        counts = df_copy.resample(interval).size()
        
        fig = go.Figure(data=[go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=COLORS["primary"]
        )])
    
    fig.update_layout(
        title=f"Anomaly Detection Rate (per {interval})",
        xaxis_title="Time",
        yaxis_title="Number of Anomalies",
        height=CHART_HEIGHT,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor='#2d2d2d'),
        yaxis=dict(gridcolor='#2d2d2d')
    )
    
    return fig


def plot_top_ips(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Plot top IPs by anomaly count.
    
    Args:
        df: DataFrame with 'ip_address' column
        top_n: Number of top IPs to show
        
    Returns:
        Plotly figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
    else:
        # Count anomalies per IP
        ip_counts = df['ip_address'].value_counts().head(top_n)
        
        fig = go.Figure(data=[go.Bar(
            x=ip_counts.values,
            y=ip_counts.index,
            orientation='h',
            marker_color=COLORS["anomaly"]
        )])
    
    fig.update_layout(
        title=f"Top {top_n} IP Addresses by Anomaly Count",
        xaxis_title="Number of Anomalies",
        yaxis_title="IP Address",
        height=CHART_HEIGHT,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor='#2d2d2d'),
        yaxis=dict(gridcolor='#2d2d2d')
    )
    
    return fig


def plot_alert_type_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot pie chart of alert types (ML vs Rule-based).
    
    Args:
        df: DataFrame with 'alert_type' column
        
    Returns:
        Plotly figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
    else:
        type_counts = df['alert_type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            marker=dict(colors=[COLORS["anomaly"], COLORS["warning"]]),
            hole=0.4
        )])
    
    fig.update_layout(
        title="Alert Type Distribution",
        height=CHART_HEIGHT,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"])
    )
    
    return fig


def plot_feature_heatmap(features_df: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap of features.
    
    Args:
        features_df: DataFrame with feature columns
        
    Returns:
        Plotly figure
    """
    if features_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
    else:
        # Calculate correlation matrix
        corr = features_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=VIZ_CONFIG["heatmap_colorscale"],
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation")
        ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"])
    )
    
    return fig


def plot_feature_scatter_2d(
    features_df: pd.DataFrame,
    labels: pd.Series,
    method: str = 'pca'
) -> go.Figure:
    """Create 2D scatter plot of features using dimensionality reduction.
    
    Args:
        features_df: DataFrame with feature columns
        labels: Series with labels (0=normal, 1=anomaly)
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        
    Returns:
        Plotly figure
    """
    if features_df.empty or len(features_df) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text"])
        )
        return fig
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        title_method = "PCA"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title_method = "t-SNE"
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=VIZ_CONFIG["umap_neighbors"],
                min_dist=VIZ_CONFIG["umap_min_dist"],
                random_state=42
            )
            title_method = "UMAP"
        except ImportError:
            # Fallback to PCA if UMAP not installed
            reducer = PCA(n_components=2)
            title_method = "PCA (UMAP unavailable)"
    else:
        reducer = PCA(n_components=2)
        title_method = "PCA"
    
    # Fit and transform
    coords = reducer.fit_transform(features_df.values)
    
    # Create scatter plot
    df_plot = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'label': labels.map({0: 'Normal', 1: 'Anomaly'})
    })
    
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='label',
        color_discrete_map={'Normal': COLORS["normal"], 'Anomaly': COLORS["anomaly"]},
        title=f"Feature Space Visualization ({title_method})",
        opacity=VIZ_CONFIG["scatter_opacity"]
    )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor='#2d2d2d', title=f"{title_method} Component 1"),
        yaxis=dict(gridcolor='#2d2d2d', title=f"{title_method} Component 2")
    )
    
    return fig


def create_metric_card(value: any, label: str, delta: Optional[any] = None) -> Dict:
    """Create a metric card data structure.
    
    Args:
        value: Metric value to display
        label: Metric label
        delta: Optional delta value for comparison
        
    Returns:
        Dictionary with metric information
    """
    return {
        "value": value,
        "label": label,
        "delta": delta
    }
