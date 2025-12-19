"""
Extended visualization utilities for the dashboard.
Includes helpers for Traffic Insights and Reporting.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Optional
from datetime import datetime

def parse_ports_from_description(df: pd.DataFrame) -> pd.DataFrame:
    """Extract port numbers from alert descriptions."""
    if df.empty:
        return df
    
    def extract_port(desc):
        # Look for "port X" patterns
        match = re.search(r'port\s+(\d+)', str(desc), re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    df['target_port'] = df['description'].apply(extract_port)
    return df

def plot_port_distribution(df: pd.DataFrame):
    """Plot distribution of targeted ports."""
    df_ports = parse_ports_from_description(df.copy())
    
    if 'target_port' not in df_ports.columns or df_ports['target_port'].dropna().empty:
        return None

    port_counts = df_ports['target_port'].value_counts().reset_index()
    port_counts.columns = ['Port', 'Count']
    port_counts['Port'] = port_counts['Port'].astype(str) # treated as categorical

    fig = px.bar(
        port_counts.head(15), 
        x='Port', 
        y='Count',
        title='Top Targeted Ports',
        labels={'Port': 'Port Number', 'Count': 'Alert Count'},
        color='Count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_hourly_heatmap(df: pd.DataFrame):
    """Plot heatmap of anomalies by hour of day and day of week."""
    if df.empty:
        return None
        
    df['hour'] = df['timestamp'].dt.hour
    df['day_name'] = df['timestamp'].dt.day_name()
    
    # Order of days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    heatmap_data = df.groupby(['day_name', 'hour']).size().reset_index(name='count')
    
    # Pivot for heatmap format
    heatmap_matrix = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
    
    # Ensure all days/hours represented if possible (optional, but cleaner)
    # for now, simpler matrix is fine
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix.values,
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Anomalies')
    ))
    
    fig.update_layout(
        title='Anomaly Heatmap (Day vs Hour)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        yaxis={'categoryorder': 'array', 'categoryarray': days_order}
    )
    return fig

def generate_html_report(df: pd.DataFrame, stats: Dict) -> str:
    """Generate a simple HTML report summary."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    top_ips_html = ""
    if not df.empty:
        top_ips = df['ip_address'].value_counts().head(5).reset_index()
        top_ips.columns = ['IP Address', 'Count']
        top_ips_html = top_ips.to_html(index=False, classes='table')

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            h1 {{ color: #cc0000; }}
            h2 {{ color: #444; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            .summary-box {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .table th, .table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
            .table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Security Incident Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <ul>
                <li><strong>Total Anomalies Detected:</strong> {stats.get('total_anomalies', 0)}</li>
                <li><strong>ML-Detected Anomalies:</strong> {stats.get('ml_anomalies', 0)}</li>
                <li><strong>Rule-Based Alerts:</strong> {stats.get('rule_anomalies', 0)}</li>
                <li><strong>Unique Suspicious IPs:</strong> {stats.get('unique_ips', 0)}</li>
            </ul>
        </div>

        <h2>Top Suspicious Source IPs</h2>
        {top_ips_html}

        <h2>Action Items</h2>
        <ul>
            <li>Investigate top source IPs for potential blacklisting.</li>
            <li>Review commonly targeted ports to ensure firewall rules are sufficient.</li>
            <li>Correlate ML anomalies with system logs.</li>
        </ul>
    </body>
    </html>
    """
    return html
