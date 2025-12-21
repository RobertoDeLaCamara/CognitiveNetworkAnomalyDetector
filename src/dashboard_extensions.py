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
import ipaddress
import socket

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
    
    # Create complete grid of all days and hours
    # This ensures the heatmap always shows the full week/day view even with sparse data
    full_index = pd.MultiIndex.from_product(
        [days_order, range(24)], 
        names=['day_name', 'hour']
    )
    
    # Group by day and hour
    heatmap_data = df.groupby(['day_name', 'hour']).size().reindex(full_index, fill_value=0).reset_index(name='count')
    
    # Pivot for heatmap format
    heatmap_matrix = heatmap_data.pivot(index='day_name', columns='hour', values='count')
    
    # Reorder index to match days_order
    heatmap_matrix = heatmap_matrix.reindex(days_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix.values,
        x=list(range(24)),
        y=heatmap_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title='Anomalies'),
        xgap=1, # Add small gap between cells for better visibility
        ygap=1
    ))
    
    fig.update_layout(
        title='Anomaly Heatmap (Day vs Hour)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        yaxis={'categoryorder': 'array', 'categoryarray': days_order},
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5]
        )
    )
    return fig

# Known public IP map
KNOWN_IPS = {
    # Google
    "8.8.8.8": "Google DNS (Primary)",
    "8.8.4.4": "Google DNS (Secondary)",
    # Cloudflare
    "1.1.1.1": "Cloudflare DNS",
    "1.0.0.1": "Cloudflare DNS",
    # Quad9
    "9.9.9.9": "Quad9 DNS",
    "149.112.112.112": "Quad9 DNS",
    # OpenDNS
    "208.67.222.222": "OpenDNS Home",
    "208.67.220.220": "OpenDNS Home",
    # Level3
    "4.2.2.1": "Level3 DNS",
    "4.2.2.2": "Level3 DNS",
    # Local router/gateway usually 
    "192.168.1.1": "Local Gateway",
    "192.168.0.1": "Local Gateway",
}

def get_ip_info(ip_str: str) -> str:
    """Categorize IP address."""
    # Check specific known IPs first
    if ip_str in KNOWN_IPS:
        return KNOWN_IPS[ip_str]
        
    try:
        ip = ipaddress.ip_address(ip_str)
        
        if ip.is_loopback:
            return "Localhost (Loopback)"
        if ip.is_private:
            return "Local Network (Private)"
        if ip.is_multicast:
            return "Multicast"
        if ip.is_reserved:
            return "Reserved"
            
        # Try dynamic resolution for public IPs
        try:
            # Short timeout to prevent UI blocking
            socket.setdefaulttimeout(0.5)
            hostname, _, _ = socket.gethostbyaddr(ip_str)
            return f"{hostname} (Public)"
        except (socket.herror, socket.timeout):
            return "Public Internet"
            
    except ValueError:
        return "Unknown/Invalid"

def plot_ip_category_distribution(df: pd.DataFrame):
    """Plot distribution of IP categories."""
    if df.empty:
        return None
        
    # Enrich data
    df_cat = df.copy()
    df_cat['category'] = df_cat['ip_address'].apply(get_ip_info)
    
    counts = df_cat['category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    
    fig = px.pie(
        counts, 
        values='Count', 
        names='Category',
        title='Source IP Categories',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    
    return fig

def get_known_ips_table(df: pd.DataFrame) -> pd.DataFrame:
    """Get table of detected known public IPs."""
    if df.empty:
        return pd.DataFrame()
        
    # Get categories for all unique IPs
    unique_ips = df['ip_address'].unique()
    ip_map = {ip: get_ip_info(ip) for ip in unique_ips}
    
    # Filter out generic or local categories
    ignored_categories = [
        "Public Internet", 
        "Local Network (Private)", 
        "Localhost (Loopback)", 
        "Multicast", 
        "Reserved",
        "Unknown/Invalid"
    ]
    
    # Create valid map for "Known" entities (Static + Resolved)
    valid_map = {
        ip: cat for ip, cat in ip_map.items() 
        if cat not in ignored_categories
    }
    
    if not valid_map:
        return pd.DataFrame()
        
    # Filter dataframe
    known_mask = df['ip_address'].isin(valid_map.keys())
    df_known = df[known_mask].copy()
    
    if df_known.empty:
        return pd.DataFrame()
        
    df_known['Entity Name'] = df_known['ip_address'].map(valid_map)
    
    summary = df_known.groupby(['ip_address', 'Entity Name']).size().reset_index(name='Alert Count')
    summary = summary.sort_values('Alert Count', ascending=False)
    
    return summary

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
