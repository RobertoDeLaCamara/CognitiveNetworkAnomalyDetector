"""Streamlit Anomaly Detection Dashboard

A comprehensive visualization dashboard for real-time and historical
anomaly detection monitoring.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import dashboard modules
from src.dashboard_config import (
    DASHBOARD_TITLE, DASHBOARD_ICON, DEFAULT_REFRESH_INTERVAL,
    COLORS, MLFLOW_ENABLED, MAX_ANOMALIES_DISPLAY
)
from src.dashboard_data import (
    AnomalyDataLoader, ModelMetricsLoader, MLflowDataLoader
)
from src.visualization_utils import (
    plot_anomaly_timeline, plot_score_distribution,
    plot_anomaly_rate_over_time, plot_top_ips,
    plot_alert_type_distribution
)


# Page configuration
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon=DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown(f"""
    <style>
    .stMetric {{
        background-color: {COLORS["background"]};
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #2d2d2d;
    }}
    .metric-good {{
        color: {COLORS["normal"]};
    }}
    .metric-bad {{
        color: {COLORS["anomaly"]};
    }}
    .metric-warning {{
        color: {COLORS["warning"]};
    }}
    </style>
""", unsafe_allow_html=True)


# Initialize data loaders
@st.cache_resource
def init_data_loaders():
    """Initialize data loaders (cached)."""
    return {
        "anomaly": AnomalyDataLoader(),
        "model": ModelMetricsLoader(),
        "mlflow": MLflowDataLoader() if MLFLOW_ENABLED else None
    }


loaders = init_data_loaders()


# Sidebar navigation
st.sidebar.title(f"{DASHBOARD_ICON} Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Historical Analysis", "🔍 Anomaly Inspector", "🤖 Model Info", "📈 MLflow"]
)

# Auto-refresh toggle
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=1,
        max_value=60,
        value=DEFAULT_REFRESH_INTERVAL
    )
    st.sidebar.info(f"Refreshing every {refresh_interval}s")


# ============================================================================
# HOME PAGE - Real-time Monitoring
# ============================================================================
if page == "🏠 Home":
    st.title(f"{DASHBOARD_ICON} Real-Time Anomaly Monitoring")
    
    # Load recent anomalies
    recent_minutes = st.slider("Show anomalies from last N minutes", 5, 120, 30)
    
    with st.spinner("Loading anomaly data..."):
        recent_df = loaders["anomaly"].get_recent_anomalies(minutes=recent_minutes)
        stats = loaders["anomaly"].get_anomaly_stats()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Anomalies",
            value=stats["total_anomalies"],
            delta=None
        )
    
    with col2:
        st.metric(
            label="ML Detections",
            value=stats["ml_anomalies"],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Rule Detections",
            value=stats["rule_anomalies"],
            delta=None
        )
    
    with col4:
        st.metric(
            label="Unique IPs",
            value=stats["unique_ips"],
            delta=None
        )
    
    # Charts row 1
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_anomaly_timeline(recent_df),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_score_distribution(recent_df),
            use_container_width=True
        )
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            plot_top_ips(recent_df, top_n=10),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            plot_alert_type_distribution(recent_df),
            use_container_width=True
        )
    
    # Recent anomalies table
    st.markdown("---")
    st.subheader("📋 Recent Anomalies")
    
    if not recent_df.empty:
        display_df = recent_df.head(MAX_ANOMALIES_DISPLAY)[
            ['timestamp', 'ip_address', 'alert_type', 'anomaly_score', 'description']
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No anomalies detected in the selected time range.")
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================================
# HISTORICAL ANALYSIS PAGE
# ============================================================================
elif page == "📊 Historical Analysis":
    st.title("📊 Historical Anomaly Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7)
        )
        start_time = st.time_input("Start Time", value=datetime.min.time())
        start_datetime = datetime.combine(start_date, start_time)
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
        end_time = st.time_input("End Time", value=datetime.now().time())
        end_datetime = datetime.combine(end_date, end_time)
    
    # Load historical data
    with st.spinner("Loading historical data..."):
        df = loaders["anomaly"].load_anomalies(
            start_date=start_datetime,
            end_date=end_datetime
        )
    
    if df.empty:
        st.warning("No anomalies found in the selected date range.")
    else:
        st.success(f"Found {len(df)} anomalies")
        
        # Time interval selector
        interval = st.selectbox(
            "Aggregation Interval",
            ["5min", "15min", "1H", "4H", "1D"],
            index=2
        )
        
        # Anomaly rate over time
        st.plotly_chart(
            plot_anomaly_rate_over_time(df, interval=interval),
            use_container_width=True
        )
        
        # Two-column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_anomaly_timeline(df),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_score_distribution(df),
                use_container_width=True
            )
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        ml_df = df[df['alert_type'] == 'ML']
        
        with col1:
            st.metric("Total Anomalies", len(df))
        
        with col2:
            st.metric("ML Anomalies", len(ml_df))
        
        with col3:
            if len(ml_df) > 0:
                st.metric("Avg Score", f"{ml_df['anomaly_score'].mean():.3f}")
            else:
                st.metric("Avg Score", "N/A")
        
        with col4:
            st.metric("Unique IPs", df['ip_address'].nunique())
        
        # Top IPs
        st.plotly_chart(
            plot_top_ips(df, top_n=15),
            use_container_width=True
        )
        
        # Export data
        st.markdown("---")
        if st.button("📥 Download Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"anomalies_{start_date}_{end_date}.csv",
                mime="text/csv"
            )


# ============================================================================
# ANOMALY INSPECTOR PAGE
# ============================================================================
elif page == "🔍 Anomaly Inspector":
    st.title("🔍 Anomaly Inspector")
    
    # Search filters
    st.subheader("Search Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_ip = st.text_input("IP Address", placeholder="e.g., 192.168.1.100")
    
    with col2:
        alert_type_filter = st.selectbox("Alert Type", ["All", "ML", "RULE"])
    
    with col3:
        min_score = st.number_input(
            "Min Anomaly Score",
            value=-1.0,
            step=0.1,
            format="%.2f"
        )
    
    # Load and filter data
    with st.spinner("Searching anomalies..."):
        df = loaders["anomaly"].load_anomalies()
        
        # Apply filters
        if search_ip:
            df = df[df['ip_address'].str.contains(search_ip, case=False, na=False)]
        
        if alert_type_filter != "All":
            df = df[df['alert_type'] == alert_type_filter]
        
        if alert_type_filter == "ML" or alert_type_filter == "All":
            df = df[(df['alert_type'] == 'RULE') | (df['anomaly_score'] <= min_score)]
    
    st.info(f"Found {len(df)} matching anomalies")
    
    if not df.empty:
        # Display results
        st.dataframe(
            df[['timestamp', 'ip_address', 'alert_type', 'anomaly_score', 'description']],
            use_container_width=True,
            hide_index=True
        )
        
        # Detail view
        if len(df) > 0:
            st.markdown("---")
            st.subheader("Anomaly Details")
            
            selected_idx = st.selectbox(
                "Select anomaly to inspect",
                range(len(df)),
                format_func=lambda i: f"{df.iloc[i]['timestamp']} - {df.iloc[i]['ip_address']}"
            )
            
            if selected_idx is not None:
                anomaly = df.iloc[selected_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Timestamp:**")
                    st.write(anomaly['timestamp'])
                    
                    st.markdown("**IP Address:**")
                    st.write(anomaly['ip_address'])
                
                with col2:
                    st.markdown("**Alert Type:**")
                    st.write(anomaly['alert_type'])
                    
                    if anomaly['alert_type'] == 'ML':
                        st.markdown("**Anomaly Score:**")
                        st.write(f"{anomaly['anomaly_score']:.4f}")
                
                st.markdown("**Description:**")
                st.write(anomaly['description'])
    else:
        st.warning("No anomalies match the search criteria.")


# ============================================================================
# MODEL INFO PAGE
# ============================================================================
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")
    
    model_info = loaders["model"].get_model_info()
    
    if model_info:
        st.success("✅ Model loaded successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            st.write(f"**Model Type:** {model_info['model_type']}")
            st.write(f"**Number of Features:** {model_info['n_features']}")
            st.write(f"**Contamination:** {model_info['contamination']}")
            st.write(f"**Number of Estimators:** {model_info['n_estimators']}")
        
        with col2:
            st.subheader("Model Metadata")
            st.write(f"**Trained Date:** {model_info['trained_date']}")
            st.write(f"**Model Version:** {model_info['model_version']}")
            st.write(f"**File Size:** {model_info['file_size']:.2f} KB")
        
        st.markdown("---")
        st.subheader("Feature Names")
        
        from src.dashboard_config import FEATURE_NAMES
        
        # Display features in a nice grid
        cols = st.columns(3)
        for i, feature in enumerate(FEATURE_NAMES):
            with cols[i % 3]:
                st.write(f"{i+1}. {feature}")
    
    else:
        st.error("❌ No trained model found")
        st.info("Train a model using: `python train_model.py --duration 60 --version 1`")


# ============================================================================
# MLFLOW PAGE
# ============================================================================
elif page == "📈 MLflow":
    st.title("📈 MLflow Integration")
    
    if not MLFLOW_ENABLED:
        st.warning("⚠️ MLflow is not configured")
        st.info("""
        To enable MLflow integration:
        1. Set up a remote MLflow server
        2. Configure environment variables in `.env`
        3. See REMOTE_MLFLOW_SETUP.md for details
        """)
    else:
        mlflow_loader = loaders["mlflow"]
        
        st.success(f"✅ Connected to MLflow: {mlflow_loader.tracking_uri}")
        
        # Experiments
        st.subheader("Recent Experiments")
        experiments = mlflow_loader.get_experiments(limit=10)
        
        if experiments:
            exp_df = pd.DataFrame(experiments)
            st.dataframe(exp_df, use_container_width=True, hide_index=True)
        else:
            st.info("No experiments found")
        
        # Runs
        st.markdown("---")
        st.subheader("Recent Runs")
        
        runs_df = mlflow_loader.get_recent_runs(limit=20)
        
        if not runs_df.empty:
            # Select relevant columns
            display_cols = [col for col in [
                'run_name', 'start_time', 'status',
                'params.contamination', 'params.n_estimators',
                'metrics.training_time'
            ] if col in runs_df.columns]
            
            if display_cols:
                st.dataframe(
                    runs_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(runs_df, use_container_width=True)
        else:
            st.info("No runs found")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**{DASHBOARD_TITLE}**  
Real-time anomaly detection monitoring

Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")
