"""
Traffic Flow Prediction Dashboard
Real-time traffic prediction using DCRNN model

Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dashboard.predictor import load_predictor
from dashboard.utils import (
    get_traffic_color, get_speed_category, calculate_congestion_score,
    get_congestion_emoji, load_sensor_metadata, save_sensor_metadata,
    get_timestep_for_horizon
)
from dashboard.visualization import (
    create_traffic_map, create_time_series_plot,
    create_multi_sensor_comparison, create_horizon_comparison,
    create_network_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Traffic Flow Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction',
        'Report a bug': "https://github.com/vaish725/Spatio-Temporal-Traffic-Flow-Prediction/issues",
        'About': "# Traffic Flow Prediction Dashboard\nForecasting traffic using DCRNN model"
    }
)

# Custom CSS for better UI - Dark/Light mode responsive
st.markdown("""
    <style>
    /* Main container */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Metrics styling - auto adapts to theme */
    .stMetric {
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Sensor cards - responsive to dark/light mode */
    .sensor-card {
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        min-height: 100px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    /* Light mode sensor cards */
    @media (prefers-color-scheme: light) {
        .sensor-card {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sensor-card .highway {
            color: #666666;
        }
    }
    
    /* Dark mode sensor cards */
    @media (prefers-color-scheme: dark) {
        .sensor-card {
            background-color: #262730;
            border: 1px solid #464655;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .sensor-card .highway {
            color: #a0a0a0;
        }
    }
    
    /* Streamlit dark mode override */
    [data-testid="stAppViewContainer"][data-theme="dark"] .sensor-card {
        background-color: #262730;
        border: 1px solid #464655;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    [data-testid="stAppViewContainer"][data-theme="dark"] .sensor-card .highway {
        color: #a0a0a0;
    }
    
    /* Streamlit light mode override */
    [data-testid="stAppViewContainer"][data-theme="light"] .sensor-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stAppViewContainer"][data-theme="light"] .sensor-card .highway {
        color: #666666;
    }
    
    /* Card text elements */
    .sensor-card b {
        font-size: 0.9em;
        display: block;
        margin-bottom: 4px;
    }
    
    .sensor-card .highway {
        font-size: 0.75em;
        margin-bottom: 6px;
        display: block;
    }
    
    .sensor-card .speed {
        font-size: 1.5em;
        font-weight: bold;
        margin: 8px 0;
        display: block;
    }
    
    .sensor-card .status {
        font-weight: 600;
        font-size: 0.85em;
        display: block;
    }
    
    /* Hover effects */
    .sensor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """Load model and sensor metadata (cached)"""
    try:
        predictor = load_predictor()
        sensors = load_sensor_metadata()
        
        # Generate and save sensor metadata if not exists
        if not os.path.exists('data/sensor_metadata.json'):
            save_sensor_metadata(sensors)
        
        return predictor, sensors
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure you have:\n1. Trained model in checkpoints/\n2. Processed data in data/")
        st.stop()


def main():
    # Header
    st.title("Traffic Flow Prediction Dashboard")
    st.markdown("Multi-horizon traffic forecasting using Deep Convolutional Recurrent Neural Network (DCRNN)")
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        predictor, sensors = load_model_and_data()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Get test data date range
    date_range = predictor.get_test_date_range()
    
    # Time horizon selector
    time_horizon = st.sidebar.selectbox(
        "🕐 Prediction Time Horizon",
        options=[5, 15, 30, 60],
        index=0,
        help="How far ahead to predict traffic conditions"
    )
    
    st.sidebar.markdown("---")
    
    # Date/Time picker
    st.sidebar.markdown("**📅 Select Date & Time**")
    
    col1, col2 = st.sidebar.columns([2, 1])
    
    with col1:
        # Date input
        from datetime import datetime
        selected_date = st.date_input(
            "Date",
            value=date_range['end_datetime'].date(),
            min_value=date_range['start_datetime'].date(),
            max_value=date_range['end_datetime'].date(),
            help=f"Available: {date_range['start_date']} to {date_range['end_date']}"
        )
    
    with col2:
        # Time input
        selected_time = st.time_input(
            "Time",
            value=date_range['end_datetime'].time(),
            help="Select hour and minute"
        )
    
    # Find corresponding sample
    result = predictor.find_sample_by_datetime(
        str(selected_date),
        selected_time.strftime('%H:%M:%S')
    )
    
    if 'error' in result:
        st.sidebar.error(result['error'])
        sample_idx = -1  # Fallback to most recent
    else:
        sample_idx = result['sample_idx']
        st.sidebar.success(
            f"✓ Found sample at:\n"
            f"{result['actual_date']} {result['actual_time']}\n"
            f"({result['weekday']})"
        )
    
    # Display timestamp information
    timestamp_info = predictor.get_sample_timestamp(sample_idx)
    st.sidebar.info(
        f"**🕐 Current Time Point**\n\n"
        f"📅 **Date:** {timestamp_info['date']}\n\n"
        f"⏰ **Time:** {timestamp_info['time']}\n\n"
        f"📆 **Day:** {timestamp_info['weekday']}\n\n"
        f"📍 **Sample:** {timestamp_info['sample_idx'] + 1} / {timestamp_info['total_test_samples']}"
    )
    
    # View mode
    view_mode = st.sidebar.radio(
        "📍 View Mode",
        options=["Network Overview", "Sensor Details", "Comparison View"],
        index=0
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Predictions"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Legend")
    st.sidebar.markdown("🟢 **Free Flow** (≥50 mph)")
    st.sidebar.markdown("🟡 **Moderate** (35-50 mph)")
    st.sidebar.markdown("🟠 **Slow** (20-35 mph)")
    st.sidebar.markdown("🔴 **Congested** (<20 mph)")
    
    # Generate predictions
    with st.spinner("Generating predictions..."):
        historical, ground_truth = predictor.get_latest_data(sample_idx)
        predictions = predictor.predict(historical, sample_idx)
        network_summary = predictor.get_network_summary(predictions)
    
    # Get timestep for selected horizon
    timestep = get_timestep_for_horizon(time_horizon)
    
    # Display based on view mode
    if view_mode == "Network Overview":
        show_network_overview(predictor, sensors, predictions, timestep, time_horizon, network_summary)
    
    elif view_mode == "Sensor Details":
        show_sensor_details(predictor, sensors, predictions, historical, ground_truth, timestep, time_horizon)
    
    else:  # Comparison View
        show_comparison_view(predictor, sensors, predictions, historical, ground_truth)
    
    # Footer
    st.sidebar.markdown("---")


def show_network_overview(predictor, sensors, predictions, timestep, time_horizon, summary):
    """Display network-wide overview"""
    
    # Network health metrics
    st.header(f"🌐 Network Overview - {time_horizon} Minutes Ahead")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emoji = get_congestion_emoji(100 - summary['health_score'])
        st.metric(
            label="Network Health",
            value=f"{summary['health_score']:.0f}/100",
            delta=emoji
        )
    
    with col2:
        st.metric(
            label="Average Speed",
            value=f"{summary['avg_speed']:.1f} mph",
            delta=get_speed_category(summary['avg_speed'])
        )
    
    with col3:
        st.metric(
            label="Congested Sensors",
            value=summary['congested_count'],
            delta=f"{(summary['congested_count']/summary['total_sensors']*100):.1f}%"
        )
    
    with col4:
        st.metric(
            label="Free Flow Sensors",
            value=summary['free_flow_count'],
            delta=f"{(summary['free_flow_count']/summary['total_sensors']*100):.1f}%"
        )
    
    # Traffic distribution
    st.subheader("Traffic Distribution")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive map
        fig_map = create_traffic_map(sensors, predictions, timestep)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("### Sensor Status")
        st.markdown(f"🟢 Free Flow: **{summary['free_flow_count']}**")
        st.markdown(f"🟡 Moderate: **{summary['moderate_count']}**")
        st.markdown(f"🟠 Slow: **{summary['slow_count']}**")
        st.markdown(f"🔴 Congested: **{summary['congested_count']}**")
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.markdown(f"**Max Speed:** {summary['max_speed']:.1f} mph")
        st.markdown(f"**Min Speed:** {summary['min_speed']:.1f} mph")
        st.markdown(f"**Total Sensors:** {summary['total_sensors']}")
    
    # Top congested areas
    st.subheader("Top 10 Most Congested Areas Right Now")
    congested = predictor.get_top_congested_sensors(predictions, n=10)
    
    cols = st.columns(5)
    for i, (sensor_id, speed) in enumerate(congested):
        with cols[i % 5]:
            sensor_info = sensors[str(sensor_id)]
            color, status = get_traffic_color(speed)
            st.markdown(
                f"<div class='sensor-card' style='border-left: 5px solid {color};'>"
                f"<b>Sensor {sensor_id}</b>"
                f"<span class='highway'>{sensor_info['highway']} {sensor_info['direction']}</span>"
                f"<span class='speed' style='color: {color};'>{speed:.1f} mph</span>"
                f"<span class='status' style='color: {color};'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    # Network speed heatmap
    st.subheader("Network Speed Heatmap")
    fig_heatmap = create_network_heatmap(predictions, timestep, num_sensors=100)
    st.plotly_chart(fig_heatmap, use_container_width=True)


def show_sensor_details(predictor, sensors, predictions, historical, ground_truth, timestep, time_horizon):
    """Display detailed sensor information"""
    
    st.header("🔍 Sensor Details")
    
    # Sensor selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sensor_id = st.number_input(
            "Select Sensor ID",
            min_value=0,
            max_value=predictor.num_nodes - 1,
            value=0,
            step=1
        )
    
    with col2:
        # Quick search by highway
        highways = sorted(list(set(s['highway'] for s in sensors.values())))
        selected_highway = st.selectbox("Filter by Highway", ["All"] + highways)
    
    # Get sensor info
    sensor_info = sensors[str(sensor_id)]
    current_speed = predictions[timestep, sensor_id]
    color, status = get_traffic_color(current_speed)
    
    # Sensor info card
    st.markdown(
        f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; color: white;'>"
        f"<h2>Sensor {sensor_id}</h2>"
        f"<h3>{sensor_info['highway']} {sensor_info['direction']}</h3>"
        f"<h1>{current_speed:.1f} mph</h1>"
        f"<h4>{status}</h4>"
        f"<p>Prediction for {time_horizon} minutes ahead</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Time series prediction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Speed Prediction Timeline")
        fig_ts = create_time_series_plot(
            historical[:, sensor_id],
            predictions[:, sensor_id],
            ground_truth[:, sensor_id],
            title=f"Sensor {sensor_id} - Speed Over Time"
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with col2:
        st.subheader("⏰ Multi-Horizon Predictions")
        fig_horizon = create_horizon_comparison(predictions, sensor_id)
        st.plotly_chart(fig_horizon, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📊 Detailed Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hist_avg = historical[:, sensor_id].mean()
        st.metric("Historical Avg", f"{hist_avg:.1f} mph")
    
    with col2:
        pred_avg = predictions[:, sensor_id].mean()
        st.metric("Predicted Avg", f"{pred_avg:.1f} mph")
    
    with col3:
        change = current_speed - historical[-1, sensor_id]
        st.metric(
            "Speed Change",
            f"{abs(change):.1f} mph",
            delta=f"{'↑' if change > 0 else '↓'}"
        )
    
    with col4:
        trend = "Improving" if pred_avg > hist_avg else "Worsening"
        st.metric("Trend", trend)
    
    # Location info
    with st.expander("📍 Location Details"):
        st.write(f"**Highway:** {sensor_info['highway']}")
        st.write(f"**Direction:** {sensor_info['direction']}")
        st.write(f"**Coordinates:** {sensor_info['lat']:.4f}, {sensor_info['lon']:.4f}")


def show_comparison_view(predictor, sensors, predictions, historical, ground_truth):
    """Display comparison between multiple sensors"""
    
    st.header("⚖️ Sensor Comparison")
    
    st.markdown("Compare traffic predictions across multiple sensors")
    
    # Sensor selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sensor_ids_input = st.text_input(
            "Enter Sensor IDs (comma-separated)",
            value="0, 10, 50, 100",
            help="E.g., 0, 10, 50, 100"
        )
    
    with col2:
        num_random = st.number_input("Or Random Sensors", min_value=2, max_value=10, value=4)
        if st.button("🎲 Random"):
            sensor_ids_input = ", ".join(map(str, np.random.choice(predictor.num_nodes, num_random, replace=False)))
    
    # Parse sensor IDs
    try:
        sensor_ids = [int(x.strip()) for x in sensor_ids_input.split(",")]
        sensor_ids = [s for s in sensor_ids if 0 <= s < predictor.num_nodes]
    except:
        st.error("Invalid sensor IDs. Please enter comma-separated numbers.")
        return
    
    if not sensor_ids:
        st.warning("Please enter valid sensor IDs.")
        return
    
    # Get comparison data
    sensor_data = predictor.compare_sensors(sensor_ids, historical)
    
    # Display sensor cards
    st.subheader("📊 Selected Sensors")
    cols = st.columns(min(4, len(sensor_ids)))
    for i, sensor_id in enumerate(sensor_ids):
        with cols[i % len(cols)]:
            sensor_info = sensors[str(sensor_id)]
            current_speed = predictions[0, sensor_id]
            color, status = get_traffic_color(current_speed)
            
            st.markdown(
                f"<div class='sensor-card' style='border-left: 5px solid {color};'>"
                f"<b>Sensor {sensor_id}</b>"
                f"<span class='highway'>{sensor_info['highway']} {sensor_info['direction']}</span>"
                f"<span class='speed' style='color: {color};'>{current_speed:.1f} mph</span>"
                f"<span class='status' style='color: {color};'>{status}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    # Comparison charts
    st.subheader("📈 Prediction Comparison")
    
    tab1, tab2 = st.tabs(["Time Series", "Side-by-Side"])
    
    with tab1:
        fig_comparison = create_multi_sensor_comparison(sensor_data, sensors, time_range='both')
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with tab2:
        cols = st.columns(len(sensor_ids))
        for i, sensor_id in enumerate(sensor_ids):
            with cols[i]:
                fig_ts = create_time_series_plot(
                    sensor_data[sensor_id]['historical'],
                    sensor_data[sensor_id]['predicted'],
                    sensor_data[sensor_id]['ground_truth'],
                    title=f"Sensor {sensor_id}"
                )
                st.plotly_chart(fig_ts, use_container_width=True)
    
    # Statistics table
    st.subheader("📊 Comparison Statistics")
    
    stats_data = []
    for sensor_id in sensor_ids:
        sensor_info = sensors[str(sensor_id)]
        hist_avg = sensor_data[sensor_id]['historical'].mean()
        pred_avg = sensor_data[sensor_id]['predicted'].mean()
        
        stats_data.append({
            "Sensor": sensor_id,
            "Highway": sensor_info['highway'],
            "Historical Avg": f"{hist_avg:.1f} mph",
            "Predicted Avg": f"{pred_avg:.1f} mph",
            "Change": f"{pred_avg - hist_avg:+.1f} mph",
            "Status": get_speed_category(pred_avg)
        })
    
    st.table(stats_data)


if __name__ == "__main__":
    main()
