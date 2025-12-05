"""
Interactive map visualization for traffic predictions
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dashboard.utils import get_traffic_color


def create_traffic_map(sensors, predictions, timestep=0, selected_sensors=None):
    """
    Create interactive Plotly map with color-coded traffic sensors
    
    Args:
        sensors: Dictionary of sensor metadata
        predictions: (12, num_nodes) array of predicted speeds
        timestep: Which timestep to display (0-11)
        selected_sensors: List of sensor IDs to highlight
        
    Returns:
        Plotly figure
    """
    if selected_sensors is None:
        selected_sensors = []
    
    # Extract sensor data
    lats = []
    lons = []
    speeds = []
    colors = []
    texts = []
    sizes = []
    sensor_ids = []
    
    for sensor_id, sensor_info in sensors.items():
        idx = int(sensor_id)
        speed = predictions[timestep, idx]
        
        lats.append(sensor_info['lat'])
        lons.append(sensor_info['lon'])
        speeds.append(speed)
        
        color, status = get_traffic_color(speed)
        colors.append(color)
        
        # Hover text
        text = (
            f"<b>Sensor {idx}</b><br>"
            f"{sensor_info['highway']} {sensor_info['direction']}<br>"
            f"Speed: {speed:.1f} mph<br>"
            f"Status: {status}"
        )
        texts.append(text)
        
        # Highlight selected sensors
        if idx in selected_sensors:
            sizes.append(15)
        else:
            sizes.append(10)
        
        sensor_ids.append(idx)
    
    # Create scatter mapbox
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
        ),
        text=texts,
        hoverinfo='text',
        customdata=sensor_ids,
        name='Sensors'
    ))
    
    # Calculate center
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def create_sensor_comparison_map(sensors, sensor_ids, predictions, timestep=0):
    """
    Create map focused on selected sensors for comparison
    
    Args:
        sensors: Dictionary of sensor metadata
        sensor_ids: List of sensor IDs to display
        predictions: (12, num_nodes) array of predicted speeds
        timestep: Which timestep to display
        
    Returns:
        Plotly figure
    """
    lats = []
    lons = []
    speeds = []
    colors = []
    texts = []
    
    for idx in sensor_ids:
        sensor_info = sensors[str(idx)]
        speed = predictions[timestep, idx]
        
        lats.append(sensor_info['lat'])
        lons.append(sensor_info['lon'])
        speeds.append(speed)
        
        color, status = get_traffic_color(speed)
        colors.append(color)
        
        text = (
            f"<b>Sensor {idx}</b><br>"
            f"{sensor_info['highway']} {sensor_info['direction']}<br>"
            f"Speed: {speed:.1f} mph<br>"
            f"Status: {status}"
        )
        texts.append(text)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+text',
        marker=dict(
            size=20,
            color=colors,
            opacity=0.9,
        ),
        text=[f"S{idx}" for idx in sensor_ids],
        textposition="top center",
        textfont=dict(size=10, color='black'),
        hovertext=texts,
        hoverinfo='text'
    ))
    
    # Center on selected sensors
    if lats:
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        zoom = 11
    else:
        center_lat = 37.6
        center_lon = -122.2
        zoom = 9
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        showlegend=False
    )
    
    return fig


def create_time_series_plot(historical, predicted, ground_truth=None, title="Traffic Speed Prediction"):
    """
    Create time series plot with historical and predicted speeds
    
    Args:
        historical: (12,) array of historical speeds
        predicted: (12,) array of predicted speeds
        ground_truth: Optional (12,) array of actual speeds
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Time labels (in 5-minute intervals)
    hist_times = list(range(-60, 0, 5))  # -60 to -5 minutes
    pred_times = list(range(5, 65, 5))    # 5 to 60 minutes
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_times,
        y=historical,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Predicted data
    fig.add_trace(go.Scatter(
        x=pred_times,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Ground truth (if available)
    if ground_truth is not None:
        fig.add_trace(go.Scatter(
            x=pred_times,
            y=ground_truth,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=6),
            opacity=0.7
        ))
    
    # Add vertical line at present
    fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    # Add traffic level zones
    fig.add_hrect(y0=50, y1=100, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=35, y1=50, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=20, y1=35, fillcolor="orange", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=20, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (minutes from now)",
        yaxis_title="Speed (mph)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_multi_sensor_comparison(sensor_data, sensors, time_range='prediction'):
    """
    Create comparison plot for multiple sensors
    
    Args:
        sensor_data: Dict mapping sensor_id to {historical, predicted, ground_truth}
        sensors: Sensor metadata dictionary
        time_range: 'historical', 'prediction', or 'both'
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for sensor_id, data in sensor_data.items():
        sensor_info = sensors[str(sensor_id)]
        label = f"Sensor {sensor_id} ({sensor_info['highway']})"
        
        if time_range in ['historical', 'both']:
            hist_times = list(range(-60, 0, 5))
            fig.add_trace(go.Scatter(
                x=hist_times,
                y=data['historical'],
                mode='lines',
                name=f"{label} (Hist)",
                line=dict(width=2),
                opacity=0.6
            ))
        
        if time_range in ['prediction', 'both']:
            pred_times = list(range(5, 65, 5))
            fig.add_trace(go.Scatter(
                x=pred_times,
                y=data['predicted'],
                mode='lines+markers',
                name=f"{label} (Pred)",
                line=dict(width=2, dash='dash'),
                marker=dict(size=5)
            ))
    
    if time_range == 'both':
        fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Multi-Sensor Comparison",
        xaxis_title="Time (minutes from now)",
        yaxis_title="Speed (mph)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    return fig


def create_horizon_comparison(predictions, sensor_id, horizons=[5, 15, 30, 60]):
    """
    Compare predictions at different time horizons for a sensor
    
    Args:
        predictions: (12, num_nodes) prediction array
        sensor_id: Sensor index
        horizons: List of time horizons in minutes
        
    Returns:
        Plotly figure
    """
    horizon_map = {5: 0, 15: 2, 30: 5, 60: 11}
    
    speeds = []
    labels = []
    colors = []
    
    for h in horizons:
        timestep = horizon_map.get(h, 0)
        speed = predictions[timestep, sensor_id]
        speeds.append(speed)
        labels.append(f"{h} min")
        
        color, _ = get_traffic_color(speed)
        colors.append(color)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=speeds,
        marker_color=colors,
        text=[f"{s:.1f}" for s in speeds],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Sensor {sensor_id}: Speed Predictions at Different Horizons",
        xaxis_title="Time Ahead",
        yaxis_title="Speed (mph)",
        height=350,
        showlegend=False
    )
    
    return fig


def create_network_heatmap(predictions, timestep=0, num_sensors=50):
    """
    Create heatmap showing speed across sensors
    
    Args:
        predictions: (12, num_nodes) prediction array
        timestep: Which timestep to display
        num_sensors: Number of sensors to show (for readability)
        
    Returns:
        Plotly figure
    """
    speeds = predictions[timestep, :num_sensors]
    
    # Create color scale
    colors = [get_traffic_color(s)[0] for s in speeds]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(range(num_sensors)),
        y=speeds,
        marker_color=colors,
        hovertemplate='Sensor %{x}<br>Speed: %{y:.1f} mph<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Network-Wide Speed Distribution (First {num_sensors} Sensors)",
        xaxis_title="Sensor ID",
        yaxis_title="Speed (mph)",
        height=350,
        showlegend=False
    )
    
    return fig
