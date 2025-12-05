"""
Utility functions for the traffic prediction dashboard
"""
import numpy as np
import json
import os
from datetime import datetime, timedelta


def get_traffic_color(speed_mph):
    """
    Get traffic color based on speed (Google Maps style)
    
    Args:
        speed_mph: Speed in miles per hour
        
    Returns:
        Color string and status
    """
    if speed_mph >= 50:
        return "#4CAF50", "Free Flow"  # Green
    elif speed_mph >= 35:
        return "#FFC107", "Moderate"  # Yellow
    elif speed_mph >= 20:
        return "#FF9800", "Slow"  # Orange
    else:
        return "#F44336", "Congested"  # Red


def get_speed_category(speed_mph):
    """Get speed category for display"""
    if speed_mph >= 50:
        return "🟢 Free Flow"
    elif speed_mph >= 35:
        return "🟡 Moderate Traffic"
    elif speed_mph >= 20:
        return "🟠 Slow Traffic"
    else:
        return "🔴 Congested"


def calculate_travel_time(distance_miles, speed_mph):
    """Calculate estimated travel time"""
    if speed_mph == 0:
        return "N/A"
    time_hours = distance_miles / speed_mph
    time_minutes = time_hours * 60
    return f"{int(time_minutes)} min"


def format_speed(speed_mph):
    """Format speed for display"""
    return f"{speed_mph:.1f} mph"


def get_time_labels(current_time=None, num_steps=12, interval_minutes=5):
    """
    Generate time labels for predictions
    
    Args:
        current_time: Starting datetime (default: now)
        num_steps: Number of time steps
        interval_minutes: Minutes between steps
        
    Returns:
        List of time strings
    """
    if current_time is None:
        current_time = datetime.now()
    
    times = []
    for i in range(num_steps):
        future_time = current_time + timedelta(minutes=i * interval_minutes)
        times.append(future_time.strftime("%H:%M"))
    
    return times


def get_timestep_for_horizon(minutes_ahead):
    """
    Convert time horizon to timestep index
    5 min intervals: timestep = minutes / 5 - 1
    
    Args:
        minutes_ahead: 5, 15, 30, or 60 minutes
        
    Returns:
        Timestep index (0-11)
    """
    horizon_map = {
        5: 0,    # 5 min ahead -> timestep 0
        15: 2,   # 15 min ahead -> timestep 2
        30: 5,   # 30 min ahead -> timestep 5
        60: 11   # 60 min ahead -> timestep 11
    }
    return horizon_map.get(minutes_ahead, 0)


def calculate_congestion_score(speeds):
    """
    Calculate overall congestion score (0-100)
    
    Args:
        speeds: Array of speeds in mph
        
    Returns:
        Congestion score (higher = more congestion)
    """
    avg_speed = np.mean(speeds)
    # Normalize: 0 mph = 100% congestion, 65 mph = 0% congestion
    max_speed = 65.0
    congestion = (1 - (avg_speed / max_speed)) * 100
    return max(0, min(100, congestion))


def get_congestion_emoji(score):
    """Get emoji based on congestion score"""
    if score < 25:
        return "✅"
    elif score < 50:
        return "⚠️"
    elif score < 75:
        return "🚨"
    else:
        return "🔴"


def load_sensor_metadata(data_dir='data'):
    """
    Load or generate sensor metadata
    
    Returns:
        Dictionary with sensor information
    """
    metadata_file = os.path.join(data_dir, 'sensor_metadata.json')
    
    # Try to load existing metadata
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    # Generate default metadata for PEMS-BAY (325 sensors)
    # These are approximate Bay Area highway locations
    return generate_default_sensor_locations()


def generate_default_sensor_locations(num_sensors=325):
    """
    Generate realistic Bay Area sensor locations
    Distributed along major highways (I-80, I-880, I-280, US-101)
    """
    sensors = {}
    
    # Bay Area bounding box
    lat_min, lat_max = 37.3, 38.0
    lon_min, lon_max = -122.5, -121.9
    
    # Generate sensors along major corridors
    for i in range(num_sensors):
        # Create corridors (4 major highways)
        corridor = i % 4
        progress = (i // 4) / (num_sensors // 4)
        
        if corridor == 0:  # I-80 (East-West, North)
            lat = 37.8 + (progress * 0.1) + np.random.normal(0, 0.01)
            lon = -122.4 + (progress * 0.5) + np.random.normal(0, 0.02)
            highway = "I-80"
        elif corridor == 1:  # I-880 (North-South, East)
            lat = 37.4 + (progress * 0.5) + np.random.normal(0, 0.02)
            lon = -122.1 + (progress * 0.1) + np.random.normal(0, 0.01)
            highway = "I-880"
        elif corridor == 2:  # I-280 (North-South, West)
            lat = 37.3 + (progress * 0.5) + np.random.normal(0, 0.02)
            lon = -122.3 + (progress * 0.2) + np.random.normal(0, 0.01)
            highway = "I-280"
        else:  # US-101 (North-South, Central)
            lat = 37.3 + (progress * 0.6) + np.random.normal(0, 0.02)
            lon = -122.2 + (progress * 0.15) + np.random.normal(0, 0.01)
            highway = "US-101"
        
        sensors[str(i)] = {
            'id': i,
            'name': f"Sensor {i}",
            'highway': highway,
            'lat': float(np.clip(lat, lat_min, lat_max)),
            'lon': float(np.clip(lon, lon_min, lon_max)),
            'direction': 'NB' if corridor in [1, 2, 3] else 'EB'
        }
    
    return sensors


def save_sensor_metadata(sensors, data_dir='data'):
    """Save sensor metadata to JSON"""
    metadata_file = os.path.join(data_dir, 'sensor_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(sensors, f, indent=2)
    return metadata_file
