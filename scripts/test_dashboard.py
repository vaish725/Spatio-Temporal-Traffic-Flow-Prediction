"""
Quick test script for the dashboard components
Run this to verify everything is working before launching the full dashboard
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🧪 Testing Dashboard Components")
print("=" * 50)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from dashboard.predictor import load_predictor
    from dashboard.utils import get_traffic_color, load_sensor_metadata
    from dashboard.visualization import create_traffic_map, create_time_series_plot
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: Load model
print("\n2. Testing model loading...")
try:
    predictor = load_predictor()
    print(f"   ✅ Model loaded: {predictor.num_nodes} sensors, device={predictor.device}")
except Exception as e:
    print(f"   ❌ Model loading error: {e}")
    print("   💡 Make sure you have trained the model first")
    sys.exit(1)

# Test 3: Load sensor metadata
print("\n3. Testing sensor metadata...")
try:
    sensors = load_sensor_metadata()
    print(f"   ✅ Loaded {len(sensors)} sensor locations")
except Exception as e:
    print(f"   ❌ Metadata error: {e}")
    sys.exit(1)

# Test 4: Generate predictions
print("\n4. Testing predictions...")
try:
    historical, ground_truth = predictor.get_latest_data()
    predictions = predictor.predict(historical)
    print(f"   ✅ Generated predictions: shape {predictions.shape}")
    print(f"      Historical avg: {historical.mean():.2f} mph")
    print(f"      Predicted avg: {predictions.mean():.2f} mph")
except Exception as e:
    print(f"   ❌ Prediction error: {e}")
    sys.exit(1)

# Test 5: Network summary
print("\n5. Testing network summary...")
try:
    summary = predictor.get_network_summary(predictions)
    print(f"   ✅ Network health: {summary['health_score']:.1f}/100")
    print(f"      Average speed: {summary['avg_speed']:.1f} mph")
    print(f"      Congested sensors: {summary['congested_count']}/{summary['total_sensors']}")
except Exception as e:
    print(f"   ❌ Summary error: {e}")
    sys.exit(1)

# Test 6: Color coding
print("\n6. Testing traffic colors...")
try:
    test_speeds = [60, 45, 30, 15]
    expected_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
    for speed, expected in zip(test_speeds, expected_colors):
        color, status = get_traffic_color(speed)
        assert color == expected, f"Color mismatch for {speed} mph"
    print("   ✅ Color coding working correctly")
except Exception as e:
    print(f"   ❌ Color coding error: {e}")
    sys.exit(1)

# Test 7: Visualization components
print("\n7. Testing visualization components...")
try:
    # Test map creation
    fig_map = create_traffic_map(sensors, predictions, timestep=0)
    print(f"   ✅ Map created: {len(fig_map.data)} traces")
    
    # Test time series
    sensor_id = 0
    fig_ts = create_time_series_plot(
        historical[:, sensor_id],
        predictions[:, sensor_id],
        ground_truth[:, sensor_id]
    )
    print(f"   ✅ Time series plot created: {len(fig_ts.data)} traces")
except Exception as e:
    print(f"   ❌ Visualization error: {e}")
    sys.exit(1)

# Test 8: Sensor comparison
print("\n8. Testing sensor comparison...")
try:
    sensor_ids = [0, 10, 50]
    comparison = predictor.compare_sensors(sensor_ids, historical)
    print(f"   ✅ Compared {len(comparison)} sensors")
    for sid in sensor_ids:
        avg_pred = comparison[sid]['predicted'].mean()
        print(f"      Sensor {sid}: {avg_pred:.1f} mph (predicted avg)")
except Exception as e:
    print(f"   ❌ Comparison error: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("\n🚀 Dashboard is ready to launch:")
print("   streamlit run app.py")
print("\nOr use the quick start script:")
print("   ./run_dashboard.sh")
print("=" * 50)
