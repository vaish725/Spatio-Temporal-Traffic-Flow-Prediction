#!/usr/bin/env python3
"""
Generate Realistic Synthetic Traffic Data

This creates synthetic traffic data with realistic patterns:
- Daily/weekly seasonality
- Rush hour peaks
- Weekend/weekday differences  
- Spatial correlations between nearby sensors
- Random incidents/congestion events

This is MORE COMPLEX than the simple mock data, allowing your model
to actually learn meaningful patterns.
"""

import numpy as np
import os
from tqdm import tqdm

print("""
╔══════════════════════════════════════════════════════════════╗
║     Realistic Synthetic Traffic Data Generator             ║
║                                                              ║
║  Creates complex traffic patterns for meaningful training   ║
╚══════════════════════════════════════════════════════════════╝
""")

def generate_realistic_traffic_data(
    num_nodes=325,
    num_days=180,  # 6 months
    interval_minutes=5,
    seed=42
):
    """
    Generate realistic traffic speed data with:
    - Daily patterns (rush hours)
    - Weekly patterns (weekends)
    - Spatial correlations
    - Random incidents
    """
    
    np.random.seed(seed)
    
    timesteps_per_day = 24 * 60 // interval_minutes  # 288 for 5-min intervals
    total_timesteps = num_days * timesteps_per_day
    
    print(f"\n📊 Generating data:")
    print(f"   Nodes: {num_nodes}")
    print(f"   Days: {num_days}")
    print(f"   Timesteps: {total_timesteps:,}")
    print(f"   Interval: {interval_minutes} minutes")
    
    # Initialize speed matrix (timesteps, nodes)
    speeds = np.zeros((total_timesteps, num_nodes))
    
    # Base speed for highways (mph)
    base_speed = 60.0
    
    # Create time features
    print("\n⏰ Creating time patterns...")
    for t in tqdm(range(total_timesteps)):
        day = t // timesteps_per_day
        time_of_day = (t % timesteps_per_day) / timesteps_per_day * 24  # 0-24 hours
        day_of_week = day % 7  # 0=Monday, 6=Sunday
        
        # Daily pattern: rush hours (7-9am, 5-7pm)
        morning_rush = np.exp(-((time_of_day - 8)**2) / 2)  # Peak at 8am
        evening_rush = np.exp(-((time_of_day - 18)**2) / 2)  # Peak at 6pm
        rush_factor = 1 - 0.4 * (morning_rush + evening_rush)  # 40% slowdown
        
        # Night pattern: less traffic, higher speeds
        night_factor = 1 + 0.2 * np.exp(-((time_of_day - 2)**2) / 8)  # Peak at 2am
        
        # Weekend pattern: less congestion
        is_weekend = day_of_week >= 5
        weekend_factor = 1.1 if is_weekend else 1.0
        
        # Combine all factors
        speed_factor = rush_factor * night_factor * weekend_factor
        
        # Apply to all nodes with some spatial variation
        for node in range(num_nodes):
            # Each node has slight variation in timing
            node_offset = np.sin(node * 0.1) * 0.5  # ±0.5 hour shift
            adjusted_speed = base_speed * speed_factor
            
            # Add some randomness
            noise = np.random.normal(0, 3)  # ±3 mph noise
            
            speeds[t, node] = max(10, adjusted_speed + noise)  # Min 10 mph
    
    # Add spatial correlations
    print("\n🗺️  Adding spatial correlations...")
    for node in tqdm(range(1, num_nodes)):
        # Smooth with previous node (highway continuity)
        alpha = 0.7
        speeds[:, node] = alpha * speeds[:, node] + (1-alpha) * speeds[:, node-1]
    
    # Add random incidents (sudden slowdowns)
    print("\n🚨 Simulating random incidents...")
    num_incidents = num_days * 3  # ~3 incidents per day
    for _ in range(num_incidents):
        incident_time = np.random.randint(0, total_timesteps - 12)
        incident_node = np.random.randint(0, num_nodes)
        incident_duration = np.random.randint(6, 24)  # 30 min - 2 hours
        incident_severity = np.random.uniform(0.3, 0.7)  # 30-70% speed reduction
        
        # Affect nearby nodes
        for dt in range(incident_duration):
            t = incident_time + dt
            if t < total_timesteps:
                for dn in range(-3, 4):  # ±3 nodes
                    n = incident_node + dn
                    if 0 <= n < num_nodes:
                        distance_factor = np.exp(-(dn**2) / 4)
                        speeds[t, n] *= (1 - incident_severity * distance_factor)
    
    # Add weekly trends (slight increase/decrease over months)
    print("\n📈 Adding long-term trends...")
    for node in range(num_nodes):
        trend = np.linspace(0, np.random.uniform(-5, 5), total_timesteps)
        speeds[:, node] += trend
    
    # Clip to reasonable range
    speeds = np.clip(speeds, 5, 75)  # 5-75 mph
    
    print(f"\n✅ Generated traffic data:")
    print(f"   Shape: {speeds.shape}")
    print(f"   Mean speed: {speeds.mean():.2f} mph")
    print(f"   Std: {speeds.std():.2f} mph")
    print(f"   Range: [{speeds.min():.2f}, {speeds.max():.2f}] mph")
    
    return speeds


def create_adjacency_matrix(num_nodes=325):
    """Create highway network adjacency matrix"""
    print(f"\n🛣️  Creating highway network topology...")
    
    # Simple linear highway with some branches
    adj = np.zeros((num_nodes, num_nodes))
    
    # Main highway: connect sequential nodes
    for i in range(num_nodes - 1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1
    
    # Add some branches/connections every ~20 nodes
    for i in range(0, num_nodes, 20):
        if i + 10 < num_nodes:
            adj[i, i+10] = 1
            adj[i+10, i] = 1
    
    # Add self-loops
    adj += np.eye(num_nodes)
    
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {int(adj.sum() - num_nodes)} (excluding self-loops)")
    print(f"   Avg degree: {adj.sum(axis=1).mean():.2f}")
    
    return adj


def save_processed_data(speeds, adj_matrix, output_file='data/pems_bay_processed.npz'):
    """Split and save data in the same format as real PEMS-BAY"""
    
    print("\n" + "="*70)
    print("PREPROCESSING DATA")
    print("="*70)
    
    # Normalize
    mean = speeds.mean()
    std = speeds.std()
    speeds_norm = (speeds - mean) / std
    
    print(f"\n🔢 Normalization:")
    print(f"   Mean: {mean:.2f}")
    print(f"   Std: {std:.2f}")
    
    # Create sequences
    T_in = 12
    T_out = 12
    timesteps, nodes = speeds_norm.shape
    num_samples = timesteps - T_in - T_out + 1
    
    print(f"\n📦 Creating sequences (T_in={T_in}, T_out={T_out})...")
    X = np.zeros((num_samples, T_in, nodes, 1))
    y = np.zeros((num_samples, T_out, nodes, 1))
    
    for i in tqdm(range(num_samples)):
        X[i, :, :, 0] = speeds_norm[i:i+T_in, :]
        y[i, :, :, 0] = speeds_norm[i+T_in:i+T_in+T_out, :]
    
    print(f"   Created {num_samples:,} sequences")
    
    # Split: 70% train, 10% val, 20% test
    train_split = int(0.7 * num_samples)
    val_split = int(0.8 * num_samples)
    
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]
    
    print(f"\n✂️  Splits:")
    print(f"   Train: {X_train.shape[0]:,} samples")
    print(f"   Val:   {X_val.shape[0]:,} samples")
    print(f"   Test:  {X_test.shape[0]:,} samples")
    
    # Create transition matrices
    P_fwd = adj_matrix / (adj_matrix.sum(axis=1, keepdims=True) + 1e-8)
    P_bwd = adj_matrix / (adj_matrix.sum(axis=0, keepdims=True) + 1e-8).T
    
    # Save
    os.makedirs('data', exist_ok=True)
    
    print(f"\n💾 Saving to: {output_file}")
    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        P_fwd=P_fwd,
        P_bwd=P_bwd,
        mean=mean,
        std=std,
        adj_matrix=adj_matrix
    )
    
    file_size = os.path.getsize(output_file) / 1e6
    print(f"✅ Saved! Size: {file_size:.2f} MB")
    
    return output_file


def main():
    print("\n🚀 Starting realistic data generation...\n")
    
    # Check if already exists
    output_file = 'data/pems_bay_processed.npz'
    if os.path.exists(output_file):
        print(f"⚠️  Found existing file: {output_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing data. Done!")
            return
    
    # Generate traffic data
    speeds = generate_realistic_traffic_data(
        num_nodes=325,
        num_days=180,  # 6 months (vs 180 days in real PEMS-BAY)
        interval_minutes=5
    )
    
    # Create network topology
    adj_matrix = create_adjacency_matrix(num_nodes=325)
    
    # Save processed data
    output_file = save_processed_data(speeds, adj_matrix)
    
    print("\n" + "="*70)
    print("✅ REALISTIC SYNTHETIC DATA READY!")
    print("="*70)
    print(f"\nOutput: {output_file}")
    print("\n📊 Comparison with your mock data:")
    print("   Mock data:       600 samples")
    print(f"   Realistic data:  {int(180*288*0.7):,}+ training samples")
    print("   Improvement:     ~60x more data with realistic patterns!")
    print("\n🎯 This data includes:")
    print("   ✓ Daily rush hour patterns")
    print("   ✓ Weekend/weekday differences")
    print("   ✓ Spatial correlations between sensors")
    print("   ✓ Random incident simulations")
    print("   ✓ Long-term trends")
    print("\n📝 Next steps:")
    print("   1. Update train.py to load this data")
    print("   2. Retrain: python3 scripts/train.py --epochs 100")
    print("   3. Your model will actually learn meaningful patterns!")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
