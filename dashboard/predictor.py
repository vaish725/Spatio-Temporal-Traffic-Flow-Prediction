"""
DCRNN Model Predictor
Loads trained model and generates traffic predictions
"""
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.dcrnn import DCRNN


class TrafficPredictor:
    """
    Traffic prediction using trained DCRNN model
    """
    
    def __init__(self, checkpoint_path, data_path='data/pems_bay_processed.npz', device=None):
        """
        Initialize predictor
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_path: Path to preprocessed data (for normalization stats)
            device: torch device (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data for normalization and adjacency matrix
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        
        self.mean = float(data['mean'])
        self.std = float(data['std'])
        self.P_fwd = torch.FloatTensor(data['P_fwd']).to(self.device)
        self.P_bwd = torch.FloatTensor(data['P_bwd']).to(self.device)
        
        self.num_nodes = self.P_fwd.shape[0]
        print(f"Loaded data: {self.num_nodes} sensors, mean={self.mean:.2f}, std={self.std:.2f}")
        
        # Load most recent test data for live predictions
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get state dict
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        # Infer model config from checkpoint weights
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            hidden_dim = config.get('hidden_dim', 64)
            num_layers = config.get('num_layers', 2)
        else:
            # Infer from state_dict
            hidden_dim = state_dict['encoder.layers.0.conv_xz.bias'].shape[0]
            num_layers = sum(1 for k in state_dict.keys() if 'encoder.layers.' in k and '.conv_xz.weight' in k)
            print(f"Inferred config: hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # Create model
        model = DCRNN(
            input_dim=1,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers
        ).to(self.device)
        
        # Load weights
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def get_sample_timestamp(self, sample_idx=-1):
        """
        Get the timestamp info for a sample
        
        Args:
            sample_idx: Index in test set (default: -1 for most recent)
            
        Returns:
            dict with timestamp information
        """
        # Convert negative index to positive
        actual_idx = sample_idx if sample_idx >= 0 else len(self.X_test) + sample_idx
        
        # PEMS-BAY data: 5-minute intervals starting from 2017-01-01
        # Total dataset has train + val + test samples
        # Test set is the last portion
        from datetime import datetime, timedelta
        
        # Approximate the timestamp (assuming sequential 5-min intervals)
        # Each sample represents a 5-minute window
        start_date = datetime(2017, 1, 1)
        
        # Estimate offset (train: ~70%, val: ~15%, test: ~15% of total data)
        # With 52,116 total samples (as per PEMS-BAY dataset)
        total_samples = 52116
        train_samples = int(total_samples * 0.7)
        val_samples = int(total_samples * 0.15)
        
        # Test sample offset from start
        sample_offset = train_samples + val_samples + actual_idx
        minutes_offset = sample_offset * 5
        
        current_time = start_date + timedelta(minutes=minutes_offset)
        
        return {
            'sample_idx': actual_idx,
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'date': current_time.strftime('%Y-%m-%d'),
            'time': current_time.strftime('%H:%M:%S'),
            'weekday': current_time.strftime('%A'),
            'total_test_samples': len(self.X_test),
            'datetime_obj': current_time
        }
    
    def get_test_date_range(self):
        """
        Get the date range of test data
        
        Returns:
            dict with start_date, end_date, start_datetime, end_datetime
        """
        first_sample = self.get_sample_timestamp(0)
        last_sample = self.get_sample_timestamp(-1)
        
        return {
            'start_date': first_sample['date'],
            'end_date': last_sample['date'],
            'start_time': first_sample['time'],
            'end_time': last_sample['time'],
            'start_datetime': first_sample['datetime_obj'],
            'end_datetime': last_sample['datetime_obj'],
            'total_samples': len(self.X_test)
        }
    
    def find_sample_by_datetime(self, target_date, target_time):
        """
        Find the sample index closest to the given date and time
        
        Args:
            target_date: Date string in format 'YYYY-MM-DD'
            target_time: Time string in format 'HH:MM:SS' or 'HH:MM'
            
        Returns:
            dict with sample_idx and actual datetime
        """
        from datetime import datetime, timedelta
        
        # Parse target datetime
        if len(target_time) == 5:  # HH:MM format
            target_time += ':00'
        
        target_dt = datetime.strptime(f"{target_date} {target_time}", '%Y-%m-%d %H:%M:%S')
        
        # Get test data range
        date_range = self.get_test_date_range()
        start_dt = date_range['start_datetime']
        end_dt = date_range['end_datetime']
        
        # Check if target is within range
        if target_dt < start_dt or target_dt > end_dt:
            return {
                'error': f"Date/time out of range. Test data spans {date_range['start_date']} to {date_range['end_date']}",
                'start_date': date_range['start_date'],
                'end_date': date_range['end_date']
            }
        
        # Calculate sample index (5-minute intervals)
        time_diff = target_dt - start_dt
        minutes_diff = time_diff.total_seconds() / 60
        sample_idx = int(minutes_diff / 5)
        
        # Clamp to valid range
        sample_idx = max(0, min(sample_idx, len(self.X_test) - 1))
        
        # Get actual timestamp for this sample
        actual_info = self.get_sample_timestamp(sample_idx)
        
        return {
            'sample_idx': sample_idx,
            'actual_datetime': actual_info['datetime_obj'],
            'actual_date': actual_info['date'],
            'actual_time': actual_info['time'],
            'weekday': actual_info['weekday']
        }
    
    def get_latest_data(self, sample_idx=-1):
        """
        Get latest historical data for prediction
        
        Args:
            sample_idx: Index in test set (default: -1 for most recent)
            
        Returns:
            Tuple of (historical_speeds, ground_truth_speeds)
        """
        historical = self.X_test[sample_idx]  # (12, num_nodes, 1)
        ground_truth = self.y_test[sample_idx]  # (12, num_nodes, 1)
        
        # Denormalize
        historical_mph = historical * self.std + self.mean
        ground_truth_mph = ground_truth * self.std + self.mean
        
        return historical_mph.squeeze(), ground_truth_mph.squeeze()
    
    @torch.no_grad()
    def predict(self, historical_data=None, sample_idx=-1):
        """
        Generate traffic predictions
        
        Args:
            historical_data: (12, num_nodes) array of speeds in mph
                            If None, uses latest test data
            sample_idx: Test sample to use if historical_data is None
            
        Returns:
            predictions: (12, num_nodes) array of predicted speeds in mph
        """
        # Get historical data
        if historical_data is None:
            historical_data, _ = self.get_latest_data(sample_idx)
        
        # Normalize
        historical_norm = (historical_data - self.mean) / self.std
        
        # Prepare input: (batch=1, seq_len=12, num_nodes, features=1)
        x = torch.FloatTensor(historical_norm).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Predict
        predictions_norm = self.model(x, self.P_fwd, self.P_bwd)  # (1, 12, num_nodes, 1)
        
        # Denormalize
        predictions = predictions_norm.cpu().numpy().squeeze() * self.std + self.mean
        
        return predictions  # (12, num_nodes)
    
    def predict_for_sensor(self, sensor_id, historical_data=None, sample_idx=-1):
        """
        Get predictions for a specific sensor
        
        Args:
            sensor_id: Sensor index (0 to num_nodes-1)
            historical_data: Optional historical data
            sample_idx: Test sample index
            
        Returns:
            dict with historical, predicted, and ground_truth speeds
        """
        # Get data
        if historical_data is None:
            historical, ground_truth = self.get_latest_data(sample_idx)
        else:
            historical = historical_data
            _, ground_truth = self.get_latest_data(sample_idx)
        
        # Get predictions
        predictions = self.predict(historical, sample_idx)
        
        return {
            'historical': historical[:, sensor_id],  # (12,)
            'predicted': predictions[:, sensor_id],  # (12,)
            'ground_truth': ground_truth[:, sensor_id],  # (12,)
            'sensor_id': sensor_id
        }
    
    def get_network_summary(self, predictions):
        """
        Get network-wide traffic summary
        
        Args:
            predictions: (12, num_nodes) prediction array
            
        Returns:
            Dictionary with network statistics
        """
        # Use first timestep (5 min ahead)
        current_speeds = predictions[0, :]
        
        # Calculate statistics
        avg_speed = np.mean(current_speeds)
        min_speed = np.min(current_speeds)
        max_speed = np.max(current_speeds)
        
        # Count sensors by traffic level
        free_flow = np.sum(current_speeds >= 50)
        moderate = np.sum((current_speeds >= 35) & (current_speeds < 50))
        slow = np.sum((current_speeds >= 20) & (current_speeds < 35))
        congested = np.sum(current_speeds < 20)
        
        # Overall health (0-100)
        health_score = (avg_speed / 65.0) * 100
        
        return {
            'avg_speed': float(avg_speed),
            'min_speed': float(min_speed),
            'max_speed': float(max_speed),
            'health_score': float(health_score),
            'free_flow_count': int(free_flow),
            'moderate_count': int(moderate),
            'slow_count': int(slow),
            'congested_count': int(congested),
            'total_sensors': self.num_nodes
        }
    
    def get_top_congested_sensors(self, predictions, n=10):
        """
        Get sensors with worst predicted traffic
        
        Args:
            predictions: (12, num_nodes) prediction array
            n: Number of sensors to return
            
        Returns:
            List of (sensor_id, speed) tuples
        """
        # Use first timestep
        speeds = predictions[0, :]
        
        # Get indices of slowest sensors
        slowest_indices = np.argsort(speeds)[:n]
        
        return [(int(idx), float(speeds[idx])) for idx in slowest_indices]
    
    def compare_sensors(self, sensor_ids, historical_data=None, sample_idx=-1):
        """
        Compare predictions for multiple sensors
        
        Args:
            sensor_ids: List of sensor indices
            historical_data: Optional historical data
            sample_idx: Test sample index
            
        Returns:
            Dictionary mapping sensor_id to prediction data
        """
        # Get full predictions
        if historical_data is None:
            historical, ground_truth = self.get_latest_data(sample_idx)
        else:
            historical = historical_data
            _, ground_truth = self.get_latest_data(sample_idx)
        
        predictions = self.predict(historical, sample_idx)
        
        # Extract data for each sensor
        result = {}
        for sensor_id in sensor_ids:
            result[sensor_id] = {
                'historical': historical[:, sensor_id],
                'predicted': predictions[:, sensor_id],
                'ground_truth': ground_truth[:, sensor_id]
            }
        
        return result


def load_predictor(checkpoint_path=None, data_path='data/pems_bay_processed.npz'):
    """
    Convenience function to load predictor with default checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint (default: auto-detect)
        data_path: Path to data file
        
    Returns:
        TrafficPredictor instance
    """
    # Auto-detect checkpoint if not provided
    if checkpoint_path is None:
        possible_checkpoints = [
            'checkpoints_colab/best_model_optimized.pt',  # 10K model - OPTIMAL (1.930 mph test MAE)
            'checkpoints_colab/best_model_20k.pt',        # 20K model backup
            'checkpoints/best_model.pt',
            'checkpoints_colab/best_model.pt',
            'results/latest_results/checkpoints/best_model.pt'
        ]
        
        for path in possible_checkpoints:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No trained model found. Please train a model first.\n"
                "Expected locations: " + ", ".join(possible_checkpoints)
            )
    
    print(f"Using checkpoint: {checkpoint_path}")
    return TrafficPredictor(checkpoint_path, data_path)
