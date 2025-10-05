"""
Fixed Synthetic Data Loader for Adversarial IRL Navigation System
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class FixedSyntheticDataset(Dataset):
    """
    Fixed synthetic dataset that generates proper tensor dimensions.
    """
    
    def __init__(self, config: Dict, num_samples: int = 1000):
        """
        Initialize the synthetic dataset.
        
        Args:
            config: Configuration dictionary with fixed dimensions
            num_samples: Number of synthetic samples to generate
        """
        self.config = config
        self.num_samples = num_samples
        
        # Fixed dimensions from config
        self.camera_size = config.get('camera_size', (3, 224, 224))
        self.max_lidar_points = config.get('max_lidar_points', 1024)
        self.max_radar_points = config.get('max_radar_points', 32)
        self.gps_dim = config.get('gps_dim', 9)
        self.action_dim = config.get('action_dim', 4)
        
        # Pre-generate all data for consistency
        self.data = self._generate_all_data()
        
    def _generate_all_data(self) -> List[Dict]:
        """Pre-generate all synthetic data samples."""
        print(f"Generating {self.num_samples} synthetic samples...")
        
        data = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(self.num_samples):
            # Generate multimodal sensor data
            multimodal_data = self._generate_multimodal_sample()
            
            # Generate expert actions (realistic driving behavior)
            actions = self._generate_expert_actions()
            
            # Create sample dictionary
            sample = {
                'multimodal': multimodal_data,
                'actions': actions,
                'trajectory_id': i // 20,  # 20 samples per trajectory
                'step_id': i % 20,
                'timestamp': float(i)
            }
            
            data.append(sample)
        
        print(f"Successfully generated {len(data)} samples")
        return data
    
    def _generate_multimodal_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a single multimodal sensor sample with fixed dimensions."""
        
        # Camera data: RGB image
        camera = torch.randn(self.camera_size, dtype=torch.float32)
        
        # LiDAR data: Point cloud (x, y, z)
        lidar = torch.randn(self.max_lidar_points, 3, dtype=torch.float32)
        # Add some realistic structure (ground plane, obstacles)
        lidar[:100, 2] = torch.abs(lidar[:100, 2]) * 0.1  # Ground points
        lidar[100:200, 2] = torch.abs(lidar[100:200, 2]) * 2.0 + 1.0  # Obstacles
        
        # Radar data: [range, azimuth, elevation, velocity]
        radar = torch.randn(self.max_radar_points, 4, dtype=torch.float32)
        radar[:, 0] = torch.abs(radar[:, 0]) * 50.0  # Range 0-50m
        radar[:, 1] = radar[:, 1] * np.pi  # Azimuth -π to π
        radar[:, 2] = radar[:, 2] * 0.5    # Elevation -0.5 to 0.5
        radar[:, 3] = radar[:, 3] * 20.0   # Velocity -20 to 20 m/s
        
        # GPS/IMU data: [lat, lon, alt, heading, pitch, roll, vel_x, vel_y, vel_z]
        gps = torch.randn(self.gps_dim, dtype=torch.float32)
        gps[0] = gps[0] * 0.01 + 37.7749   # Latitude (around San Francisco)
        gps[1] = gps[1] * 0.01 - 122.4194  # Longitude
        gps[2] = torch.abs(gps[2]) * 100    # Altitude 0-100m
        gps[3] = gps[3] * np.pi             # Heading -π to π
        gps[4:6] = gps[4:6] * 0.1           # Pitch, roll ±0.1 rad
        gps[6:9] = gps[6:9] * 10.0          # Velocities ±10 m/s
        
        return {
            'camera': camera,
            'lidar': lidar,
            'radar': radar,
            'gps': gps
        }
    
    def _generate_expert_actions(self) -> torch.Tensor:
        """Generate realistic expert driving actions."""
        
        # Expert behavior patterns
        steering = np.random.normal(0.0, 0.1)  # Small steering corrections
        steering = np.clip(steering, -1.0, 1.0)
        
        throttle = np.random.uniform(0.2, 0.7)  # Moderate throttle
        brake = 0.0 if throttle > 0.3 else np.random.uniform(0.0, 0.3)
        gear = 1.0  # Forward gear
        
        actions = torch.tensor([steering, throttle, brake, gear], dtype=torch.float32)
        return actions
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        return self.data[idx]
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of samples."""
        batch_data = {'multimodal': {}, 'actions': []}
        
        # Initialize multimodal batch containers
        batch_data['multimodal']['camera'] = []
        batch_data['multimodal']['lidar'] = []
        batch_data['multimodal']['radar'] = []
        batch_data['multimodal']['gps'] = []
        
        for idx in indices:
            sample = self.data[idx]
            batch_data['multimodal']['camera'].append(sample['multimodal']['camera'])
            batch_data['multimodal']['lidar'].append(sample['multimodal']['lidar'])
            batch_data['multimodal']['radar'].append(sample['multimodal']['radar'])
            batch_data['multimodal']['gps'].append(sample['multimodal']['gps'])
            batch_data['actions'].append(sample['actions'])
        
        # Stack tensors
        batch_data['multimodal']['camera'] = torch.stack(batch_data['multimodal']['camera'])
        batch_data['multimodal']['lidar'] = torch.stack(batch_data['multimodal']['lidar'])
        batch_data['multimodal']['radar'] = torch.stack(batch_data['multimodal']['radar'])
        batch_data['multimodal']['gps'] = torch.stack(batch_data['multimodal']['gps'])
        batch_data['actions'] = torch.stack(batch_data['actions'])
        
        return batch_data


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    
    # Separate multimodal data and actions
    multimodal_batch = {
        'camera': torch.stack([item['multimodal']['camera'] for item in batch]),
        'lidar': torch.stack([item['multimodal']['lidar'] for item in batch]),
        'radar': torch.stack([item['multimodal']['radar'] for item in batch]),
        'gps': torch.stack([item['multimodal']['gps'] for item in batch])
    }
    
    actions_batch = torch.stack([item['actions'] for item in batch])
    
    return {
        'multimodal': multimodal_batch,
        'actions': actions_batch
    }


if __name__ == "__main__":
    # Test the fixed synthetic dataset
    from config.fixed_config import get_config
    
    config = get_config()
    dataset = FixedSyntheticDataset(config, num_samples=10)
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test a single sample
    sample = dataset[0]
    print("Sample structure:")
    print(f"  Multimodal keys: {list(sample['multimodal'].keys())}")
    
    for key, tensor in sample['multimodal'].items():
        print(f"  {key}: {tensor.shape}")
    
    print(f"  Actions: {sample['actions'].shape}")
    
    # Test batch creation
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    print("\nBatch shapes:")
    for key, tensor in batch['multimodal'].items():
        print(f"  {key}: {tensor.shape}")
    print(f"  Actions: {batch['actions'].shape}")
    
    print("\n✅ Fixed synthetic dataset test completed successfully!")
