"""
Data loader for multimodal navigation dataset with adversarial IRL support.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import pickle

# Optional Open3D import for point cloud processing
# We disable importing Open3D by default to avoid segfaults on some macOS setups.
# To enable, set environment variable ENABLE_OPEN3D=1 before running.
ENABLE_OPEN3D = os.environ.get("ENABLE_OPEN3D", "0") == "1"
HAS_OPEN3D = False
if ENABLE_OPEN3D:
    try:
        import open3d as o3d
        HAS_OPEN3D = True
    except Exception:
        HAS_OPEN3D = False
        print("Warning: Open3D failed to import. Point cloud processing will be disabled.")
else:
    print("Info: Open3D disabled by default. Set ENABLE_OPEN3D=1 to enable point cloud processing.")


class MultimodalNavigationDataset(Dataset):
    """
    Dataset class for loading multimodal sensor data for autonomous navigation.
    Supports camera, LiDAR, radar, and GPS/IMU data.
    """
    
    def __init__(self, data_path: str, config: Dict, is_expert: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset directory
            config: Configuration dictionary
            is_expert: Whether this is expert demonstration data
        """
        self.data_path = Path(data_path)
        self.config = config
        self.is_expert = is_expert
        
        # Data loading parameters
        self.camera_size = config.get('camera_size', (224, 224))
        self.max_lidar_points = config.get('max_lidar_points', 16384)
        self.max_radar_detections = config.get('max_radar_detections', 64)
        
        # Load dataset index
        self.data_index = self._load_data_index()
        
        print(f"Loaded {len(self.data_index)} samples from {data_path}")
    
    def _load_data_index(self) -> List[Dict]:
        """Load the dataset index file."""
        index_file = self.data_path / "dataset_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        else:
            # Create index from directory structure
            return self._create_data_index()
    
    def _create_data_index(self) -> List[Dict]:
        """Create dataset index from directory structure."""
        data_index = []
        
        # Look for trajectory directories
        trajectory_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        for traj_dir in trajectory_dirs:
            # Get all timesteps in this trajectory
            camera_files = sorted((traj_dir / "camera").glob("*.jpg"))
            
            for i, camera_file in enumerate(camera_files):
                timestamp = camera_file.stem
                
                sample = {
                    'trajectory_id': traj_dir.name,
                    'timestamp': timestamp,
                    'camera_path': str(camera_file),
                    'lidar_path': str(traj_dir / "lidar" / f"{timestamp}.pcd"),
                    'radar_path': str(traj_dir / "radar" / f"{timestamp}.json"),
                    'gps_path': str(traj_dir / "gps" / f"{timestamp}.json"),
                    'action_path': str(traj_dir / "actions" / f"{timestamp}.json")
                }
                
                # Check if all files exist
                if all(Path(path).exists() for path in [
                    sample['camera_path'], 
                    sample['lidar_path'], 
                    sample['gps_path'], 
                    sample['action_path']
                ]):
                    data_index.append(sample)
        
        return data_index
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing multimodal data and actions
        """
        sample_info = self.data_index[idx]
        
        # Load multimodal data
        multimodal_data = {}
        
        # Load camera image
        camera_data = self._load_camera_data(sample_info['camera_path'])
        if camera_data is not None:
            multimodal_data['camera'] = camera_data
        
        # Load LiDAR point cloud
        lidar_data = self._load_lidar_data(sample_info['lidar_path'])
        if lidar_data is not None:
            multimodal_data['lidar'] = lidar_data
        
        # Load radar data
        radar_data = self._load_radar_data(sample_info['radar_path'])
        if radar_data is not None:
            multimodal_data['radar'] = radar_data
        
        # Load GPS/IMU data
        gps_data = self._load_gps_data(sample_info['gps_path'])
        if gps_data is not None:
            multimodal_data['gps'] = gps_data
        
        # Load action data
        action_data = self._load_action_data(sample_info['action_path'])
        
        return {
            'multimodal': multimodal_data,
            'actions': action_data,
            'trajectory_id': sample_info['trajectory_id'],
            'timestamp': sample_info['timestamp']
        }
    
    def _load_camera_data(self, camera_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess camera image."""
        try:
            if not Path(camera_path).exists():
                return None
                
            # Load image
            image = cv2.imread(camera_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.camera_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor and transpose to CHW format
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            
            return image_tensor
            
        except Exception as e:
            print(f"Error loading camera data from {camera_path}: {e}")
            return None
    
    def _load_lidar_data(self, lidar_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess LiDAR point cloud."""
        try:
            if not Path(lidar_path).exists():
                return None
            
            if not HAS_OPEN3D:
                # Fallback: return dummy point cloud data
                return np.zeros((self.max_lidar_points, 3), dtype=np.float32)
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(lidar_path)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                return None
            
            # Subsample or pad points
            if len(points) > self.max_lidar_points:
                # Random subsampling
                indices = np.random.choice(len(points), self.max_lidar_points, replace=False)
                points = points[indices]
            elif len(points) < self.max_lidar_points:
                # Pad with zeros
                padding = np.zeros((self.max_lidar_points - len(points), 3))
                points = np.vstack([points, padding])
            
            # Convert to tensor
            points_tensor = torch.from_numpy(points.astype(np.float32))
            
            return points_tensor
            
        except Exception as e:
            print(f"Error loading LiDAR data from {lidar_path}: {e}")
            return None
    
    def _load_radar_data(self, radar_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess radar data."""
        try:
            if not Path(radar_path).exists():
                return None
            
            with open(radar_path, 'r') as f:
                radar_data = json.load(f)
            
            # Extract radar detections
            detections = radar_data.get('detections', [])
            
            if len(detections) == 0:
                # Return zeros if no detections
                return torch.zeros(self.max_radar_detections, 4)
            
            # Convert to numpy array (range, azimuth, elevation, velocity)
            radar_array = []
            for detection in detections[:self.max_radar_detections]:
                radar_array.append([
                    detection.get('range', 0.0),
                    detection.get('azimuth', 0.0),
                    detection.get('elevation', 0.0),
                    detection.get('velocity', 0.0)
                ])
            
            # Pad if necessary
            while len(radar_array) < self.max_radar_detections:
                radar_array.append([0.0, 0.0, 0.0, 0.0])
            
            radar_tensor = torch.from_numpy(np.array(radar_array, dtype=np.float32))
            
            return radar_tensor
            
        except Exception as e:
            print(f"Error loading radar data from {radar_path}: {e}")
            return None
    
    def _load_gps_data(self, gps_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess GPS/IMU data."""
        try:
            if not Path(gps_path).exists():
                return None
            
            with open(gps_path, 'r') as f:
                gps_data = json.load(f)
            
            # Extract GPS/IMU features
            gps_features = [
                gps_data.get('latitude', 0.0),
                gps_data.get('longitude', 0.0),
                gps_data.get('altitude', 0.0),
                gps_data.get('heading', 0.0),
                gps_data.get('pitch', 0.0),
                gps_data.get('roll', 0.0),
                gps_data.get('velocity_x', 0.0),
                gps_data.get('velocity_y', 0.0),
                gps_data.get('velocity_z', 0.0)
            ]
            
            gps_tensor = torch.tensor(gps_features, dtype=torch.float32)
            
            return gps_tensor
            
        except Exception as e:
            print(f"Error loading GPS data from {gps_path}: {e}")
            return None
    
    def _load_action_data(self, action_path: str) -> torch.Tensor:
        """Load action data."""
        try:
            with open(action_path, 'r') as f:
                action_data = json.load(f)
            
            # Extract steering, throttle, brake
            actions = [
                action_data.get('steering', 0.0),
                action_data.get('throttle', 0.0),
                action_data.get('brake', 0.0)
            ]
            
            # Add additional actions if present
            for key in ['gear', 'handbrake', 'reverse']:
                if key in action_data:
                    actions.append(float(action_data[key]))
            
            action_tensor = torch.tensor(actions, dtype=torch.float32)
            
            return action_tensor
            
        except Exception as e:
            print(f"Error loading action data from {action_path}: {e}")
            # Return default action (no steering, no throttle, no brake)
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)


class SyntheticNavigationDataset(Dataset):
    """
    Synthetic dataset generator for testing and augmentation.
    Creates realistic navigation scenarios with procedural generation.
    """
    
    def __init__(self, config: Dict, num_samples: int = 1000):
        self.config = config
        self.num_samples = num_samples
        self.camera_size = config.get('camera_size', (224, 224))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a synthetic sample."""
        np.random.seed(idx)  # For reproducibility
        
        # Generate synthetic multimodal data
        multimodal_data = {}
        
        # Synthetic camera image (simple road scene)
        camera_image = self._generate_synthetic_camera()
        multimodal_data['camera'] = camera_image
        
        # Synthetic LiDAR data
        lidar_points = self._generate_synthetic_lidar()
        multimodal_data['lidar'] = lidar_points
        
        # Synthetic radar data
        radar_detections = self._generate_synthetic_radar()
        multimodal_data['radar'] = radar_detections
        
        # Synthetic GPS data
        gps_data = self._generate_synthetic_gps()
        multimodal_data['gps'] = gps_data
        
        # Synthetic actions (random driving behavior)
        actions = self._generate_synthetic_actions()
        
        return {
            'multimodal': multimodal_data,
            'actions': actions,
            'trajectory_id': f'synthetic_{idx // 100}',
            'timestamp': f'synthetic_{idx}'
        }
    
    def _generate_synthetic_camera(self) -> torch.Tensor:
        """Generate a synthetic camera image."""
        # Create a simple road scene
        image = np.ones((*self.camera_size, 3), dtype=np.float32) * 0.5  # Gray background
        
        # Add road
        road_width = self.camera_size[1] // 3
        road_start = (self.camera_size[1] - road_width) // 2
        image[self.camera_size[0]//2:, road_start:road_start+road_width] = [0.3, 0.3, 0.3]  # Dark gray road
        
        # Add lane markings
        lane_y = self.camera_size[1] // 2
        image[self.camera_size[0]//2:, lane_y-2:lane_y+2] = [1.0, 1.0, 1.0]  # White line
        
        # Add some random noise
        noise = np.random.normal(0, 0.02, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        return image_tensor
    
    def _generate_synthetic_lidar(self) -> torch.Tensor:
        """Generate synthetic LiDAR point cloud."""
        num_points = self.config.get('max_lidar_points', 16384)
        
        # Generate points in a semi-realistic pattern
        # Ground points
        ground_points = []
        for _ in range(num_points // 2):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(0, 50)
            z = np.random.uniform(-0.5, 0.5)  # Slight height variation
            ground_points.append([x, y, z])
        
        # Object points (cars, buildings, etc.)
        object_points = []
        for _ in range(num_points // 2):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(5, 30)
            z = np.random.uniform(0, 3)  # Above ground
            object_points.append([x, y, z])
        
        all_points = ground_points + object_points
        points_array = np.array(all_points, dtype=np.float32)
        
        return torch.from_numpy(points_array)
    
    def _generate_synthetic_radar(self) -> torch.Tensor:
        """Generate synthetic radar detections."""
        max_detections = self.config.get('max_radar_detections', 64)
        num_detections = np.random.randint(0, max_detections // 2)
        
        detections = []
        for _ in range(num_detections):
            range_val = np.random.uniform(5, 100)  # 5-100 meters
            azimuth = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
            elevation = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
            velocity = np.random.uniform(-20, 20)  # -20 to 20 m/s
            
            detections.append([range_val, azimuth, elevation, velocity])
        
        # Pad with zeros
        while len(detections) < max_detections:
            detections.append([0.0, 0.0, 0.0, 0.0])
        
        return torch.tensor(detections, dtype=torch.float32)
    
    def _generate_synthetic_gps(self) -> torch.Tensor:
        """Generate synthetic GPS/IMU data."""
        gps_data = [
            np.random.uniform(37.0, 38.0),  # Latitude (SF Bay Area)
            np.random.uniform(-122.5, -121.5),  # Longitude
            np.random.uniform(0, 100),  # Altitude
            np.random.uniform(0, 2*np.pi),  # Heading
            np.random.uniform(-0.1, 0.1),  # Pitch
            np.random.uniform(-0.1, 0.1),  # Roll
            np.random.uniform(-10, 10),  # Velocity X
            np.random.uniform(0, 30),  # Velocity Y (forward)
            np.random.uniform(-1, 1)  # Velocity Z
        ]
        
        return torch.tensor(gps_data, dtype=torch.float32)
    
    def _generate_synthetic_actions(self) -> torch.Tensor:
        """Generate synthetic driving actions."""
        # Random but plausible driving behavior
        steering = np.random.normal(0, 0.2)  # Slight steering bias toward straight
        steering = np.clip(steering, -1, 1)
        
        throttle = np.random.uniform(0, 0.8)  # Usually some throttle
        brake = np.random.uniform(0, 0.2) if throttle < 0.3 else 0  # Brake when low throttle
        
        return torch.tensor([steering, throttle, brake], dtype=torch.float32)
