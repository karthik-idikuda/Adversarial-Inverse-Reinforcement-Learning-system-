"""
Adversarial IRL Agent Components for Testing

This file contains simplified versions of the agent components needed for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


# Simple MultimodalEncoder for testing
class MultimodalEncoder(nn.Module):
    """Simple multimodal encoder for testing purposes."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_dim = config.get('fusion_dim', 256)
        
        # Simple encoders
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64)
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        self.radar_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        self.gps_encoder = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 16, self.fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        batch_size = next(iter(x.values())).shape[0]
        features = []
        
        if 'camera' in x:
            cam_feat = self.camera_encoder(x['camera'])
            features.append(cam_feat)
        else:
            features.append(torch.zeros(batch_size, 64))
            
        if 'lidar' in x:
            lidar_data = x['lidar']
            if len(lidar_data.shape) == 3:
                lidar_data = lidar_data.reshape(batch_size, -1, 3)
                lidar_feat = self.lidar_encoder(lidar_data).mean(dim=1)
            else:
                lidar_feat = torch.zeros(batch_size, 64)
            features.append(lidar_feat)
        else:
            features.append(torch.zeros(batch_size, 64))
            
        if 'radar' in x:
            radar_data = x['radar']
            if len(radar_data.shape) == 3:
                radar_data = radar_data.reshape(batch_size, -1, 4)
                radar_feat = self.radar_encoder(radar_data).mean(dim=1)
            else:
                radar_feat = torch.zeros(batch_size, 32)
            features.append(radar_feat)
        else:
            features.append(torch.zeros(batch_size, 32))
            
        if 'gps' in x:
            gps_feat = self.gps_encoder(x['gps'])
            features.append(gps_feat)
        else:
            features.append(torch.zeros(batch_size, 16))
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class PolicyNetwork(nn.Module):
    """Policy network that produces actions based on state."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Action bounds
        self.action_bounds = {
            'steering': (-1.0, 1.0),  # Steering angle
            'throttle': (0.0, 1.0),   # Throttle
            'brake': (0.0, 1.0)       # Brake
        }
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            actions: Action tensor of shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        raw_actions = self.fc3(x)
        
        # Apply action bounds
        actions = torch.zeros_like(raw_actions)
        actions[:, 0] = torch.tanh(raw_actions[:, 0])  # steering [-1, 1]
        actions[:, 1] = torch.sigmoid(raw_actions[:, 1])  # throttle [0, 1]
        actions[:, 2] = torch.sigmoid(raw_actions[:, 2])  # brake [0, 1]
        
        return actions


class RewardNetwork(nn.Module):
    """Reward network that estimates rewards based on state-action pairs."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through reward network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            rewards: Reward tensor of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        rewards = self.fc3(x)
        
        return rewards


class DiscriminatorNetwork(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Network layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)
            
        Returns:
            probs: Probability tensor of shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        
        return probs


class AdversarialIRLAgent(nn.Module):
    """Complete Adversarial IRL agent."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Get dimensions
        self.fusion_dim = config.get('fusion_dim', 256)
        self.action_dim = config.get('action_dim', 3)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # Initialize components
        self.multimodal_encoder = MultimodalEncoder(config)
        self.policy_network = PolicyNetwork(self.fusion_dim, self.action_dim, self.hidden_dim)
        self.reward_network = RewardNetwork(self.fusion_dim, self.action_dim, self.hidden_dim)
        self.discriminator = DiscriminatorNetwork(self.fusion_dim, self.action_dim, self.hidden_dim)
        
    def encode_multimodal_state(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multimodal sensor data into a unified state representation.
        
        Args:
            multimodal_data: Dictionary of sensor data tensors
            
        Returns:
            state: Encoded state tensor
        """
        return self.multimodal_encoder(multimodal_data)
    
    def get_action(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate an action from multimodal sensor data.
        
        Args:
            multimodal_data: Dictionary of sensor data tensors
            
        Returns:
            action: Action tensor
        """
        state = self.encode_multimodal_state(multimodal_data)
        action = self.policy_network(state)
        return action
    
    def get_reward(self, multimodal_data: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """
        Estimate reward from state-action pair.
        
        Args:
            multimodal_data: Dictionary of sensor data tensors
            action: Action tensor
            
        Returns:
            reward: Estimated reward
        """
        state = self.encode_multimodal_state(multimodal_data)
        reward = self.reward_network(state, action)
        return reward
    
    def discriminate(self, multimodal_data: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between expert and generated actions.
        
        Args:
            multimodal_data: Dictionary of sensor data tensors
            action: Action tensor
            
        Returns:
            prob: Probability that action is from expert
        """
        state = self.encode_multimodal_state(multimodal_data)
        prob = self.discriminator(state, action)
        return prob


class NavigationController:
    """Navigation controller for autonomous driving."""
    
    def __init__(self, config: Dict, model_path: str = None):
        """
        Initialize the navigation controller.
        
        Args:
            config: Configuration dictionary
            model_path: Path to saved model
        """
        self.config = config
        
        # Initialize agent
        self.agent = AdversarialIRLAgent(config)
        
        # Load model if provided
        if model_path is not None:
            self.load_model(model_path)
        
        self.device = torch.device(config.get('device', 'cpu'))
        self.agent.to(self.device)
        self.agent.eval()
        
    def load_model(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to saved model
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        
    def process_sensor_data(self, raw_sensor_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Process raw sensor data into tensors for the model.
        
        Args:
            raw_sensor_data: Raw sensor data
            
        Returns:
            processed_data: Processed sensor data as tensors
        """
        processed_data = {}
        
        # Process camera data if available
        if 'camera' in raw_sensor_data:
            # Convert to torch tensor if necessary
            if isinstance(raw_sensor_data['camera'], np.ndarray):
                # Convert from (H, W, C) to (C, H, W)
                camera_data = raw_sensor_data['camera']
                if camera_data.shape[2] == 3:
                    camera_data = np.transpose(camera_data, (2, 0, 1))
                camera_tensor = torch.from_numpy(camera_data).float()
                # Normalize to [0, 1]
                if camera_tensor.max() > 1:
                    camera_tensor = camera_tensor / 255.0
            else:
                camera_tensor = raw_sensor_data['camera']
            
            # Add batch dimension if needed
            if len(camera_tensor.shape) == 3:
                camera_tensor = camera_tensor.unsqueeze(0)
            
            processed_data['camera'] = camera_tensor
        
        # Process LiDAR data if available
        if 'lidar' in raw_sensor_data:
            lidar_data = raw_sensor_data['lidar']
            if isinstance(lidar_data, np.ndarray):
                lidar_tensor = torch.from_numpy(lidar_data).float()
            else:
                lidar_tensor = lidar_data
            
            # Add batch dimension if needed
            if len(lidar_tensor.shape) == 2:
                lidar_tensor = lidar_tensor.unsqueeze(0)
            
            processed_data['lidar'] = lidar_tensor
        
        # Process radar data if available
        if 'radar' in raw_sensor_data:
            radar_data = raw_sensor_data['radar']
            
            # Convert structured radar data to tensor
            if isinstance(radar_data, list):
                # Convert list of dictionaries to numpy array
                max_points = self.config.get('max_radar_points', 64)
                radar_array = np.zeros((max_points, 4), dtype=np.float32)
                
                for i, detection in enumerate(radar_data[:max_points]):
                    radar_array[i, 0] = detection['range']
                    radar_array[i, 1] = detection['azimuth']
                    radar_array[i, 2] = detection['elevation']
                    radar_array[i, 3] = detection['velocity']
                
                radar_tensor = torch.from_numpy(radar_array).float()
            else:
                radar_tensor = radar_data
            
            # Add batch dimension if needed
            if len(radar_tensor.shape) == 2:
                radar_tensor = radar_tensor.unsqueeze(0)
            
            processed_data['radar'] = radar_tensor
        
        # Process GPS data if available
        if 'gps' in raw_sensor_data:
            gps_data = raw_sensor_data['gps']
            
            # Convert dictionary to tensor
            if isinstance(gps_data, dict):
                gps_array = np.array([
                    gps_data['latitude'],
                    gps_data['longitude'],
                    gps_data['altitude'],
                    gps_data['heading'],
                    gps_data['pitch'],
                    gps_data['roll'],
                    gps_data['velocity_x'],
                    gps_data['velocity_y'],
                    gps_data['velocity_z']
                ], dtype=np.float32)
                
                gps_tensor = torch.from_numpy(gps_array).float()
            else:
                gps_tensor = gps_data
            
            # Add batch dimension if needed
            if len(gps_tensor.shape) == 1:
                gps_tensor = gps_tensor.unsqueeze(0)
            
            processed_data['gps'] = gps_tensor
        
        return processed_data
    
    def predict_action(self, raw_sensor_data: Dict) -> Dict:
        """
        Predict action from raw sensor data.
        
        Args:
            raw_sensor_data: Raw sensor data
            
        Returns:
            control_commands: Dictionary of control commands
        """
        # Process sensor data
        processed_data = self.process_sensor_data(raw_sensor_data)
        
        # Move to device
        processed_data = {k: v.to(self.device) for k, v in processed_data.items()}
        
        # Get action
        with torch.no_grad():
            action = self.agent.get_action(processed_data)
        
        # Convert to control commands
        control_commands = {
            'steering': action[0, 0].item(),
            'throttle': action[0, 1].item(),
            'brake': action[0, 2].item(),
            'emergency_stop': False  # Default
        }
        
        # Safety check
        if 'camera' not in processed_data or 'lidar' not in processed_data:
            control_commands['emergency_stop'] = True
            control_commands['throttle'] = 0.0
            control_commands['brake'] = 1.0
        
        return control_commands


class SyntheticNavigationDataset(Dataset):
    """Generate synthetic navigation data for testing."""
    
    def __init__(self, config: Dict, num_samples: int = 100):
        """
        Initialize the synthetic dataset.
        
        Args:
            config: Configuration dictionary
            num_samples: Number of samples to generate
        """
        self.config = config
        self.num_samples = num_samples
        
        # Set dimensions from config
        self.camera_size = config.get('camera_size', (3, 224, 224))
        self.max_lidar_points = config.get('max_lidar_points', 1024)
        self.max_radar_points = config.get('max_radar_detections', 32)
        self.gps_dim = config.get('gps_dim', 9)
        self.action_dim = config.get('action_dim', 3)
        
        # Generate trajectories
        self.trajectories = self._generate_trajectories(10)
        
    def _generate_trajectories(self, num_trajectories: int) -> List[Dict]:
        """
        Generate synthetic trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
            
        Returns:
            trajectories: List of trajectory dictionaries
        """
        trajectories = []
        
        for i in range(num_trajectories):
            # Generate trajectory
            traj_length = np.random.randint(5, 20)
            
            # Generate states and actions
            states = [self._generate_sample() for _ in range(traj_length)]
            actions = [self._generate_action() for _ in range(traj_length)]
            
            # Create trajectory
            trajectory = {
                'id': i,
                'states': states,
                'actions': actions,
                'length': traj_length
            }
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def _generate_sample(self) -> Dict:
        """
        Generate a synthetic sample.
        
        Returns:
            sample: Dictionary of sensor data
        """
        # Generate camera data
        camera = torch.rand(*self.camera_size)
        
        # Generate LiDAR data
        lidar = torch.rand(self.max_lidar_points, 3)
        
        # Generate radar data
        radar = torch.rand(self.max_radar_points, 4)
        
        # Generate GPS data
        gps = torch.rand(self.gps_dim)
        
        return {
            'camera': camera,
            'lidar': lidar,
            'radar': radar,
            'gps': gps
        }
    
    def _generate_action(self) -> torch.Tensor:
        """
        Generate a synthetic action.
        
        Returns:
            action: Action tensor
        """
        # Generate action tensor
        action = torch.zeros(self.action_dim)
        
        # Set values within realistic ranges
        action[0] = torch.rand(1) * 2 - 1  # steering: [-1, 1]
        action[1] = torch.rand(1)          # throttle: [0, 1]
        action[2] = torch.rand(1) * 0.3    # brake: [0, 0.3]
        
        return action
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a synthetic sample.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary with multimodal data and action
        """
        # Select a random trajectory
        traj_idx = idx % len(self.trajectories)
        trajectory = self.trajectories[traj_idx]
        
        # Select a random state from the trajectory
        state_idx = idx % trajectory['length']
        
        # Get state and action
        multimodal = trajectory['states'][state_idx]
        action = trajectory['actions'][state_idx]
        
        # Create sample
        sample = {
            'multimodal': multimodal,
            'actions': action,
            'trajectory_id': trajectory['id'],
            'timestamp': idx
        }
        
        return sample


def compute_behavioral_cloning_metrics(expert_trajectories: List[Dict], 
                                       policy_trajectories: List[Dict]) -> Dict:
    """
    Compute behavioral cloning metrics.
    
    Args:
        expert_trajectories: List of expert trajectories
        policy_trajectories: List of policy trajectories
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Calculate MSE and MAE on actions
    expert_actions = []
    policy_actions = []
    
    # Extract all actions
    for traj in expert_trajectories:
        if 'actions' in traj:
            expert_actions.append(traj['actions'])
    
    for traj in policy_trajectories:
        if 'actions' in traj:
            policy_actions.append(traj['actions'])
    
    # Convert to numpy arrays for easier computation
    if expert_actions and policy_actions:
        expert_actions = np.concatenate(expert_actions, axis=0)
        policy_actions = np.concatenate(policy_actions, axis=0)
        
        # Use min length
        min_len = min(len(expert_actions), len(policy_actions))
        expert_actions = expert_actions[:min_len]
        policy_actions = policy_actions[:min_len]
        
        # Calculate metrics
        bc_mse = np.mean((expert_actions - policy_actions)**2)
        bc_mae = np.mean(np.abs(expert_actions - policy_actions))
        
        # Calculate per-action metrics
        steering_mse = np.mean((expert_actions[:, 0] - policy_actions[:, 0])**2)
        throttle_mse = np.mean((expert_actions[:, 1] - policy_actions[:, 1])**2)
        brake_mse = np.mean((expert_actions[:, 2] - policy_actions[:, 2])**2)
        
        # Calculate correlation
        action_correlation = np.corrcoef(expert_actions.flatten(), policy_actions.flatten())[0, 1]
        
        metrics = {
            'bc_mse': bc_mse,
            'bc_mae': bc_mae,
            'action_correlation': action_correlation,
            'steering_mse': steering_mse,
            'throttle_mse': throttle_mse,
            'brake_mse': brake_mse
        }
    else:
        # Default metrics if no actions
        metrics = {
            'bc_mse': 0.0,
            'bc_mae': 0.0,
            'action_correlation': 0.0,
            'steering_mse': 0.0,
            'throttle_mse': 0.0,
            'brake_mse': 0.0
        }
    
    return metrics
