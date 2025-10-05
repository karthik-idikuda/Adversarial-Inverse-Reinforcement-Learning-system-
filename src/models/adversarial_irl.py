"""
Inverse Reinforcement Learning Model for Autonomous Navigation
with Adversarial Multimodal Data Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class MultimodalEncoder(nn.Module):
    """Encodes multimodal sensor data (camera, lidar, radar, GPS) into a unified representation."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Camera encoder (CNN)
        self.camera_encoder = self._build_camera_encoder()
        
        # LiDAR encoder (PointNet-like)
        self.lidar_encoder = self._build_lidar_encoder()
        
        # Radar encoder
        self.radar_encoder = self._build_radar_encoder()
        
        # GPS/IMU encoder
        self.gps_encoder = self._build_gps_encoder()
        
        # Fusion layer - calculate dimensions explicitly
        self.fusion_dim = config.get('fusion_dim', 512)
        total_features = 256 + 256 + 128 + 64  # camera + lidar + radar + gps
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
    def _build_camera_encoder(self) -> nn.Module:
        """Build CNN encoder for camera data."""
        return nn.Sequential(
            # Input: (batch, 3, 224, 224)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_conv_block(64, 128, stride=2),
            self._make_conv_block(128, 256, stride=2),
            self._make_conv_block(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
    
    def _build_lidar_encoder(self) -> nn.Module:
        """Build PointNet-like encoder for LiDAR point clouds."""
        return nn.Sequential(
            nn.Linear(3, 64),  # x, y, z coordinates
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    def _build_radar_encoder(self) -> nn.Module:
        """Build encoder for radar data."""
        return nn.Sequential(
            nn.Linear(4, 32),  # range, azimuth, elevation, velocity
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
    
    def _build_gps_encoder(self) -> nn.Module:
        """Build encoder for GPS/IMU data."""
        return nn.Sequential(
            nn.Linear(9, 32),  # lat, lon, alt, heading, pitch, roll, vel_x, vel_y, vel_z
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create a convolutional block with residual connection."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _get_total_feature_dim(self) -> int:
        """Calculate total feature dimension after concatenation."""
        # Calculate actual dimensions from the network architectures
        camera_dim = 256  # From camera encoder final layer
        lidar_dim = 256   # From lidar encoder final layer  
        radar_dim = 128   # From radar encoder final layer
        gps_dim = 64      # From GPS encoder final layer
        return camera_dim + lidar_dim + radar_dim + gps_dim
    
    def forward(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal encoder.
        
        Args:
            multimodal_data: Dictionary containing sensor data
                - 'camera': (batch, 3, 224, 224)
                - 'lidar': (batch, num_points, 3)
                - 'radar': (batch, num_detections, 4)
                - 'gps': (batch, 9)
        
        Returns:
            Fused feature representation (batch, fusion_dim)
        """
        features = []
        
        # Camera features
        if 'camera' in multimodal_data:
            cam_features = self.camera_encoder(multimodal_data['camera'])
            features.append(cam_features)
        
        # LiDAR features (max pooling over points)
        if 'lidar' in multimodal_data:
            lidar_data = multimodal_data['lidar']
            lidar_features = self.lidar_encoder(lidar_data)  # (batch, num_points, 256)
            lidar_features = torch.max(lidar_features, dim=1)[0]  # Global max pooling
            features.append(lidar_features)
        
        # Radar features (max pooling over detections)
        if 'radar' in multimodal_data:
            radar_data = multimodal_data['radar']
            radar_features = self.radar_encoder(radar_data)  # (batch, num_detections, 128)
            radar_features = torch.max(radar_features, dim=1)[0]  # Global max pooling
            features.append(radar_features)
        
        # GPS features
        if 'gps' in multimodal_data:
            gps_features = self.gps_encoder(multimodal_data['gps'])
            features.append(gps_features)
        
        # Concatenate and fuse features
        if features:
            combined_features = torch.cat(features, dim=1)
            fused_features = self.fusion_layer(combined_features)
            return fused_features
        else:
            raise ValueError("No valid sensor data provided")


class RewardNetwork(nn.Module):
    """Neural network that learns to predict rewards from state-action pairs."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for given state-action pair.
        
        Args:
            state: State representation (batch, state_dim)
            action: Action representation (batch, action_dim)
        
        Returns:
            Predicted reward (batch, 1)
        """
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        
        combined = torch.cat([state_features, action_features], dim=1)
        reward = self.reward_head(combined)
        
        return reward


class PolicyNetwork(nn.Module):
    """Policy network that outputs actions given states."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # For continuous control (steering, throttle, brake)
        self.action_dim = action_dim
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate action given state.
        
        Args:
            state: State representation (batch, state_dim)
        
        Returns:
            Action (batch, action_dim)
        """
        action_raw = self.network(state)
        
        # Apply appropriate activation functions for different action components
        if self.action_dim >= 3:  # steering, throttle, brake
            steering = torch.tanh(action_raw[:, 0:1])  # [-1, 1]
            throttle = torch.sigmoid(action_raw[:, 1:2])  # [0, 1]
            brake = torch.sigmoid(action_raw[:, 2:3])  # [0, 1]
            
            if self.action_dim > 3:
                other_actions = action_raw[:, 3:]
                action = torch.cat([steering, throttle, brake, other_actions], dim=1)
            else:
                action = torch.cat([steering, throttle, brake], dim=1)
        else:
            action = torch.tanh(action_raw)
        
        return action


class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between expert and policy trajectories.
        
        Args:
            state: State representation (batch, state_dim)
            action: Action representation (batch, action_dim)
        
        Returns:
            Probability of being expert trajectory (batch, 1)
        """
        combined = torch.cat([state, action], dim=1)
        return self.network(combined)


class AdversarialIRLAgent(nn.Module):
    """
    Complete Adversarial IRL agent that combines all components.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Multimodal encoder
        self.multimodal_encoder = MultimodalEncoder(config)
        
        # Core IRL components
        state_dim = config.get('fusion_dim', 512)
        action_dim = config.get('action_dim', 3)  # steering, throttle, brake
        hidden_dim = config.get('hidden_dim', 256)
        
        self.reward_network = RewardNetwork(state_dim, action_dim, hidden_dim)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim)
        
        # Adversarial attack parameters
        self.epsilon = config.get('adversarial_epsilon', 0.1)
        self.alpha = config.get('adversarial_alpha', 0.01)
        self.num_attack_steps = config.get('num_attack_steps', 10)
    
    def encode_multimodal_state(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode multimodal sensor data into state representation."""
        return self.multimodal_encoder(multimodal_data)
    
    def get_action(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get action from current multimodal observation."""
        state = self.encode_multimodal_state(multimodal_data)
        action = self.policy_network(state)
        return action
    
    def get_reward(self, multimodal_data: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Get reward for state-action pair."""
        state = self.encode_multimodal_state(multimodal_data)
        reward = self.reward_network(state, action)
        return reward
    
    def discriminate(self, multimodal_data: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        """Discriminate between expert and policy actions."""
        state = self.encode_multimodal_state(multimodal_data)
        prob_expert = self.discriminator(state, action)
        return prob_expert
    
    def generate_adversarial_perturbation(self, multimodal_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generate adversarial perturbations for multimodal data using PGD.
        
        Args:
            multimodal_data: Original sensor data
        
        Returns:
            Perturbed sensor data
        """
        perturbed_data = {}
        
        for modality, data in multimodal_data.items():
            if data.requires_grad:
                # Initialize perturbation
                delta = torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon)
                delta.requires_grad_(True)
                
                # PGD attack
                for _ in range(self.num_attack_steps):
                    # Forward pass with perturbation
                    perturbed_input = {modality: data + delta}
                    state = self.encode_multimodal_state(perturbed_input)
                    action = self.policy_network(state)
                    
                    # Calculate loss (maximize uncertainty)
                    reward = self.reward_network(state, action)
                    loss = -torch.var(reward)  # Minimize variance to maximize uncertainty
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update perturbation
                    delta.data = delta.data + self.alpha * delta.grad.sign()
                    delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                    delta.grad.zero_()
                
                perturbed_data[modality] = data + delta.detach()
            else:
                perturbed_data[modality] = data
        
        return perturbed_data
