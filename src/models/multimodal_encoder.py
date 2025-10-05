"""
MultimodalEncoder class for Adversarial IRL Navigation System.

This module contains the implementation of the MultimodalEncoder class,
which is responsible for encoding multimodal sensor data (camera, lidar,
radar, and GPS) into a unified representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalEncoder(nn.Module):
    """
    Encodes multimodal sensor data into a unified representation.
    
    This encoder takes multiple sensor inputs (camera, lidar, radar, GPS)
    and processes them through separate encoder networks before fusing
    them together into a unified representation.
    """
    
    def __init__(self, config):
        """
        Initialize the MultimodalEncoder.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        
        # Get dimensions from config
        self.camera_size = config.get('camera_size', (3, 224, 224))
        self.camera_channels = self.camera_size[0]
        self.camera_height = self.camera_size[1]
        self.camera_width = self.camera_size[2]
        
        self.lidar_dim = config.get('lidar_dim', 1024 * 3)  # N points * 3 (x, y, z)
        self.radar_dim = config.get('radar_dim', 256)
        self.gps_dim = config.get('gps_dim', 9)
        
        self.camera_feat_dim = config.get('camera_feat_dim', 128)
        self.lidar_feat_dim = config.get('lidar_feat_dim', 64)
        self.radar_feat_dim = config.get('radar_feat_dim', 32)
        self.gps_feat_dim = config.get('gps_feat_dim', 16)
        
        self.fusion_dim = config.get('fusion_dim', 256)
        
        # CNN for camera
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(self.camera_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.camera_feat_dim)
        )
        
        # MLP for LiDAR
        self.lidar_encoder = nn.Sequential(
            nn.Linear(self.lidar_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.lidar_feat_dim)
        )
        
        # MLP for Radar
        self.radar_encoder = nn.Sequential(
            nn.Linear(self.radar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.radar_feat_dim)
        )
        
        # MLP for GPS
        self.gps_encoder = nn.Sequential(
            nn.Linear(self.gps_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.gps_feat_dim)
        )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(
                self.camera_feat_dim + 
                self.lidar_feat_dim + 
                self.radar_feat_dim + 
                self.gps_feat_dim,
                512
            ),
            nn.ReLU(),
            nn.Linear(512, self.fusion_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Dictionary of sensor inputs
                - 'camera': Camera image tensor (B, C, H, W)
                - 'lidar': LiDAR point cloud tensor (B, N, 3) or (B, N*3)
                - 'radar': Radar detections tensor (B, M, 4) or (B, M*4)
                - 'gps': GPS data tensor (B, 9)
        
        Returns:
            fusion_features: Fused multimodal features (B, fusion_dim)
        """
        batch_size = next(iter(x.values())).shape[0]
        
        # Process camera if available
        if 'camera' in x:
            camera_features = self.camera_encoder(x['camera'])
        else:
            # Use zeros if camera not available
            camera_features = torch.zeros(
                (batch_size, self.camera_feat_dim),
                device=next(iter(x.values())).device
            )
        
        # Process LiDAR if available
        if 'lidar' in x:
            lidar_data = x['lidar']
            # Flatten if needed
            if len(lidar_data.shape) == 3:  # (B, N, 3)
                lidar_data = lidar_data.reshape(batch_size, -1)
            lidar_features = self.lidar_encoder(lidar_data)
        else:
            # Use zeros if LiDAR not available
            lidar_features = torch.zeros(
                (batch_size, self.lidar_feat_dim),
                device=next(iter(x.values())).device
            )
        
        # Process radar if available
        if 'radar' in x:
            radar_data = x['radar']
            # Flatten if needed
            if len(radar_data.shape) == 3:  # (B, M, 4)
                radar_data = radar_data.reshape(batch_size, -1)
            radar_features = self.radar_encoder(radar_data)
        else:
            # Use zeros if radar not available
            radar_features = torch.zeros(
                (batch_size, self.radar_feat_dim),
                device=next(iter(x.values())).device
            )
        
        # Process GPS if available
        if 'gps' in x:
            gps_features = self.gps_encoder(x['gps'])
        else:
            # Use zeros if GPS not available
            gps_features = torch.zeros(
                (batch_size, self.gps_feat_dim),
                device=next(iter(x.values())).device
            )
        
        # Concatenate features
        combined_features = torch.cat([
            camera_features,
            lidar_features,
            radar_features,
            gps_features
        ], dim=1)
        
        # Fusion
        fusion_features = self.fusion_network(combined_features)
        
        return fusion_features
