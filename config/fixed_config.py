"""
Fixed configuration for the Adversarial IRL Navigation System.
This configuration resolves tensor dimension issues.
"""

import torch

# Base configuration with fixed dimensions
CONFIG = {
    # Device configuration
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    
    # Model architecture
    'state_dim': 512,       # Output dimension from multimodal encoder
    'action_dim': 4,        # [steering, throttle, brake, gear]
    'hidden_dim': 256,      # Hidden layer dimension
    'fusion_dim': 512,      # Fusion layer output dimension
    
    # Sensor dimensions (fixed for compatibility)
    'camera_features': 256,
    'lidar_features': 256,
    'radar_features': 128,
    'gps_features': 64,
    
    # Input data specifications
    'camera_size': (3, 224, 224),
    'max_lidar_points': 1024,
    'max_radar_points': 32,
    'gps_dim': 9,  # [lat, lon, alt, heading, pitch, roll, vel_x, vel_y, vel_z]
    
    # Training parameters
    'batch_size': 4,
    'learning_rate': 0.0001,
    'epochs': 50,
    'save_interval': 10,
    
    # Optimizer settings
    'lr_policy': 0.0001,
    'lr_reward': 0.0001,
    'lr_discriminator': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    
    # Loss weights
    'lambda_gp': 10.0,      # Gradient penalty weight
    'lambda_irl': 1.0,      # IRL loss weight
    'lambda_adv': 0.1,      # Adversarial loss weight
    
    # Training schedule
    'warmup_epochs': 5,
    'discriminator_steps': 5,
    'generator_steps': 1,
    
    # Data paths
    'data_path': 'data',
    'checkpoint_path': 'checkpoints',
    'log_path': 'logs',
    
    # Evaluation
    'eval_interval': 5,
    'num_eval_episodes': 10,
    'eval_trajectory_length': 100,
    
    # Adversarial training
    'adversarial_eps': 0.1,
    'adversarial_alpha': 0.01,
    'adversarial_iterations': 7,
    
    # Navigation specifics
    'max_episode_steps': 1000,
    'reward_threshold': 0.8,
    'success_threshold': 0.9
}

def get_config():
    """Return the configuration dictionary."""
    return CONFIG.copy()

def validate_config(config):
    """Validate configuration parameters."""
    required_keys = [
        'state_dim', 'action_dim', 'hidden_dim', 'fusion_dim',
        'camera_features', 'lidar_features', 'radar_features', 'gps_features',
        'batch_size', 'learning_rate'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate dimension consistency
    total_features = (config['camera_features'] + config['lidar_features'] + 
                     config['radar_features'] + config['gps_features'])
    
    if total_features != 704:  # 256 + 256 + 128 + 64
        print(f"Warning: Total features ({total_features}) may cause dimension issues")
    
    return True

if __name__ == "__main__":
    config = get_config()
    validate_config(config)
    print("Configuration validation passed!")
    print(f"Total feature dimensions: {config['camera_features'] + config['lidar_features'] + config['radar_features'] + config['gps_features']}")
