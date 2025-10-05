#!/usr/bin/env python3
"""
Example script for training the Adversarial IRL model.
This script demonstrates the complete training workflow.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.train_irl import AdversarialIRLTrainer
from utils.data_loader import SyntheticNavigationDataset
from utils.visualization import create_training_dashboard


def create_example_data(data_path: Path, num_samples: int = 1000):
    """Create synthetic training data for demonstration."""
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    # Create directory structure
    expert_path = data_path / "expert_demonstrations"
    validation_path = data_path / "validation"
    
    expert_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic trajectories
    config = {
        'camera_size': (224, 224),
        'max_lidar_points': 16384,
        'max_radar_detections': 64
    }
    
    # Expert data
    expert_dataset = SyntheticNavigationDataset(config, num_samples=num_samples)
    
    # Save synthetic data in expected format
    for i in range(min(100, num_samples)):  # Save first 100 samples as examples
        sample = expert_dataset[i]
        
        # Create trajectory directory
        traj_dir = expert_path / f"trajectory_{i:04d}"
        traj_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["camera", "lidar", "radar", "gps", "actions"]:
            (traj_dir / subdir).mkdir(exist_ok=True)
        
        # Save sample data (placeholder files)
        timestamp = f"frame_{i:06d}"
        
        # Camera (save as image)
        camera_data = sample['multimodal']['camera']
        camera_image = (camera_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        import cv2
        cv2.imwrite(str(traj_dir / "camera" / f"{timestamp}.jpg"), camera_image)
        
        # Actions (save as JSON)
        import json
        actions = sample['actions'].numpy()
        action_dict = {
            'steering': float(actions[0]),
            'throttle': float(actions[1]),
            'brake': float(actions[2])
        }
        with open(traj_dir / "actions" / f"{timestamp}.json", 'w') as f:
            json.dump(action_dict, f)
        
        # GPS data
        gps_data = sample['multimodal']['gps'].numpy()
        gps_dict = {
            'latitude': float(gps_data[0]),
            'longitude': float(gps_data[1]),
            'altitude': float(gps_data[2]),
            'heading': float(gps_data[3]),
            'pitch': float(gps_data[4]),
            'roll': float(gps_data[5]),
            'velocity_x': float(gps_data[6]),
            'velocity_y': float(gps_data[7]),
            'velocity_z': float(gps_data[8])
        }
        with open(traj_dir / "gps" / f"{timestamp}.json", 'w') as f:
            json.dump(gps_dict, f)
        
        # LiDAR (save as PCD file using a simple format)
        lidar_points = sample['multimodal']['lidar'].numpy()
        pcd_content = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(lidar_points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(lidar_points)}
DATA ascii
"""
        for point in lidar_points:
            pcd_content += f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"
        
        with open(traj_dir / "lidar" / f"{timestamp}.pcd", 'w') as f:
            f.write(pcd_content)
        
        # Radar data
        radar_data = sample['multimodal']['radar'].numpy()
        radar_detections = []
        for detection in radar_data:
            if detection.sum() > 0:  # Skip zero detections
                radar_detections.append({
                    'range': float(detection[0]),
                    'azimuth': float(detection[1]),
                    'elevation': float(detection[2]),
                    'velocity': float(detection[3])
                })
        
        radar_dict = {'detections': radar_detections}
        with open(traj_dir / "radar" / f"{timestamp}.json", 'w') as f:
            json.dump(radar_dict, f)
    
    # Copy some samples to validation directory
    import shutil
    validation_samples = min(20, num_samples // 10)
    for i in range(validation_samples):
        src_dir = expert_path / f"trajectory_{i:04d}"
        dst_dir = validation_path / f"trajectory_{i:04d}"
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    
    print(f"Created {100} expert demonstrations and {validation_samples} validation samples")


def main():
    """Main training example."""
    print("=" * 60)
    print("ADVERSARIAL IRL TRAINING EXAMPLE")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data"
    config_path = project_root / "configs" / "irl_config.yaml"
    
    # Create synthetic data if it doesn't exist
    expert_data_path = data_path / "expert_demonstrations"
    if not expert_data_path.exists() or len(list(expert_data_path.iterdir())) == 0:
        print("Creating synthetic training data...")
        create_example_data(data_path, num_samples=200)
    
    # Load configuration
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with correct paths
    config['expert_data_path'] = str(data_path / "expert_demonstrations")
    config['validation_data_path'] = str(data_path / "validation")
    
    # Reduce training time for example
    config['num_epochs'] = 5
    config['batch_size'] = 8
    config['use_wandb'] = False  # Disable wandb for example
    
    print("Configuration loaded:")
    print(f"  Expert data: {config['expert_data_path']}")
    print(f"  Validation data: {config['validation_data_path']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    try:
        trainer = AdversarialIRLTrainer(config)
        print("Trainer initialized successfully!")
        
        # Start training
        print("\nStarting training...")
        trainer.train()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print final model location
        checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        best_model_path = checkpoint_dir / 'best_model.pth'
        
        if best_model_path.exists():
            print(f"\nBest model saved to: {best_model_path}")
            print("You can now test the model with:")
            print(f"python examples/test_navigation.py --model {best_model_path}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("This might be due to missing dependencies or data issues.")
        print("Please check the requirements.txt and ensure all dependencies are installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
