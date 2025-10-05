#!/usr/bin/env python3
"""
Example script for testing the trained navigation model.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import argparse
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from navigation.navigation_controller import NavigationController, NavigationSimulator
from utils.data_loader import SyntheticNavigationDataset
from utils.visualization import plot_action_comparison, visualize_multimodal_data


def create_test_data(data_path: Path, num_samples: int = 50):
    """Create synthetic test data."""
    print(f"Creating synthetic test dataset with {num_samples} samples...")
    
    test_path = data_path / "test"
    test_path.mkdir(parents=True, exist_ok=True)
    
    config = {
        'camera_size': (224, 224),
        'max_lidar_points': 16384,
        'max_radar_detections': 64
    }
    
    # Generate synthetic test trajectories
    test_dataset = SyntheticNavigationDataset(config, num_samples=num_samples)
    
    # Save test data in expected format
    for i in range(min(50, num_samples)):
        sample = test_dataset[i]
        
        # Create trajectory directory
        traj_dir = test_path / f"test_trajectory_{i:04d}"
        traj_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["camera", "lidar", "radar", "gps", "actions"]:
            (traj_dir / subdir).mkdir(exist_ok=True)
        
        # Save sample data (similar to training example)
        timestamp = f"frame_{i:06d}"
        
        # Camera
        camera_data = sample['multimodal']['camera']
        camera_image = (camera_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        import cv2
        cv2.imwrite(str(traj_dir / "camera" / f"{timestamp}.jpg"), camera_image)
        
        # Actions
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
        
        # LiDAR
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
            if detection.sum() > 0:
                radar_detections.append({
                    'range': float(detection[0]),
                    'azimuth': float(detection[1]),
                    'elevation': float(detection[2]),
                    'velocity': float(detection[3])
                })
        
        radar_dict = {'detections': radar_detections}
        with open(traj_dir / "radar" / f"{timestamp}.json", 'w') as f:
            json.dump(radar_dict, f)
    
    print(f"Created {min(50, num_samples)} test samples")


def test_single_prediction(controller, sample):
    """Test a single prediction and visualize the result."""
    print("\nTesting single prediction...")
    
    # Convert sample to raw sensor data
    multimodal_data = sample['multimodal']
    
    raw_sensor_data = {
        'camera': (multimodal_data['camera'].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
        'lidar': multimodal_data['lidar'].numpy(),
        'gps': {
            'latitude': float(multimodal_data['gps'][0]),
            'longitude': float(multimodal_data['gps'][1]),
            'altitude': float(multimodal_data['gps'][2]),
            'heading': float(multimodal_data['gps'][3]),
            'pitch': float(multimodal_data['gps'][4]),
            'roll': float(multimodal_data['gps'][5]),
            'velocity_x': float(multimodal_data['gps'][6]),
            'velocity_y': float(multimodal_data['gps'][7]),
            'velocity_z': float(multimodal_data['gps'][8])
        }
    }
    
    # Get prediction
    control_commands = controller.predict_action(raw_sensor_data)
    
    print("Control Commands:")
    print(f"  Steering: {control_commands['steering']:.3f}")
    print(f"  Throttle: {control_commands['throttle']:.3f}")
    print(f"  Brake: {control_commands['brake']:.3f}")
    print(f"  Emergency Stop: {control_commands['emergency_stop']}")
    
    # Compare with ground truth
    gt_actions = sample['actions'].numpy()
    print("\nGround Truth Actions:")
    print(f"  Steering: {gt_actions[0]:.3f}")
    print(f"  Throttle: {gt_actions[1]:.3f}")
    print(f"  Brake: {gt_actions[2]:.3f}")
    
    # Calculate errors
    steering_error = abs(control_commands['steering'] - gt_actions[0])
    throttle_error = abs(control_commands['throttle'] - gt_actions[1])
    brake_error = abs(control_commands['brake'] - gt_actions[2])
    
    print("\nPrediction Errors:")
    print(f"  Steering Error: {steering_error:.3f}")
    print(f"  Throttle Error: {throttle_error:.3f}")
    print(f"  Brake Error: {brake_error:.3f}")
    
    return control_commands, gt_actions


def main():
    parser = argparse.ArgumentParser(description='Test Navigation Model')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--test_data', type=str, help='Path to test dataset')
    parser.add_argument('--output', type=str, default='test_results.json', help='Output file')
    parser.add_argument('--single_test', action='store_true', help='Run single prediction test')
    parser.add_argument('--full_simulation', action='store_true', help='Run full simulation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ADVERSARIAL IRL NAVIGATION TESTING")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    
    # Use defaults if not provided
    model_path = args.model or str(project_root / "checkpoints" / "best_model.pth")
    config_path = args.config or str(project_root / "configs" / "navigation_config.yaml")
    test_data_path = args.test_data or str(project_root / "data" / "test")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please train a model first using examples/train_example.py")
        return 1
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"Error: Config file not found at {config_path}")
        print("Using default configuration...")
        config = {
            'fusion_dim': 512,
            'action_dim': 3,
            'hidden_dim': 256,
            'camera_size': [224, 224],
            'max_lidar_points': 16384,
            'max_radar_detections': 64,
            'max_speed': 30.0,
            'target_fps': 30,
            'safety_margin': 2.0,
            'simulation_fps': 30,
            'simulation_duration': 60.0
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Create test data if it doesn't exist
    if not Path(test_data_path).exists():
        print("Creating test data...")
        create_test_data(project_root / "data")
        test_data_path = str(project_root / "data" / "test")
    
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Test Data: {test_data_path}")
    
    try:
        # Initialize navigation controller
        print("\nInitializing navigation controller...")
        controller = NavigationController(config, model_path)
        print("Controller initialized successfully!")
        
        # Single prediction test
        if args.single_test or not args.full_simulation:
            print("\n" + "=" * 40)
            print("SINGLE PREDICTION TEST")
            print("=" * 40)
            
            # Create a single synthetic sample for testing
            config_dataset = {
                'camera_size': tuple(config['camera_size']),
                'max_lidar_points': config['max_lidar_points'],
                'max_radar_detections': config['max_radar_detections']
            }
            synthetic_dataset = SyntheticNavigationDataset(config_dataset, num_samples=1)
            sample = synthetic_dataset[0]
            
            # Test prediction
            control_commands, gt_actions = test_single_prediction(controller, sample)
            
            # Visualize input data
            print("\nVisualizing input sensor data...")
            try:
                visualize_multimodal_data(sample, save_path="sensor_visualization.png")
                print("Sensor visualization saved to sensor_visualization.png")
            except Exception as e:
                print(f"Could not create visualization: {e}")
        
        # Full simulation test
        if args.full_simulation:
            print("\n" + "=" * 40)
            print("FULL SIMULATION TEST")
            print("=" * 40)
            
            # Initialize simulator
            simulator = NavigationSimulator(controller, config)
            
            # Run simulation
            results = simulator.run_simulation(test_data_path)
            
            # Save results
            simulator.save_results(results, args.output)
            
            # Print summary
            print("\nSimulation Results:")
            print(f"Simulation Time: {results['simulation_time']:.2f} seconds")
            print(f"Frames Processed: {results['frame_count']}")
            print(f"Average FPS: {results['average_fps']:.2f}")
            
            if 'action_statistics' in results:
                stats = results['action_statistics']
                print(f"Emergency Stops: {stats['emergency_stop_count']} ({stats['emergency_stop_rate']:.2%})")
                print(f"Average Steering: {stats['steering_mean']:.3f} ± {stats['steering_std']:.3f}")
                print(f"Average Throttle: {stats['throttle_mean']:.3f} ± {stats['throttle_std']:.3f}")
                print(f"Average Brake: {stats['brake_mean']:.3f} ± {stats['brake_std']:.3f}")
            
            if 'performance_statistics' in results:
                perf = results['performance_statistics']
                print(f"\nPerformance Statistics:")
                print(f"Average FPS: {perf['fps_mean']:.2f} ± {perf['fps_std']:.2f}")
                print(f"FPS Range: [{perf['fps_min']:.2f}, {perf['fps_max']:.2f}]")
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("This might be due to model compatibility issues or missing dependencies.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
