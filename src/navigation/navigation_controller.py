"""
Navigation system for autonomous vehicle using trained IRL model.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import yaml
import cv2
from collections import deque
import threading
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.adversarial_irl import AdversarialIRLAgent
from utils.data_loader import MultimodalNavigationDataset


class NavigationController:
    """
    Real-time navigation controller using trained IRL agent.
    """
    
    def __init__(self, config: Dict, model_path: str):
        """
        Initialize the navigation controller.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model checkpoint
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load trained model
        self.agent = self._load_model(model_path)
        
        # Navigation parameters
        self.max_speed = config.get('max_speed', 30.0)  # m/s
        self.target_fps = config.get('target_fps', 30)
        self.safety_margin = config.get('safety_margin', 2.0)  # meters
        
        # State tracking
        self.current_state = None
        self.action_history = deque(maxlen=10)
        self.speed_history = deque(maxlen=20)
        
        # Safety systems
        self.emergency_brake = False
        self.safety_override = False
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.time()
        
        self.logger.info("Navigation controller initialized")
    
    def _load_model(self, model_path: str) -> AdversarialIRLAgent:
        """Load the trained IRL model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize agent with config from checkpoint
        model_config = checkpoint.get('config', self.config)
        agent = AdversarialIRLAgent(model_config).to(self.device)
        
        # Load state dict
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()
        
        self.logger.info(f"Model loaded from {model_path}")
        return agent
    
    def process_sensor_data(self, raw_sensor_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process raw sensor data into model-compatible format.
        
        Args:
            raw_sensor_data: Dictionary of raw sensor data
        
        Returns:
            Processed multimodal data ready for the model
        """
        processed_data = {}
        
        # Process camera image
        if 'camera' in raw_sensor_data:
            camera_image = raw_sensor_data['camera']
            if isinstance(camera_image, np.ndarray):
                # Resize and normalize
                camera_size = self.config.get('camera_size', (224, 224))
                image = cv2.resize(camera_image, camera_size)
                image = image.astype(np.float32) / 255.0
                
                # Convert to tensor (CHW format)
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                processed_data['camera'] = image_tensor.to(self.device)
        
        # Process LiDAR point cloud
        if 'lidar' in raw_sensor_data:
            lidar_points = raw_sensor_data['lidar']
            if isinstance(lidar_points, np.ndarray):
                max_points = self.config.get('max_lidar_points', 16384)
                
                # Subsample or pad points
                if len(lidar_points) > max_points:
                    indices = np.random.choice(len(lidar_points), max_points, replace=False)
                    lidar_points = lidar_points[indices]
                elif len(lidar_points) < max_points:
                    padding = np.zeros((max_points - len(lidar_points), 3))
                    lidar_points = np.vstack([lidar_points, padding])
                
                lidar_tensor = torch.from_numpy(lidar_points.astype(np.float32)).unsqueeze(0)
                processed_data['lidar'] = lidar_tensor.to(self.device)
        
        # Process radar data
        if 'radar' in raw_sensor_data:
            radar_detections = raw_sensor_data['radar']
            if isinstance(radar_detections, list):
                max_detections = self.config.get('max_radar_detections', 64)
                
                # Convert detections to array format
                radar_array = []
                for detection in radar_detections[:max_detections]:
                    radar_array.append([
                        detection.get('range', 0.0),
                        detection.get('azimuth', 0.0),
                        detection.get('elevation', 0.0),
                        detection.get('velocity', 0.0)
                    ])
                
                # Pad if necessary
                while len(radar_array) < max_detections:
                    radar_array.append([0.0, 0.0, 0.0, 0.0])
                
                radar_tensor = torch.tensor(radar_array, dtype=torch.float32).unsqueeze(0)
                processed_data['radar'] = radar_tensor.to(self.device)
        
        # Process GPS/IMU data
        if 'gps' in raw_sensor_data:
            gps_data = raw_sensor_data['gps']
            if isinstance(gps_data, dict):
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
                
                gps_tensor = torch.tensor(gps_features, dtype=torch.float32).unsqueeze(0)
                processed_data['gps'] = gps_tensor.to(self.device)
        
        return processed_data
    
    def run_safety_checks(self, 
                         multimodal_data: Dict[str, torch.Tensor], 
                         predicted_action: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Run safety checks on predicted actions.
        
        Args:
            multimodal_data: Processed sensor data
            predicted_action: Predicted action from the model
        
        Returns:
            Tuple of (safe_action, emergency_stop)
        """
        safe_action = predicted_action.clone()
        emergency_stop = False
        
        # Check 1: Obstacle detection from LiDAR
        if 'lidar' in multimodal_data:
            lidar_points = multimodal_data['lidar'].squeeze(0)  # Remove batch dimension
            
            # Check for close obstacles in front (simple distance-based check)
            forward_points = lidar_points[(lidar_points[:, 1] > 0) & (lidar_points[:, 1] < self.safety_margin)]
            close_obstacles = forward_points[torch.norm(forward_points[:, :2], dim=1) < self.safety_margin]
            
            if len(close_obstacles) > 10:  # Threshold for obstacle detection
                self.logger.warning("Close obstacle detected - applying emergency brake")
                safe_action[:, 1] = 0.0  # Zero throttle
                safe_action[:, 2] = 1.0  # Full brake
                emergency_stop = True
        
        # Check 2: Speed limit enforcement
        current_speed = self._estimate_current_speed(multimodal_data)
        if current_speed > self.max_speed:
            self.logger.warning(f"Speed limit exceeded: {current_speed:.2f} m/s")
            safe_action[:, 1] = torch.clamp(safe_action[:, 1], max=0.3)  # Reduce throttle
        
        # Check 3: Action smoothness (prevent jerky movements)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action_diff = torch.abs(safe_action - last_action)
            
            # Limit steering rate
            max_steering_change = 0.2  # Maximum steering change per frame
            if action_diff[0, 0] > max_steering_change:
                steering_sign = torch.sign(safe_action[0, 0] - last_action[0, 0])
                safe_action[0, 0] = last_action[0, 0] + steering_sign * max_steering_change
        
        # Check 4: Throttle-brake conflict
        if safe_action[0, 1] > 0.1 and safe_action[0, 2] > 0.1:  # Both throttle and brake
            self.logger.warning("Throttle-brake conflict detected")
            safe_action[:, 1] = 0.0  # Prioritize braking
        
        return safe_action, emergency_stop
    
    def _estimate_current_speed(self, multimodal_data: Dict[str, torch.Tensor]) -> float:
        """Estimate current speed from GPS/IMU data."""
        if 'gps' in multimodal_data:
            gps_data = multimodal_data['gps'].squeeze(0)  # Remove batch dimension
            vel_x = gps_data[6].item()
            vel_y = gps_data[7].item()
            speed = np.sqrt(vel_x**2 + vel_y**2)
            return speed
        return 0.0
    
    def predict_action(self, raw_sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Main prediction method for navigation.
        
        Args:
            raw_sensor_data: Raw sensor data from the vehicle
        
        Returns:
            Dictionary containing control commands
        """
        start_time = time.time()
        
        try:
            # Process sensor data
            multimodal_data = self.process_sensor_data(raw_sensor_data)
            
            if not multimodal_data:
                self.logger.warning("No valid sensor data received")
                return self._get_safe_stop_action()
            
            # Get prediction from model
            with torch.no_grad():
                predicted_action = self.agent.get_action(multimodal_data)
            
            # Run safety checks
            safe_action, emergency_stop = self.run_safety_checks(multimodal_data, predicted_action)
            
            # Convert to control commands
            control_commands = self._action_to_control_commands(safe_action, emergency_stop)
            
            # Update history
            self.action_history.append(safe_action)
            self.current_state = multimodal_data
            
            # Performance monitoring
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) % 30 == 0:  # Log every 30 frames
                avg_frame_time = np.mean(self.frame_times)
                fps = 1.0 / avg_frame_time
                self.logger.info(f"Average FPS: {fps:.2f}")
            
            return control_commands
            
        except Exception as e:
            self.logger.error(f"Error in navigation prediction: {e}")
            return self._get_safe_stop_action()
    
    def _action_to_control_commands(self, action: torch.Tensor, emergency_stop: bool) -> Dict[str, float]:
        """Convert model action to vehicle control commands."""
        action_cpu = action.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
        
        if emergency_stop:
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'brake': 1.0,
                'emergency_stop': True
            }
        
        # Extract individual actions
        steering = float(action_cpu[0])  # Already in [-1, 1]
        throttle = float(action_cpu[1])  # Already in [0, 1]
        brake = float(action_cpu[2])     # Already in [0, 1]
        
        return {
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'emergency_stop': False
        }
    
    def _get_safe_stop_action(self) -> Dict[str, float]:
        """Return safe stopping action."""
        return {
            'steering': 0.0,
            'throttle': 0.0,
            'brake': 1.0,
            'emergency_stop': True
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {}
        
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            metrics['average_fps'] = 1.0 / avg_frame_time
            metrics['frame_time_std'] = np.std(self.frame_times)
        
        if self.action_history:
            recent_actions = torch.stack(list(self.action_history))
            metrics['steering_smoothness'] = torch.std(recent_actions[:, 0, 0]).item()
            metrics['throttle_smoothness'] = torch.std(recent_actions[:, 0, 1]).item()
        
        return metrics


class NavigationSimulator:
    """
    Simulator for testing navigation controller with synthetic or recorded data.
    """
    
    def __init__(self, controller: NavigationController, config: Dict):
        """
        Initialize the navigation simulator.
        
        Args:
            controller: Navigation controller instance
            config: Configuration dictionary
        """
        self.controller = controller
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Simulation parameters
        self.simulation_fps = config.get('simulation_fps', 30)
        self.simulation_duration = config.get('simulation_duration', 60.0)  # seconds
        
        # Results tracking
        self.trajectory = []
        self.actions = []
        self.performance_metrics = []
    
    def run_simulation(self, test_data_path: str) -> Dict[str, Any]:
        """
        Run navigation simulation with test data.
        
        Args:
            test_data_path: Path to test dataset
        
        Returns:
            Simulation results
        """
        self.logger.info("Starting navigation simulation...")
        
        # Load test dataset
        test_dataset = MultimodalNavigationDataset(
            data_path=test_data_path,
            config=self.config,
            is_expert=False
        )
        
        start_time = time.time()
        frame_count = 0
        
        for i, sample in enumerate(test_dataset):
            if time.time() - start_time > self.simulation_duration:
                break
            
            # Convert sample to raw sensor data format
            raw_sensor_data = self._convert_sample_to_raw_data(sample)
            
            # Get navigation prediction
            control_commands = self.controller.predict_action(raw_sensor_data)
            
            # Store results
            self.actions.append(control_commands)
            self.trajectory.append({
                'timestamp': i / self.simulation_fps,
                'control_commands': control_commands,
                'sample_id': i
            })
            
            # Update performance metrics
            perf_metrics = self.controller.get_performance_metrics()
            perf_metrics['timestamp'] = i / self.simulation_fps
            self.performance_metrics.append(perf_metrics)
            
            frame_count += 1
            
            # Maintain simulation FPS
            target_time = start_time + frame_count / self.simulation_fps
            current_time = time.time()
            if current_time < target_time:
                time.sleep(target_time - current_time)
        
        simulation_time = time.time() - start_time
        self.logger.info(f"Simulation completed: {frame_count} frames in {simulation_time:.2f} seconds")
        
        return self._compile_simulation_results(simulation_time, frame_count)
    
    def _convert_sample_to_raw_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset sample to raw sensor data format."""
        raw_data = {}
        
        multimodal_data = sample.get('multimodal', {})
        
        # Camera data
        if 'camera' in multimodal_data:
            camera_tensor = multimodal_data['camera']
            # Convert CHW to HWC and scale to 0-255
            camera_image = camera_tensor.permute(1, 2, 0).numpy() * 255
            camera_image = camera_image.astype(np.uint8)
            raw_data['camera'] = camera_image
        
        # LiDAR data
        if 'lidar' in multimodal_data:
            raw_data['lidar'] = multimodal_data['lidar'].numpy()
        
        # Radar data (convert back to list format)
        if 'radar' in multimodal_data:
            radar_tensor = multimodal_data['radar']
            radar_detections = []
            for detection in radar_tensor:
                if detection.sum() > 0:  # Skip zero detections
                    radar_detections.append({
                        'range': detection[0].item(),
                        'azimuth': detection[1].item(),
                        'elevation': detection[2].item(),
                        'velocity': detection[3].item()
                    })
            raw_data['radar'] = radar_detections
        
        # GPS data
        if 'gps' in multimodal_data:
            gps_tensor = multimodal_data['gps']
            raw_data['gps'] = {
                'latitude': gps_tensor[0].item(),
                'longitude': gps_tensor[1].item(),
                'altitude': gps_tensor[2].item(),
                'heading': gps_tensor[3].item(),
                'pitch': gps_tensor[4].item(),
                'roll': gps_tensor[5].item(),
                'velocity_x': gps_tensor[6].item(),
                'velocity_y': gps_tensor[7].item(),
                'velocity_z': gps_tensor[8].item()
            }
        
        return raw_data
    
    def _compile_simulation_results(self, simulation_time: float, frame_count: int) -> Dict[str, Any]:
        """Compile and analyze simulation results."""
        results = {
            'simulation_time': simulation_time,
            'frame_count': frame_count,
            'average_fps': frame_count / simulation_time,
            'trajectory': self.trajectory,
            'actions': self.actions,
            'performance_metrics': self.performance_metrics
        }
        
        # Analyze actions
        if self.actions:
            steering_values = [action['steering'] for action in self.actions]
            throttle_values = [action['throttle'] for action in self.actions]
            brake_values = [action['brake'] for action in self.actions]
            emergency_stops = sum(1 for action in self.actions if action.get('emergency_stop', False))
            
            results['action_statistics'] = {
                'steering_mean': np.mean(steering_values),
                'steering_std': np.std(steering_values),
                'throttle_mean': np.mean(throttle_values),
                'throttle_std': np.std(throttle_values),
                'brake_mean': np.mean(brake_values),
                'brake_std': np.std(brake_values),
                'emergency_stop_count': emergency_stops,
                'emergency_stop_rate': emergency_stops / frame_count
            }
        
        # Analyze performance
        if self.performance_metrics:
            fps_values = [m.get('average_fps', 0) for m in self.performance_metrics if 'average_fps' in m]
            if fps_values:
                results['performance_statistics'] = {
                    'fps_mean': np.mean(fps_values),
                    'fps_std': np.std(fps_values),
                    'fps_min': np.min(fps_values),
                    'fps_max': np.max(fps_values)
                }
        
        return results
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save simulation results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Simulation results saved to {save_path}")


def main():
    """Main function for testing the navigation controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Navigation Controller')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output', type=str, default='navigation_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize navigation controller
    controller = NavigationController(config, args.model)
    
    # Run simulation
    simulator = NavigationSimulator(controller, config)
    results = simulator.run_simulation(args.test_data)
    
    # Save results
    simulator.save_results(results, args.output)
    
    # Print summary
    print("\nNavigation Test Results:")
    print(f"Simulation Time: {results['simulation_time']:.2f} seconds")
    print(f"Frames Processed: {results['frame_count']}")
    print(f"Average FPS: {results['average_fps']:.2f}")
    
    if 'action_statistics' in results:
        stats = results['action_statistics']
        print(f"Emergency Stops: {stats['emergency_stop_count']} ({stats['emergency_stop_rate']:.2%})")
        print(f"Average Steering: {stats['steering_mean']:.3f} ± {stats['steering_std']:.3f}")
        print(f"Average Throttle: {stats['throttle_mean']:.3f} ± {stats['throttle_std']:.3f}")


if __name__ == '__main__':
    main()
