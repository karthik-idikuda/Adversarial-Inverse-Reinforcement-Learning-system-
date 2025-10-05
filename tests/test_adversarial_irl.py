"""
Test suite for Adversarial IRL Navigation System
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import yaml
import tempfile
import shutil

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration directly
import config.fixed_config as config_module
CONFIG = config_module.CONFIG

# Import model components - using direct imports to avoid cv2 issues
# from src.models.adversarial_irl import MultimodalEncoder
# Import test components using components from test_components
from tests.test_components import (
    MultimodalEncoder, AdversarialIRLAgent, PolicyNetwork, RewardNetwork,
    NavigationController, SyntheticNavigationDataset, 
    compute_behavioral_cloning_metrics
)


class TestMultimodalEncoder(unittest.TestCase):
    """Test cases for multimodal encoder."""
    
    def setUp(self):
        self.config = {
            'fusion_dim': 256,
            'camera_size': (224, 224),
            'max_lidar_points': 1024,
            'max_radar_detections': 32
        }
        self.encoder = MultimodalEncoder(self.config)
        
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        self.assertIsInstance(self.encoder, MultimodalEncoder)
        self.assertEqual(self.encoder.fusion_dim, 256)
    
    def test_camera_encoding(self):
        """Test camera data encoding."""
        batch_size = 2
        camera_data = torch.randn(batch_size, 3, 224, 224)
        multimodal_data = {'camera': camera_data}
        
        output = self.encoder(multimodal_data)
        self.assertEqual(output.shape, (batch_size, self.config['fusion_dim']))
    
    def test_lidar_encoding(self):
        """Test LiDAR data encoding."""
        batch_size = 2
        lidar_data = torch.randn(batch_size, 1024, 3)
        multimodal_data = {'lidar': lidar_data}
        
        output = self.encoder(multimodal_data)
        self.assertEqual(output.shape, (batch_size, self.config['fusion_dim']))
    
    def test_multimodal_fusion(self):
        """Test complete multimodal fusion."""
        batch_size = 2
        multimodal_data = {
            'camera': torch.randn(batch_size, 3, 224, 224),
            'lidar': torch.randn(batch_size, 1024, 3),
            'radar': torch.randn(batch_size, 32, 4),
            'gps': torch.randn(batch_size, 9)
        }
        
        output = self.encoder(multimodal_data)
        self.assertEqual(output.shape, (batch_size, self.config['fusion_dim']))


class TestAdversarialIRLAgent(unittest.TestCase):
    """Test cases for the complete IRL agent."""
    
    def setUp(self):
        self.config = {
            'fusion_dim': 256,
            'action_dim': 3,
            'hidden_dim': 128,
            'camera_size': (224, 224),
            'max_lidar_points': 1024,
            'max_radar_detections': 32,
            'adversarial_epsilon': 0.1,
            'adversarial_alpha': 0.01,
            'num_attack_steps': 5
        }
        self.agent = AdversarialIRLAgent(self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsInstance(self.agent, AdversarialIRLAgent)
        self.assertIsInstance(self.agent.multimodal_encoder, MultimodalEncoder)
        self.assertIsInstance(self.agent.reward_network, RewardNetwork)
        self.assertIsInstance(self.agent.policy_network, PolicyNetwork)
    
    def test_action_prediction(self):
        """Test action prediction."""
        batch_size = 2
        multimodal_data = {
            'camera': torch.randn(batch_size, 3, 224, 224),
            'lidar': torch.randn(batch_size, 1024, 3),
            'gps': torch.randn(batch_size, 9)
        }
        
        action = self.agent.get_action(multimodal_data)
        self.assertEqual(action.shape, (batch_size, self.config['action_dim']))
        
        # Check action bounds
        self.assertTrue(torch.all(action[:, 0] >= -1) and torch.all(action[:, 0] <= 1))  # steering
        self.assertTrue(torch.all(action[:, 1] >= 0) and torch.all(action[:, 1] <= 1))   # throttle
        self.assertTrue(torch.all(action[:, 2] >= 0) and torch.all(action[:, 2] <= 1))   # brake
    
    def test_reward_prediction(self):
        """Test reward prediction."""
        batch_size = 2
        multimodal_data = {
            'camera': torch.randn(batch_size, 3, 224, 224),
            'gps': torch.randn(batch_size, 9)
        }
        actions = torch.randn(batch_size, self.config['action_dim'])
        
        rewards = self.agent.get_reward(multimodal_data, actions)
        self.assertEqual(rewards.shape, (batch_size, 1))


class TestSyntheticDataset(unittest.TestCase):
    """Test cases for synthetic dataset generation."""
    
    def setUp(self):
        self.config = {
            'camera_size': (3, 224, 224),  # Fixed: include channels
            'max_lidar_points': 1000,
            'max_radar_detections': 50
        }
        self.dataset = SyntheticNavigationDataset(self.config, num_samples=100)
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertEqual(len(self.dataset), 100)
    
    def test_sample_structure(self):
        """Test sample structure."""
        sample = self.dataset[0]
        
        self.assertIn('multimodal', sample)
        self.assertIn('actions', sample)
        self.assertIn('trajectory_id', sample)
        self.assertIn('timestamp', sample)
        
        # Check multimodal data
        multimodal = sample['multimodal']
        self.assertIn('camera', multimodal)
        self.assertIn('lidar', multimodal)
        self.assertIn('radar', multimodal)
        self.assertIn('gps', multimodal)
        
        # Check shapes
        self.assertEqual(multimodal['camera'].shape, (3, 224, 224))
        self.assertEqual(multimodal['lidar'].shape, (self.config['max_lidar_points'], 3))
        self.assertEqual(multimodal['radar'].shape, (self.config['max_radar_detections'], 4))
        self.assertEqual(multimodal['gps'].shape, (9,))
        self.assertEqual(sample['actions'].shape, (3,))


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        # Create synthetic trajectories for testing
        self.expert_trajectories = []
        self.policy_trajectories = []
        
        for i in range(5):
            expert_traj = {
                'actions': np.random.randn(50, 3),
                'states': np.random.randn(50, 256),
                'positions': np.random.randn(50, 2)
            }
            self.expert_trajectories.append(expert_traj)
            
            policy_traj = {
                'actions': np.random.randn(45, 3),  # Slightly different length
                'states': np.random.randn(45, 256),
                'positions': np.random.randn(45, 2)
            }
            self.policy_trajectories.append(policy_traj)
    
    def test_behavioral_cloning_metrics(self):
        """Test behavioral cloning metrics computation."""
        metrics = compute_behavioral_cloning_metrics(
            self.expert_trajectories, 
            self.policy_trajectories
        )
        
        self.assertIn('bc_mse', metrics)
        self.assertIn('bc_mae', metrics)
        self.assertIn('action_correlation', metrics)
        self.assertIn('steering_mse', metrics)
        self.assertIn('throttle_mse', metrics)
        self.assertIn('brake_mse', metrics)
        
        # Check that metrics are reasonable
        self.assertGreaterEqual(metrics['bc_mse'], 0)
        self.assertGreaterEqual(metrics['bc_mae'], 0)
        self.assertGreaterEqual(metrics['action_correlation'], -1)
        self.assertLessEqual(metrics['action_correlation'], 1)


class TestNavigationController(unittest.TestCase):
    """Test cases for navigation controller."""
    
    def setUp(self):
        # Create a temporary model file
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.pth"
        
        # Create and save a test model
        config = {
            'fusion_dim': 256,
            'action_dim': 3,
            'hidden_dim': 128,
            'camera_size': (224, 224),
            'max_lidar_points': 1024,
            'max_radar_detections': 32
        }
        
        agent = AdversarialIRLAgent(config)
        checkpoint = {
            'model_state_dict': agent.state_dict(),
            'config': config
        }
        torch.save(checkpoint, self.model_path)
        
        self.controller = NavigationController(config, str(self.model_path))
    
    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertIsInstance(self.controller, NavigationController)
        self.assertIsInstance(self.controller.agent, AdversarialIRLAgent)
    
    def test_sensor_data_processing(self):
        """Test sensor data processing."""
        raw_sensor_data = {
            'camera': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'lidar': np.random.randn(1000, 3),
            'radar': [
                {'range': 10.0, 'azimuth': 0.1, 'elevation': 0.0, 'velocity': 5.0},
                {'range': 20.0, 'azimuth': -0.2, 'elevation': 0.1, 'velocity': -2.0}
            ],
            'gps': {
                'latitude': 37.7749,
                'longitude': -122.4194,
                'altitude': 50.0,
                'heading': 1.57,
                'pitch': 0.0,
                'roll': 0.0,
                'velocity_x': 10.0,
                'velocity_y': 20.0,
                'velocity_z': 0.0
            }
        }
        
        processed_data = self.controller.process_sensor_data(raw_sensor_data)
        
        self.assertIn('camera', processed_data)
        self.assertIn('lidar', processed_data)
        self.assertIn('radar', processed_data)
        self.assertIn('gps', processed_data)
        
        # Check shapes
        self.assertEqual(processed_data['camera'].shape, (1, 3, 224, 224))
        self.assertEqual(processed_data['lidar'].shape[0], 1)  # Batch dimension
        self.assertEqual(processed_data['radar'].shape, (1, 64, 4))  # Padded
        self.assertEqual(processed_data['gps'].shape, (1, 9))
    
    def test_action_prediction(self):
        """Test action prediction."""
        raw_sensor_data = {
            'camera': np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            'gps': {
                'latitude': 37.7749,
                'longitude': -122.4194,
                'altitude': 50.0,
                'heading': 1.57,
                'pitch': 0.0,
                'roll': 0.0,
                'velocity_x': 10.0,
                'velocity_y': 20.0,
                'velocity_z': 0.0
            }
        }
        
        control_commands = self.controller.predict_action(raw_sensor_data)
        
        self.assertIn('steering', control_commands)
        self.assertIn('throttle', control_commands)
        self.assertIn('brake', control_commands)
        self.assertIn('emergency_stop', control_commands)
        
        # Check value ranges
        self.assertGreaterEqual(control_commands['steering'], -1.0)
        self.assertLessEqual(control_commands['steering'], 1.0)
        self.assertGreaterEqual(control_commands['throttle'], 0.0)
        self.assertLessEqual(control_commands['throttle'], 1.0)
        self.assertGreaterEqual(control_commands['brake'], 0.0)
        self.assertLessEqual(control_commands['brake'], 1.0)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = {
            'fusion_dim': 128,  # Smaller for faster testing
            'action_dim': 3,
            'hidden_dim': 64,
            'camera_size': (3, 64, 64),  # Fixed: include channels and smaller for faster testing
            'max_lidar_points': 512,
            'max_radar_detections': 16,
            'batch_size': 4,
            'learning_rate': 0.001,
            'adversarial_epsilon': 0.05
        }
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_training_pipeline(self):
        """Test the training pipeline with synthetic data."""
        # Create synthetic dataset
        dataset = SyntheticNavigationDataset(self.config, num_samples=20)
        
        # Create agent
        agent = AdversarialIRLAgent(self.config)
        
        # Test forward pass
        sample = dataset[0]
        multimodal_data = {k: v.unsqueeze(0) for k, v in sample['multimodal'].items()}
        
        # Test encoding
        state = agent.encode_multimodal_state(multimodal_data)
        self.assertEqual(state.shape, (1, self.config['fusion_dim']))
        
        # Test action prediction
        action = agent.get_action(multimodal_data)
        self.assertEqual(action.shape, (1, self.config['action_dim']))
        
        # Test reward prediction
        reward = agent.get_reward(multimodal_data, action)
        self.assertEqual(reward.shape, (1, 1))
        
        # Test discriminator
        prob = agent.discriminate(multimodal_data, action)
        self.assertEqual(prob.shape, (1, 1))
        self.assertTrue(0 <= prob.item() <= 1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
