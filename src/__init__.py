"""
Adversarial Inverse Reinforcement Learning for Autonomous Navigation

This package implements an advanced autonomous navigation system that combines
Inverse Reinforcement Learning (IRL) with adversarial training on multimodal sensor data.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models.adversarial_irl import AdversarialIRLAgent, MultimodalEncoder, RewardNetwork, PolicyNetwork, Discriminator
from .utils.data_loader import MultimodalNavigationDataset, SyntheticNavigationDataset
from .utils.metrics import compute_irl_metrics, compute_behavioral_cloning_metrics
from .utils.visualization import plot_training_curves, plot_reward_distributions
from .navigation.navigation_controller import NavigationController, NavigationSimulator

__all__ = [
    'AdversarialIRLAgent',
    'MultimodalEncoder', 
    'RewardNetwork',
    'PolicyNetwork',
    'Discriminator',
    'MultimodalNavigationDataset',
    'SyntheticNavigationDataset',
    'compute_irl_metrics',
    'compute_behavioral_cloning_metrics',
    'plot_training_curves',
    'plot_reward_distributions',
    'NavigationController',
    'NavigationSimulator'
]
