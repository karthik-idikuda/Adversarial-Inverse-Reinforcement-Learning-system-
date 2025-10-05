"""
Visualization utilities for IRL training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
from matplotlib.animation import FuncAnimation
import cv2


def plot_training_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> None:
    """
    Plot training and validation curves for multiple loss components.
    
    Args:
        train_losses: Dictionary of training loss lists
        val_losses: Dictionary of validation loss lists
        save_path: Optional path to save the plot
        title: Title for the plot
    """
    plt.style.use('seaborn-v0_8')
    
    # Determine number of subplots needed
    all_loss_names = set(train_losses.keys()) | set(val_losses.keys())
    num_losses = len(all_loss_names)
    
    if num_losses == 0:
        print("No loss data to plot")
        return
    
    # Create subplots
    cols = min(3, num_losses)
    rows = (num_losses + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_losses == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each loss component
    for i, loss_name in enumerate(sorted(all_loss_names)):
        ax = axes[i] if i < len(axes) else plt.gca()
        
        # Plot training curve
        if loss_name in train_losses and train_losses[loss_name]:
            epochs = range(1, len(train_losses[loss_name]) + 1)
            ax.plot(epochs, train_losses[loss_name], 
                   label=f'Train {loss_name}', linewidth=2, alpha=0.8)
        
        # Plot validation curve
        if loss_name in val_losses and val_losses[loss_name]:
            epochs = range(1, len(val_losses[loss_name]) + 1)
            ax.plot(epochs, val_losses[loss_name], 
                   label=f'Val {loss_name}', linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{loss_name.replace("_", " ").title()} Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(num_losses, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reward_distributions(
    expert_rewards: List[float],
    policy_rewards: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot reward distributions for expert and policy trajectories.
    
    Args:
        expert_rewards: List of expert trajectory rewards
        policy_rewards: List of policy trajectory rewards
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Histogram comparison
    plt.subplot(1, 2, 1)
    plt.hist(expert_rewards, bins=30, alpha=0.7, label='Expert', density=True, color='blue')
    plt.hist(policy_rewards, bins=30, alpha=0.7, label='Policy', density=True, color='red')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.title('Reward Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot comparison
    plt.subplot(1, 2, 2)
    data = [expert_rewards, policy_rewards]
    labels = ['Expert', 'Policy']
    plt.boxplot(data, labels=labels)
    plt.ylabel('Reward')
    plt.title('Reward Box Plot Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_action_comparison(
    expert_actions: np.ndarray,
    policy_actions: np.ndarray,
    action_names: List[str] = ['Steering', 'Throttle', 'Brake'],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between expert and policy actions.
    
    Args:
        expert_actions: Expert actions array (N, action_dim)
        policy_actions: Policy actions array (N, action_dim)
        action_names: Names of action dimensions
        save_path: Optional path to save the plot
    """
    action_dim = min(expert_actions.shape[1], policy_actions.shape[1], len(action_names))
    
    fig, axes = plt.subplots(2, action_dim, figsize=(5 * action_dim, 8))
    if action_dim == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(action_dim):
        # Time series comparison
        axes[0, i].plot(expert_actions[:100, i], label='Expert', alpha=0.8, linewidth=2)
        axes[0, i].plot(policy_actions[:100, i], label='Policy', alpha=0.8, linewidth=2, linestyle='--')
        axes[0, i].set_title(f'{action_names[i]} Time Series')
        axes[0, i].set_xlabel('Time Step')
        axes[0, i].set_ylabel(action_names[i])
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1, i].scatter(expert_actions[:, i], policy_actions[:, i], alpha=0.6, s=20)
        
        # Add diagonal line for perfect correlation
        min_val = min(expert_actions[:, i].min(), policy_actions[:, i].min())
        max_val = max(expert_actions[:, i].max(), policy_actions[:, i].max())
        axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[1, i].set_xlabel(f'Expert {action_names[i]}')
        axes[1, i].set_ylabel(f'Policy {action_names[i]}')
        axes[1, i].set_title(f'{action_names[i]} Correlation')
        axes[1, i].grid(True, alpha=0.3)
        
        # Calculate and display correlation
        correlation = np.corrcoef(expert_actions[:, i], policy_actions[:, i])[0, 1]
        axes[1, i].text(0.05, 0.95, f'r = {correlation:.3f}', 
                       transform=axes[1, i].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_multimodal_attention(
    attention_weights: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot attention weights for different modalities.
    
    Args:
        attention_weights: Dictionary of attention weights for each modality
        save_path: Optional path to save the plot
    """
    modalities = list(attention_weights.keys())
    num_modalities = len(modalities)
    
    fig, axes = plt.subplots(1, num_modalities, figsize=(4 * num_modalities, 4))
    if num_modalities == 1:
        axes = [axes]
    
    for i, modality in enumerate(modalities):
        weights = attention_weights[modality]
        
        if len(weights.shape) == 1:
            # 1D attention weights
            axes[i].bar(range(len(weights)), weights)
            axes[i].set_xlabel('Feature Index')
        elif len(weights.shape) == 2:
            # 2D attention weights (e.g., spatial attention)
            im = axes[i].imshow(weights, cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=axes[i])
        
        axes[i].set_title(f'{modality.title()} Attention')
        axes[i].set_ylabel('Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_trajectory_comparison(
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict],
    max_trajectories: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Plot trajectory comparisons in 2D space.
    
    Args:
        expert_trajectories: List of expert trajectory dictionaries
        policy_trajectories: List of policy trajectory dictionaries
        max_trajectories: Maximum number of trajectories to plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot individual trajectories
    plt.subplot(1, 3, 1)
    for i, traj in enumerate(expert_trajectories[:max_trajectories]):
        if 'positions' in traj:
            positions = np.array(traj['positions'])
            plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2, 
                    label='Expert' if i == 0 else '')
    
    for i, traj in enumerate(policy_trajectories[:max_trajectories]):
        if 'positions' in traj:
            positions = np.array(traj['positions'])
            plt.plot(positions[:, 0], positions[:, 1], 'r--', alpha=0.7, linewidth=2,
                    label='Policy' if i == 0 else '')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot trajectory length distribution
    plt.subplot(1, 3, 2)
    expert_lengths = [len(traj.get('actions', [])) for traj in expert_trajectories]
    policy_lengths = [len(traj.get('actions', [])) for traj in policy_trajectories]
    
    plt.hist(expert_lengths, bins=20, alpha=0.7, label='Expert', density=True)
    plt.hist(policy_lengths, bins=20, alpha=0.7, label='Policy', density=True)
    plt.xlabel('Trajectory Length')
    plt.ylabel('Density')
    plt.title('Trajectory Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot speed comparison
    plt.subplot(1, 3, 3)
    expert_speeds = []
    policy_speeds = []
    
    for traj in expert_trajectories:
        if 'actions' in traj:
            actions = np.array(traj['actions'])
            if actions.shape[1] >= 2:  # Has throttle
                expert_speeds.extend(actions[:, 1])  # Throttle as proxy for speed
    
    for traj in policy_trajectories:
        if 'actions' in traj:
            actions = np.array(traj['actions'])
            if actions.shape[1] >= 2:
                policy_speeds.extend(actions[:, 1])
    
    if expert_speeds and policy_speeds:
        plt.hist(expert_speeds, bins=20, alpha=0.7, label='Expert', density=True)
        plt.hist(policy_speeds, bins=20, alpha=0.7, label='Policy', density=True)
    
    plt.xlabel('Speed (Throttle)')
    plt.ylabel('Density')
    plt.title('Speed Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_adversarial_robustness(
    robustness_metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot adversarial robustness metrics.
    
    Args:
        robustness_metrics: Dictionary of robustness metrics
        save_path: Optional path to save the plot
    """
    # Extract epsilon values and corresponding robustness scores
    epsilon_metrics = {k: v for k, v in robustness_metrics.items() if 'robustness_epsilon_' in k}
    
    if not epsilon_metrics:
        print("No adversarial robustness metrics found")
        return
    
    epsilons = []
    scores = []
    
    for key, score in epsilon_metrics.items():
        epsilon = float(key.split('_')[-1])
        epsilons.append(epsilon)
        scores.append(score)
    
    # Sort by epsilon
    sorted_data = sorted(zip(epsilons, scores))
    epsilons, scores = zip(*sorted_data)
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epsilons, scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Perturbation Magnitude (ε)')
    plt.ylabel('Action Deviation')
    plt.title('Adversarial Robustness')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot modality importance if available
    importance_metrics = {k: v for k, v in robustness_metrics.items() if '_importance' in k}
    
    if importance_metrics:
        plt.subplot(1, 2, 2)
        modalities = [k.replace('_importance', '') for k in importance_metrics.keys()]
        importance_scores = list(importance_metrics.values())
        
        bars = plt.bar(modalities, importance_scores)
        plt.xlabel('Modality')
        plt.ylabel('Importance Score')
        plt.title('Modality Importance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Color bars based on importance
        max_importance = max(importance_scores)
        for bar, score in zip(bars, importance_scores):
            normalized_score = score / max_importance if max_importance > 0 else 0
            bar.set_color(plt.cm.viridis(normalized_score))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_dashboard(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive training dashboard with multiple visualizations.
    
    Args:
        train_metrics: Training metrics over time
        val_metrics: Validation metrics over time
        expert_trajectories: Expert demonstration trajectories
        policy_trajectories: Policy-generated trajectories
        save_path: Optional path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Training curves (top row)
    ax1 = plt.subplot(3, 4, 1)
    if 'reward_loss' in train_metrics:
        plt.plot(train_metrics['reward_loss'], label='Train', linewidth=2)
    if 'reward_loss' in val_metrics:
        plt.plot(val_metrics['reward_loss'], label='Val', linewidth=2, linestyle='--')
    plt.title('Reward Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    if 'policy_loss' in train_metrics:
        plt.plot(train_metrics['policy_loss'], label='Train', linewidth=2)
    if 'policy_loss' in val_metrics:
        plt.plot(val_metrics['policy_loss'], label='Val', linewidth=2, linestyle='--')
    plt.title('Policy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 4, 3)
    if 'discriminator_loss' in train_metrics:
        plt.plot(train_metrics['discriminator_loss'], label='Train', linewidth=2)
    if 'discriminator_loss' in val_metrics:
        plt.plot(val_metrics['discriminator_loss'], label='Val', linewidth=2, linestyle='--')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 4, 4)
    if 'adversarial_loss' in train_metrics:
        plt.plot(train_metrics['adversarial_loss'], label='Train', linewidth=2)
    plt.title('Adversarial Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Action comparison (middle row)
    if expert_trajectories and policy_trajectories:
        expert_actions = []
        policy_actions = []
        
        for traj in expert_trajectories[:10]:
            if 'actions' in traj:
                expert_actions.extend(traj['actions'])
        
        for traj in policy_trajectories[:10]:
            if 'actions' in traj:
                policy_actions.extend(traj['actions'])
        
        if expert_actions and policy_actions:
            expert_actions = np.array(expert_actions)
            policy_actions = np.array(policy_actions)
            min_length = min(len(expert_actions), len(policy_actions))
            expert_actions = expert_actions[:min_length]
            policy_actions = policy_actions[:min_length]
            
            action_names = ['Steering', 'Throttle', 'Brake']
            for i in range(min(3, expert_actions.shape[1])):
                ax = plt.subplot(3, 4, 5 + i)
                plt.scatter(expert_actions[:, i], policy_actions[:, i], alpha=0.6, s=10)
                
                # Perfect correlation line
                min_val = min(expert_actions[:, i].min(), policy_actions[:, i].min())
                max_val = max(expert_actions[:, i].max(), policy_actions[:, i].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                plt.xlabel(f'Expert {action_names[i]}')
                plt.ylabel(f'Policy {action_names[i]}')
                plt.title(f'{action_names[i]} Correlation')
                plt.grid(True, alpha=0.3)
                
                # Calculate correlation
                correlation = np.corrcoef(expert_actions[:, i], policy_actions[:, i])[0, 1]
                plt.text(0.05, 0.95, f'r = {correlation:.3f}', 
                        transform=ax.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Trajectory visualization (bottom row)
    ax9 = plt.subplot(3, 4, 9)
    for i, traj in enumerate(expert_trajectories[:3]):
        if 'positions' in traj:
            positions = np.array(traj['positions'])
            plt.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=2,
                    label='Expert' if i == 0 else '')
    
    for i, traj in enumerate(policy_trajectories[:3]):
        if 'positions' in traj:
            positions = np.array(traj['positions'])
            plt.plot(positions[:, 0], positions[:, 1], 'r--', alpha=0.7, linewidth=2,
                    label='Policy' if i == 0 else '')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Sample Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Performance summary (bottom right)
    ax10 = plt.subplot(3, 4, 10)
    ax10.text(0.1, 0.8, "Training Summary", fontsize=16, fontweight='bold')
    
    if train_metrics:
        latest_epoch = max(len(losses) for losses in train_metrics.values()) if train_metrics else 0
        ax10.text(0.1, 0.6, f"Epochs Trained: {latest_epoch}", fontsize=12)
        
        if 'policy_loss' in train_metrics and train_metrics['policy_loss']:
            latest_policy_loss = train_metrics['policy_loss'][-1]
            ax10.text(0.1, 0.5, f"Latest Policy Loss: {latest_policy_loss:.4f}", fontsize=12)
        
        if 'reward_loss' in train_metrics and train_metrics['reward_loss']:
            latest_reward_loss = train_metrics['reward_loss'][-1]
            ax10.text(0.1, 0.4, f"Latest Reward Loss: {latest_reward_loss:.4f}", fontsize=12)
    
    ax10.set_xlim(0, 1)
    ax10.set_ylim(0, 1)
    ax10.axis('off')
    
    plt.suptitle('Adversarial IRL Training Dashboard', fontsize=20, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_multimodal_data(
    sample: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a sample of multimodal sensor data.
    
    Args:
        sample: Dictionary containing multimodal sensor data
        save_path: Optional path to save the visualization
    """
    multimodal_data = sample.get('multimodal', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Camera image
    if 'camera' in multimodal_data:
        camera_data = multimodal_data['camera']
        if isinstance(camera_data, torch.Tensor):
            # Convert CHW to HWC for display
            camera_image = camera_data.permute(1, 2, 0).numpy()
            axes[0, 0].imshow(camera_image)
            axes[0, 0].set_title('Camera Image')
            axes[0, 0].axis('off')
    
    # LiDAR point cloud (bird's eye view)
    if 'lidar' in multimodal_data:
        lidar_data = multimodal_data['lidar']
        if isinstance(lidar_data, torch.Tensor):
            points = lidar_data.numpy()
            # Plot top-down view (X-Y plane)
            axes[0, 1].scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                              s=1, cmap='viridis', alpha=0.6)
            axes[0, 1].set_title('LiDAR Point Cloud (Top View)')
            axes[0, 1].set_xlabel('X (m)')
            axes[0, 1].set_ylabel('Y (m)')
            axes[0, 1].axis('equal')
    
    # Radar detections
    if 'radar' in multimodal_data:
        radar_data = multimodal_data['radar']
        if isinstance(radar_data, torch.Tensor):
            detections = radar_data.numpy()
            # Convert polar to cartesian for visualization
            ranges = detections[:, 0]
            azimuths = detections[:, 1]
            
            # Filter out zero detections
            valid_mask = ranges > 0
            if valid_mask.any():
                x = ranges[valid_mask] * np.sin(azimuths[valid_mask])
                y = ranges[valid_mask] * np.cos(azimuths[valid_mask])
                
                axes[1, 0].scatter(x, y, c=detections[valid_mask, 3], 
                                  cmap='coolwarm', s=50, alpha=0.8)
                axes[1, 0].set_title('Radar Detections')
                axes[1, 0].set_xlabel('X (m)')
                axes[1, 0].set_ylabel('Y (m)')
                axes[1, 0].axis('equal')
    
    # GPS/IMU data
    if 'gps' in multimodal_data:
        gps_data = multimodal_data['gps']
        if isinstance(gps_data, torch.Tensor):
            gps_values = gps_data.numpy()
            gps_labels = ['Lat', 'Lon', 'Alt', 'Heading', 'Pitch', 'Roll', 'Vel_X', 'Vel_Y', 'Vel_Z']
            
            bars = axes[1, 1].bar(range(len(gps_values)), gps_values)
            axes[1, 1].set_title('GPS/IMU Data')
            axes[1, 1].set_xticks(range(len(gps_labels)))
            axes[1, 1].set_xticklabels(gps_labels, rotation=45)
            axes[1, 1].set_ylabel('Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
