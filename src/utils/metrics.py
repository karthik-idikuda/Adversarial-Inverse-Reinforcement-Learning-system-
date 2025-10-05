"""
Metrics and evaluation utilities for IRL and navigation systems.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats


def compute_irl_metrics(
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict],
    reward_network: torch.nn.Module,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for IRL evaluation.
    
    Args:
        expert_trajectories: List of expert demonstration trajectories
        policy_trajectories: List of policy-generated trajectories
        reward_network: Trained reward network
        device: Device for computation
    
    Returns:
        Dictionary of computed metrics
    """
    if device is None:
        device = torch.device('cpu')
    
    metrics = {}
    
    # 1. Behavioral Cloning Metrics
    bc_metrics = compute_behavioral_cloning_metrics(expert_trajectories, policy_trajectories)
    metrics.update(bc_metrics)
    
    # 2. Reward Function Metrics
    reward_metrics = compute_reward_function_metrics(
        expert_trajectories, policy_trajectories, reward_network, device
    )
    metrics.update(reward_metrics)
    
    # 3. Trajectory Quality Metrics
    traj_metrics = compute_trajectory_quality_metrics(expert_trajectories, policy_trajectories)
    metrics.update(traj_metrics)
    
    # 4. Safety Metrics
    safety_metrics = compute_safety_metrics(policy_trajectories)
    metrics.update(safety_metrics)
    
    return metrics


def compute_behavioral_cloning_metrics(
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict]
) -> Dict[str, float]:
    """Compute behavioral cloning evaluation metrics."""
    
    if not expert_trajectories or not policy_trajectories:
        return {}
    
    # Extract actions
    expert_actions = []
    policy_actions = []
    
    for traj in expert_trajectories:
        if 'actions' in traj:
            expert_actions.extend(traj['actions'])
    
    for traj in policy_trajectories:
        if 'actions' in traj:
            policy_actions.extend(traj['actions'])
    
    if not expert_actions or not policy_actions:
        return {}
    
    # Convert to numpy arrays
    expert_actions = np.array(expert_actions)
    policy_actions = np.array(policy_actions)
    
    # Ensure same length for comparison
    min_length = min(len(expert_actions), len(policy_actions))
    expert_actions = expert_actions[:min_length]
    policy_actions = policy_actions[:min_length]
    
    metrics = {}
    
    # Mean Squared Error
    mse = mean_squared_error(expert_actions.flatten(), policy_actions.flatten())
    metrics['bc_mse'] = float(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(expert_actions.flatten(), policy_actions.flatten())
    metrics['bc_mae'] = float(mae)
    
    # Action-specific metrics
    if expert_actions.shape[1] >= 3:  # steering, throttle, brake
        # Steering accuracy
        steering_mse = mean_squared_error(expert_actions[:, 0], policy_actions[:, 0])
        metrics['steering_mse'] = float(steering_mse)
        
        # Throttle accuracy
        throttle_mse = mean_squared_error(expert_actions[:, 1], policy_actions[:, 1])
        metrics['throttle_mse'] = float(throttle_mse)
        
        # Brake accuracy
        brake_mse = mean_squared_error(expert_actions[:, 2], policy_actions[:, 2])
        metrics['brake_mse'] = float(brake_mse)
    
    # Correlation coefficient
    correlation, _ = stats.pearsonr(expert_actions.flatten(), policy_actions.flatten())
    metrics['action_correlation'] = float(correlation)
    
    return metrics


def compute_reward_function_metrics(
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict],
    reward_network: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Compute reward function evaluation metrics."""
    
    reward_network.eval()
    metrics = {}
    
    expert_rewards = []
    policy_rewards = []
    
    with torch.no_grad():
        # Compute rewards for expert trajectories
        for traj in expert_trajectories[:10]:  # Sample subset for efficiency
            if 'states' in traj and 'actions' in traj:
                states = torch.tensor(traj['states'], dtype=torch.float32).to(device)
                actions = torch.tensor(traj['actions'], dtype=torch.float32).to(device)
                
                rewards = reward_network(states, actions)
                expert_rewards.extend(rewards.cpu().numpy().flatten())
        
        # Compute rewards for policy trajectories
        for traj in policy_trajectories[:10]:  # Sample subset for efficiency
            if 'states' in traj and 'actions' in traj:
                states = torch.tensor(traj['states'], dtype=torch.float32).to(device)
                actions = torch.tensor(traj['actions'], dtype=torch.float32).to(device)
                
                rewards = reward_network(states, actions)
                policy_rewards.extend(rewards.cpu().numpy().flatten())
    
    if expert_rewards and policy_rewards:
        # Expert vs Policy reward comparison
        expert_reward_mean = np.mean(expert_rewards)
        policy_reward_mean = np.mean(policy_rewards)
        
        metrics['expert_reward_mean'] = float(expert_reward_mean)
        metrics['policy_reward_mean'] = float(policy_reward_mean)
        metrics['reward_gap'] = float(expert_reward_mean - policy_reward_mean)
        
        # Reward distribution metrics
        metrics['expert_reward_std'] = float(np.std(expert_rewards))
        metrics['policy_reward_std'] = float(np.std(policy_rewards))
        
        # Statistical test for reward difference
        if len(expert_rewards) > 1 and len(policy_rewards) > 1:
            t_stat, p_value = stats.ttest_ind(expert_rewards, policy_rewards)
            metrics['reward_ttest_pvalue'] = float(p_value)
    
    return metrics


def compute_trajectory_quality_metrics(
    expert_trajectories: List[Dict],
    policy_trajectories: List[Dict]
) -> Dict[str, float]:
    """Compute trajectory quality and similarity metrics."""
    
    metrics = {}
    
    if not expert_trajectories or not policy_trajectories:
        return metrics
    
    # Path smoothness metrics
    expert_smoothness = []
    policy_smoothness = []
    
    for traj in expert_trajectories:
        if 'actions' in traj:
            actions = np.array(traj['actions'])
            if len(actions) > 1:
                # Compute action smoothness (variance of action differences)
                action_diffs = np.diff(actions, axis=0)
                smoothness = np.mean(np.var(action_diffs, axis=0))
                expert_smoothness.append(smoothness)
    
    for traj in policy_trajectories:
        if 'actions' in traj:
            actions = np.array(traj['actions'])
            if len(actions) > 1:
                action_diffs = np.diff(actions, axis=0)
                smoothness = np.mean(np.var(action_diffs, axis=0))
                policy_smoothness.append(smoothness)
    
    if expert_smoothness and policy_smoothness:
        metrics['expert_smoothness'] = float(np.mean(expert_smoothness))
        metrics['policy_smoothness'] = float(np.mean(policy_smoothness))
        metrics['smoothness_ratio'] = float(np.mean(policy_smoothness) / np.mean(expert_smoothness))
    
    # Trajectory length comparison
    expert_lengths = [len(traj.get('actions', [])) for traj in expert_trajectories]
    policy_lengths = [len(traj.get('actions', [])) for traj in policy_trajectories]
    
    if expert_lengths and policy_lengths:
        metrics['expert_traj_length_mean'] = float(np.mean(expert_lengths))
        metrics['policy_traj_length_mean'] = float(np.mean(policy_lengths))
        metrics['traj_length_ratio'] = float(np.mean(policy_lengths) / np.mean(expert_lengths))
    
    return metrics


def compute_safety_metrics(policy_trajectories: List[Dict]) -> Dict[str, float]:
    """Compute safety-related metrics for policy trajectories."""
    
    metrics = {}
    
    if not policy_trajectories:
        return metrics
    
    # Safety violations
    hard_brake_count = 0
    extreme_steering_count = 0
    total_timesteps = 0
    
    for traj in policy_trajectories:
        if 'actions' in traj:
            actions = np.array(traj['actions'])
            total_timesteps += len(actions)
            
            if actions.shape[1] >= 3:  # steering, throttle, brake
                # Hard braking detection (brake > 0.8)
                hard_brakes = np.sum(actions[:, 2] > 0.8)
                hard_brake_count += hard_brakes
                
                # Extreme steering detection (|steering| > 0.9)
                extreme_steering = np.sum(np.abs(actions[:, 0]) > 0.9)
                extreme_steering_count += extreme_steering
    
    if total_timesteps > 0:
        metrics['hard_brake_frequency'] = float(hard_brake_count / total_timesteps)
        metrics['extreme_steering_frequency'] = float(extreme_steering_count / total_timesteps)
        
        # Combine into safety score (lower is better)
        safety_score = (hard_brake_count + extreme_steering_count) / total_timesteps
        metrics['safety_score'] = float(safety_score)
    
    return metrics


def compute_adversarial_robustness_metrics(
    model: torch.nn.Module,
    test_data: List[Dict],
    perturbation_magnitudes: List[float] = [0.01, 0.05, 0.1, 0.2],
    device: torch.device = None
) -> Dict[str, float]:
    """
    Compute adversarial robustness metrics.
    
    Args:
        model: Trained model to evaluate
        test_data: Test dataset
        perturbation_magnitudes: List of perturbation magnitudes to test
        device: Device for computation
    
    Returns:
        Dictionary of robustness metrics
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    metrics = {}
    
    for epsilon in perturbation_magnitudes:
        action_differences = []
        
        with torch.no_grad():
            for sample in test_data[:100]:  # Sample subset for efficiency
                if 'multimodal' in sample:
                    # Original prediction
                    multimodal_data = {
                        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sample['multimodal'].items()
                    }
                    
                    original_action = model.get_action(multimodal_data)
                    
                    # Add noise to inputs
                    perturbed_data = {}
                    for modality, data in multimodal_data.items():
                        if isinstance(data, torch.Tensor) and data.dtype == torch.float32:
                            noise = torch.randn_like(data) * epsilon
                            perturbed_data[modality] = data + noise
                        else:
                            perturbed_data[modality] = data
                    
                    # Perturbed prediction
                    perturbed_action = model.get_action(perturbed_data)
                    
                    # Compute action difference
                    action_diff = torch.norm(original_action - perturbed_action, dim=1)
                    action_differences.append(action_diff.item())
        
        if action_differences:
            metrics[f'robustness_epsilon_{epsilon}'] = float(np.mean(action_differences))
    
    return metrics


def compute_multimodal_importance_metrics(
    model: torch.nn.Module,
    test_data: List[Dict],
    device: torch.device = None
) -> Dict[str, float]:
    """
    Compute the importance of different modalities using ablation study.
    
    Args:
        model: Trained multimodal model
        test_data: Test dataset
        device: Device for computation
    
    Returns:
        Dictionary of modality importance scores
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    metrics = {}
    
    modalities = ['camera', 'lidar', 'radar', 'gps']
    baseline_errors = []
    
    with torch.no_grad():
        # Compute baseline performance with all modalities
        for sample in test_data[:50]:  # Sample subset
            if 'multimodal' in sample and 'actions' in sample:
                multimodal_data = {
                    k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in sample['multimodal'].items()
                }
                expert_action = sample['actions'].unsqueeze(0).to(device)
                
                predicted_action = model.get_action(multimodal_data)
                error = torch.norm(predicted_action - expert_action, dim=1).item()
                baseline_errors.append(error)
        
        baseline_error = np.mean(baseline_errors) if baseline_errors else 0.0
        
        # Test each modality ablation
        for modality in modalities:
            ablation_errors = []
            
            for sample in test_data[:50]:
                if 'multimodal' in sample and 'actions' in sample:
                    multimodal_data = {
                        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in sample['multimodal'].items()
                    }
                    expert_action = sample['actions'].unsqueeze(0).to(device)
                    
                    # Remove the modality
                    ablated_data = {k: v for k, v in multimodal_data.items() if k != modality}
                    
                    if ablated_data:  # Ensure we have at least one modality
                        try:
                            predicted_action = model.get_action(ablated_data)
                            error = torch.norm(predicted_action - expert_action, dim=1).item()
                            ablation_errors.append(error)
                        except:
                            # Skip if ablation causes errors
                            continue
            
            if ablation_errors:
                ablation_error = np.mean(ablation_errors)
                # Importance = increase in error when modality is removed
                importance = ablation_error - baseline_error
                metrics[f'{modality}_importance'] = float(importance)
    
    return metrics


def plot_metrics_comparison(
    metrics_dict: Dict[str, float],
    save_path: str = None
) -> None:
    """
    Plot comparison of different metrics.
    
    Args:
        metrics_dict: Dictionary of metrics to plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Group metrics by category
    bc_metrics = {k: v for k, v in metrics_dict.items() if 'bc_' in k or 'steering' in k or 'throttle' in k or 'brake' in k}
    reward_metrics = {k: v for k, v in metrics_dict.items() if 'reward' in k}
    safety_metrics = {k: v for k, v in metrics_dict.items() if 'safety' in k or 'brake_frequency' in k or 'steering_frequency' in k}
    robustness_metrics = {k: v for k, v in metrics_dict.items() if 'robustness' in k or 'importance' in k}
    
    metric_groups = [
        ('Behavioral Cloning', bc_metrics),
        ('Reward Function', reward_metrics),
        ('Safety Metrics', safety_metrics),
        ('Robustness/Importance', robustness_metrics)
    ]
    
    for i, (title, metrics) in enumerate(metric_groups):
        if metrics and i < len(axes):
            names = list(metrics.keys())
            values = list(metrics.values())
            
            axes[i].bar(range(len(names)), values)
            axes[i].set_title(title)
            axes[i].set_xticks(range(len(names)))
            axes[i].set_xticklabels(names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_metrics_report(
    metrics_dict: Dict[str, float],
    save_path: str = None
) -> str:
    """
    Create a comprehensive metrics report.
    
    Args:
        metrics_dict: Dictionary of computed metrics
        save_path: Optional path to save the report
    
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("INVERSE REINFORCEMENT LEARNING EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Behavioral Cloning Section
    report_lines.append("BEHAVIORAL CLONING METRICS:")
    report_lines.append("-" * 30)
    bc_metrics = {k: v for k, v in metrics_dict.items() if 'bc_' in k or any(action in k for action in ['steering', 'throttle', 'brake'])}
    for metric, value in bc_metrics.items():
        report_lines.append(f"{metric:25s}: {value:.6f}")
    report_lines.append("")
    
    # Reward Function Section
    report_lines.append("REWARD FUNCTION METRICS:")
    report_lines.append("-" * 30)
    reward_metrics = {k: v for k, v in metrics_dict.items() if 'reward' in k}
    for metric, value in reward_metrics.items():
        report_lines.append(f"{metric:25s}: {value:.6f}")
    report_lines.append("")
    
    # Safety Section
    report_lines.append("SAFETY METRICS:")
    report_lines.append("-" * 30)
    safety_metrics = {k: v for k, v in metrics_dict.items() if any(keyword in k for keyword in ['safety', 'brake_frequency', 'steering_frequency'])}
    for metric, value in safety_metrics.items():
        report_lines.append(f"{metric:25s}: {value:.6f}")
    report_lines.append("")
    
    # Robustness Section
    report_lines.append("ROBUSTNESS & IMPORTANCE METRICS:")
    report_lines.append("-" * 30)
    robustness_metrics = {k: v for k, v in metrics_dict.items() if any(keyword in k for keyword in ['robustness', 'importance'])}
    for metric, value in robustness_metrics.items():
        report_lines.append(f"{metric:25s}: {value:.6f}")
    report_lines.append("")
    
    # Summary
    report_lines.append("SUMMARY:")
    report_lines.append("-" * 30)
    if 'bc_mse' in metrics_dict:
        report_lines.append(f"Overall BC Performance: {'Good' if metrics_dict['bc_mse'] < 0.1 else 'Needs Improvement'}")
    if 'reward_gap' in metrics_dict:
        reward_gap = metrics_dict['reward_gap']
        report_lines.append(f"Reward Gap: {reward_gap:.4f} ({'Positive - Good' if reward_gap > 0 else 'Negative - Needs Work'})")
    if 'safety_score' in metrics_dict:
        safety_score = metrics_dict['safety_score']
        report_lines.append(f"Safety Score: {safety_score:.4f} ({'Good' if safety_score < 0.05 else 'Concerning'})")
    
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text
