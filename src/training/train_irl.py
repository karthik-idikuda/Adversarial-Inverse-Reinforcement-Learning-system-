"""
Training script for Adversarial Inverse Reinforcement Learning
with Multimodal Data for Autonomous Navigation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.adversarial_irl import AdversarialIRLAgent
from utils.data_loader import MultimodalNavigationDataset
from utils.metrics import compute_irl_metrics
from utils.visualization import plot_training_curves


class AdversarialIRLTrainer:
    """Trainer for Adversarial IRL with multimodal data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize model
        self.agent = AdversarialIRLAgent(config).to(self.device)
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize data loaders
        self.setup_data_loaders()
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Loss weights
        self.reward_loss_weight = config.get('reward_loss_weight', 1.0)
        self.policy_loss_weight = config.get('policy_loss_weight', 1.0)
        self.adversarial_loss_weight = config.get('adversarial_loss_weight', 0.1)
        self.discriminator_loss_weight = config.get('discriminator_loss_weight', 1.0)
        
    def setup_logging(self):
        """Setup logging and wandb."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('project_name', 'adversarial-irl-navigation'),
                config=self.config,
                name=self.config.get('run_name', 'adversarial_irl_run')
            )
    
    def setup_optimizers(self):
        """Setup optimizers for different components."""
        lr = self.config.get('learning_rate', 1e-4)
        
        # Separate optimizers for different components
        self.reward_optimizer = optim.Adam(
            self.agent.reward_network.parameters(), 
            lr=lr, 
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        self.policy_optimizer = optim.Adam(
            list(self.agent.multimodal_encoder.parameters()) + 
            list(self.agent.policy_network.parameters()),
            lr=lr,
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.agent.discriminator.parameters(),
            lr=lr * 0.5,  # Slower learning for discriminator
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate schedulers
        self.reward_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.reward_optimizer, mode='min', patience=10, factor=0.5
        )
        self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.policy_optimizer, mode='min', patience=10, factor=0.5
        )
        self.discriminator_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.discriminator_optimizer, mode='min', patience=10, factor=0.5
        )
    
    def setup_data_loaders(self):
        """Setup data loaders for expert demonstrations and validation."""
        # Expert demonstration dataset
        expert_dataset = MultimodalNavigationDataset(
            data_path=self.config['expert_data_path'],
            config=self.config,
            is_expert=True
        )
        
        # Validation dataset
        val_dataset = MultimodalNavigationDataset(
            data_path=self.config['validation_data_path'],
            config=self.config,
            is_expert=False
        )
        
        self.expert_loader = DataLoader(
            expert_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.logger.info(f"Loaded {len(expert_dataset)} expert demonstrations")
        self.logger.info(f"Loaded {len(val_dataset)} validation samples")
    
    def compute_reward_loss(self, expert_batch: Dict, policy_batch: Dict) -> torch.Tensor:
        """
        Compute reward loss using Maximum Entropy IRL objective.
        
        Args:
            expert_batch: Batch of expert demonstrations
            policy_batch: Batch of policy-generated trajectories
        
        Returns:
            Reward loss
        """
        # Expert rewards
        expert_multimodal = {k: v.to(self.device) for k, v in expert_batch['multimodal'].items()}
        expert_actions = expert_batch['actions'].to(self.device)
        expert_rewards = self.agent.get_reward(expert_multimodal, expert_actions)
        
        # Policy rewards
        policy_multimodal = {k: v.to(self.device) for k, v in policy_batch['multimodal'].items()}
        policy_actions = policy_batch['actions'].to(self.device)
        policy_rewards = self.agent.get_reward(policy_multimodal, policy_actions)
        
        # Maximum margin loss: expert rewards should be higher than policy rewards
        margin = 1.0
        loss = torch.clamp(margin + policy_rewards.mean() - expert_rewards.mean(), min=0.0)
        
        # Add regularization
        l2_reg = 0.01 * (expert_rewards.pow(2).mean() + policy_rewards.pow(2).mean())
        
        return loss + l2_reg
    
    def compute_policy_loss(self, expert_batch: Dict) -> torch.Tensor:
        """
        Compute policy loss using behavioral cloning with the learned reward.
        
        Args:
            expert_batch: Batch of expert demonstrations
        
        Returns:
            Policy loss
        """
        expert_multimodal = {k: v.to(self.device) for k, v in expert_batch['multimodal'].items()}
        expert_actions = expert_batch['actions'].to(self.device)
        
        # Get policy actions
        predicted_actions = self.agent.get_action(expert_multimodal)
        
        # Behavioral cloning loss
        bc_loss = nn.MSELoss()(predicted_actions, expert_actions)
        
        # Reward-weighted loss (higher weight for higher reward states)
        with torch.no_grad():
            rewards = self.agent.get_reward(expert_multimodal, expert_actions)
            weights = torch.softmax(rewards.squeeze(), dim=0).detach()
        
        weighted_bc_loss = (weights * nn.MSELoss(reduction='none')(predicted_actions, expert_actions).mean(dim=1)).mean()
        
        return 0.7 * bc_loss + 0.3 * weighted_bc_loss
    
    def compute_discriminator_loss(self, expert_batch: Dict, policy_batch: Dict) -> torch.Tensor:
        """
        Compute discriminator loss for adversarial training.
        
        Args:
            expert_batch: Batch of expert demonstrations
            policy_batch: Batch of policy-generated trajectories
        
        Returns:
            Discriminator loss
        """
        # Expert data
        expert_multimodal = {k: v.to(self.device) for k, v in expert_batch['multimodal'].items()}
        expert_actions = expert_batch['actions'].to(self.device)
        expert_probs = self.agent.discriminate(expert_multimodal, expert_actions)
        
        # Policy data
        policy_multimodal = {k: v.to(self.device) for k, v in policy_batch['multimodal'].items()}
        policy_actions = policy_batch['actions'].to(self.device)
        policy_probs = self.agent.discriminate(policy_multimodal, policy_actions)
        
        # Binary cross-entropy loss
        expert_loss = nn.BCELoss()(expert_probs, torch.ones_like(expert_probs))
        policy_loss = nn.BCELoss()(policy_probs, torch.zeros_like(policy_probs))
        
        return (expert_loss + policy_loss) / 2
    
    def compute_adversarial_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute adversarial robustness loss.
        
        Args:
            batch: Batch of data
        
        Returns:
            Adversarial loss
        """
        multimodal_data = {k: v.to(self.device) for k, v in batch['multimodal'].items()}
        
        # Enable gradients for adversarial perturbation
        for modality, data in multimodal_data.items():
            if data.dtype == torch.float32:
                data.requires_grad_(True)
        
        # Generate adversarial perturbations
        perturbed_data = self.agent.generate_adversarial_perturbation(multimodal_data)
        
        # Original predictions
        original_actions = self.agent.get_action(multimodal_data)
        
        # Perturbed predictions
        perturbed_actions = self.agent.get_action(perturbed_data)
        
        # Consistency loss
        adversarial_loss = nn.MSELoss()(original_actions, perturbed_actions)
        
        return adversarial_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.agent.train()
        
        total_reward_loss = 0.0
        total_policy_loss = 0.0
        total_discriminator_loss = 0.0
        total_adversarial_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.expert_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, expert_batch in enumerate(pbar):
            # Generate policy batch (rollout from current policy)
            policy_batch = self.generate_policy_batch(batch_size=expert_batch['actions'].size(0))
            
            # Update reward network
            self.reward_optimizer.zero_grad()
            reward_loss = self.compute_reward_loss(expert_batch, policy_batch)
            (self.reward_loss_weight * reward_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.agent.reward_network.parameters(), max_norm=1.0)
            self.reward_optimizer.step()
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss = self.compute_policy_loss(expert_batch)
            adversarial_loss = self.compute_adversarial_loss(expert_batch)
            total_policy_objective = (
                self.policy_loss_weight * policy_loss + 
                self.adversarial_loss_weight * adversarial_loss
            )
            total_policy_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.agent.multimodal_encoder.parameters()) + 
                list(self.agent.policy_network.parameters()), 
                max_norm=1.0
            )
            self.policy_optimizer.step()
            
            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            discriminator_loss = self.compute_discriminator_loss(expert_batch, policy_batch)
            (self.discriminator_loss_weight * discriminator_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.agent.discriminator.parameters(), max_norm=1.0)
            self.discriminator_optimizer.step()
            
            # Accumulate losses
            total_reward_loss += reward_loss.item()
            total_policy_loss += policy_loss.item()
            total_discriminator_loss += discriminator_loss.item()
            total_adversarial_loss += adversarial_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'R_Loss': f'{reward_loss.item():.4f}',
                'P_Loss': f'{policy_loss.item():.4f}',
                'D_Loss': f'{discriminator_loss.item():.4f}',
                'A_Loss': f'{adversarial_loss.item():.4f}'
            })
            
            # Log to wandb
            if self.config.get('use_wandb', True) and batch_idx % 10 == 0:
                wandb.log({
                    'train/reward_loss': reward_loss.item(),
                    'train/policy_loss': policy_loss.item(),
                    'train/discriminator_loss': discriminator_loss.item(),
                    'train/adversarial_loss': adversarial_loss.item(),
                    'epoch': self.epoch + batch_idx / len(self.expert_loader)
                })
        
        return {
            'reward_loss': total_reward_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'discriminator_loss': total_discriminator_loss / num_batches,
            'adversarial_loss': total_adversarial_loss / num_batches
        }
    
    def generate_policy_batch(self, batch_size: int) -> Dict:
        """Generate a batch of policy trajectories for training."""
        # This would typically involve rolling out the current policy
        # For now, we'll use a simplified version that samples from validation data
        policy_batch = next(iter(self.val_loader))
        
        # Optionally add noise to make it more like policy-generated data
        if 'actions' in policy_batch:
            noise_scale = 0.1
            noise = torch.randn_like(policy_batch['actions']) * noise_scale
            policy_batch['actions'] = policy_batch['actions'] + noise
        
        return policy_batch
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.agent.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                multimodal_data = {k: v.to(self.device) for k, v in batch['multimodal'].items()}
                expert_actions = batch['actions'].to(self.device)
                
                # Compute validation loss (behavioral cloning)
                predicted_actions = self.agent.get_action(multimodal_data)
                loss = nn.MSELoss()(predicted_actions, expert_actions)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Update learning rate schedulers
        self.reward_scheduler.step(avg_loss)
        self.policy_scheduler.step(avg_loss)
        self.discriminator_scheduler.step(avg_loss)
        
        return {'validation_loss': avg_loss}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.agent.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        save_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        self.logger.info(f'Checkpoint saved to {save_path}')
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / filename
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            
            self.logger.info(f'Checkpoint loaded from {checkpoint_path}')
        else:
            self.logger.warning(f'Checkpoint not found at {checkpoint_path}')
    
    def train(self):
        """Main training loop."""
        self.logger.info('Starting adversarial IRL training...')
        
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.logger.info(
                f'Epoch {epoch}: '
                f'Train Loss - R: {train_metrics["reward_loss"]:.4f}, '
                f'P: {train_metrics["policy_loss"]:.4f}, '
                f'D: {train_metrics["discriminator_loss"]:.4f}, '
                f'A: {train_metrics["adversarial_loss"]:.4f} | '
                f'Val Loss: {val_metrics["validation_loss"]:.4f}'
            )
            
            if self.config.get('use_wandb', True):
                wandb.log({
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch': epoch
                })
            
            # Save best model
            current_loss = val_metrics['validation_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        self.logger.info('Training completed!')


def main():
    parser = argparse.ArgumentParser(description='Train Adversarial IRL for Navigation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = AdversarialIRLTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
