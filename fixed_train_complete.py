#!/usr/bin/env python3
"""
Fixed Training Script for Adversarial IRL Navigation System

This script addresses all tensor dimension issues and provides a complete
training pipeline for the adversarial inverse reinforcement learning system.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time

# Add project paths (repository root)
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "config"))

# Import fixed components
from config.fixed_config import get_config, validate_config
from utils.fixed_data_loader import FixedSyntheticDataset, collate_fn
from models.adversarial_irl import MultimodalEncoder, AdversarialIRLAgent


class FixedAdversarialIRLTrainer:
    """Fixed trainer for Adversarial IRL with proper tensor handling."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        print(f"🚀 Initializing trainer on device: {self.device}")
        
        # Initialize components
        self._setup_model()
        self._setup_optimizers()
        self._setup_data()
        
        # Training metrics
        self.train_metrics = {
            'discriminator_losses': [],
            'generator_losses': [],
            'reward_losses': [],
            'expert_rewards': [],
            'generated_rewards': []
        }
        
    def _setup_model(self):
        """Initialize the adversarial IRL model components."""
        print("🧠 Setting up model components...")
        
        # Multimodal encoder
        self.encoder = MultimodalEncoder(self.config).to(self.device)
        
        # Policy network (generator)
        self.policy = nn.Sequential(
            nn.Linear(self.config['fusion_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['action_dim']),
            nn.Tanh()  # Actions bounded to [-1, 1]
        ).to(self.device)
        
        # Reward network
        self.reward_net = nn.Sequential(
            nn.Linear(self.config['fusion_dim'] + self.config['action_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], 1)
        ).to(self.device)
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.config['fusion_dim'] + self.config['action_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Print model info
        total_params = (
            sum(p.numel() for p in self.encoder.parameters())
            + sum(p.numel() for p in self.policy.parameters())
            + sum(p.numel() for p in self.reward_net.parameters())
            + sum(p.numel() for p in self.discriminator.parameters())
        )
        print(f"   📊 Total parameters: {total_params:,}")
        
    def _setup_optimizers(self):
        """Setup optimizers for all components."""
        print("⚙️ Setting up optimizers...")
        
        self.encoder_opt = optim.Adam(self.encoder.parameters(), 
                                    lr=self.config['learning_rate'], 
                                    betas=(self.config['beta1'], self.config['beta2']))
        
        self.policy_opt = optim.Adam(self.policy.parameters(), 
                                   lr=self.config['lr_policy'], 
                                   betas=(self.config['beta1'], self.config['beta2']))
        
        self.reward_opt = optim.Adam(self.reward_net.parameters(), 
                                   lr=self.config['lr_reward'], 
                                   betas=(self.config['beta1'], self.config['beta2']))
        
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), 
                                          lr=self.config['lr_discriminator'], 
                                          betas=(self.config['beta1'], self.config['beta2']))
        
    def _setup_data(self):
        """Setup training data."""
        print("📊 Setting up training data...")
        
        # Create synthetic dataset
        self.dataset = FixedSyntheticDataset(self.config, num_samples=2000)
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0  # Set to 0 for compatibility
        )
        
        print(f"   📦 Dataset size: {len(self.dataset)}")
        print(f"   🔄 Batches per epoch: {len(self.dataloader)}")
        
    def train_step(self, batch):
        """Single training step."""
        multimodal_data = batch['multimodal']
        expert_actions = batch['actions'].to(self.device)
        
        # Move data to device
        for key in multimodal_data:
            multimodal_data[key] = multimodal_data[key].to(self.device)
        
        # Encode states
        with torch.no_grad():
            states = self.encoder(multimodal_data)
        
        batch_size = states.size(0)
        
        # === Train Discriminator ===
        self.discriminator_opt.zero_grad()
        
        # Real data (expert demonstrations)
        real_state_action = torch.cat([states, expert_actions], dim=1)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_state_action)
        real_loss = nn.BCELoss()(real_output, real_labels)
        
        # Fake data (policy generated)
        with torch.no_grad():
            fake_actions = self.policy(states)
        fake_state_action = torch.cat([states, fake_actions], dim=1)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.discriminator(fake_state_action)
        fake_loss = nn.BCELoss()(fake_output, fake_labels)
        
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        self.discriminator_opt.step()
        
        # === Train Generator (Policy) ===
        self.encoder_opt.zero_grad()
        self.policy_opt.zero_grad()
        
        # Re-encode with gradients
        states = self.encoder(multimodal_data)
        generated_actions = self.policy(states)
        
        # Generator tries to fool discriminator
        gen_state_action = torch.cat([states, generated_actions], dim=1)
        gen_output = self.discriminator(gen_state_action)
        gen_labels = torch.ones(batch_size, 1, device=self.device)  # Want discriminator to think it's real
        generator_loss = nn.BCELoss()(gen_output, gen_labels)
        
        generator_loss.backward()
        self.policy_opt.step()
        self.encoder_opt.step()
        
        # === Train Reward Network ===
        self.reward_opt.zero_grad()
        
        with torch.no_grad():
            states = self.encoder(multimodal_data)
        
        # Expert rewards should be higher than generated rewards
        expert_state_action = torch.cat([states, expert_actions], dim=1)
        expert_rewards = self.reward_net(expert_state_action)
        
        gen_state_action = torch.cat([states, generated_actions.detach()], dim=1)
        gen_rewards = self.reward_net(gen_state_action)
        
        # IRL objective: expert demonstrations should have higher reward
        reward_loss = (
            nn.MSELoss()(expert_rewards, torch.ones_like(expert_rewards))
            + nn.MSELoss()(gen_rewards, torch.zeros_like(gen_rewards))
        )
        
        reward_loss.backward()
        self.reward_opt.step()
        
        return {
            'discriminator_loss': discriminator_loss.item(),
            'generator_loss': generator_loss.item(),
            'reward_loss': reward_loss.item(),
            'expert_reward': expert_rewards.mean().item(),
            'generated_reward': gen_rewards.mean().item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.encoder.train()
        self.policy.train()
        self.reward_net.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'discriminator_loss': 0.0,
            'generator_loss': 0.0,
            'reward_loss': 0.0,
            'expert_reward': 0.0,
            'generated_reward': 0.0
        }
        
        pbar = tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        for batch_idx, batch in enumerate(pbar):
            step_metrics = self.train_step(batch)
            
            # Update epoch metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] += value
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'D_loss': f"{step_metrics['discriminator_loss']:.4f}",
                    'G_loss': f"{step_metrics['generator_loss']:.4f}",
                    'R_loss': f"{step_metrics['reward_loss']:.4f}"
                })
        
        # Average metrics
        num_batches = len(self.dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self):
        """Full training loop."""
        print(f"\\n🚀 Starting training for {self.config['epochs']} epochs...")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_metrics = self.train_epoch(epoch)
            
            # Store metrics
            self.train_metrics['discriminator_losses'].append(epoch_metrics['discriminator_loss'])
            self.train_metrics['generator_losses'].append(epoch_metrics['generator_loss'])
            self.train_metrics['reward_losses'].append(epoch_metrics['reward_loss'])
            self.train_metrics['expert_rewards'].append(epoch_metrics['expert_reward'])
            self.train_metrics['generated_rewards'].append(epoch_metrics['generated_reward'])
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"\\nEpoch {epoch+1}/{self.config['epochs']}:")
                print(f"  Discriminator Loss: {epoch_metrics['discriminator_loss']:.6f}")
                print(f"  Generator Loss:     {epoch_metrics['generator_loss']:.6f}")
                print(f"  Reward Loss:        {epoch_metrics['reward_loss']:.6f}")
                print(f"  Expert Reward:      {epoch_metrics['expert_reward']:.6f}")
                print(f"  Generated Reward:   {epoch_metrics['generated_reward']:.6f}")
        
        training_time = time.time() - start_time
        print(f"\\n✅ Training completed in {training_time:.2f} seconds")
        
        return self.train_metrics
    
    def evaluate(self, num_samples=100):
        """Evaluate the trained model."""
        print("\\n🔍 Evaluating trained model...")
        
        self.encoder.eval()
        self.policy.eval()
        self.reward_net.eval()
        
        eval_metrics = {
            'action_similarity': [],
            'reward_values': [],
            'consistency_scores': []
        }
        
        with torch.no_grad():
            for i in range(min(num_samples, len(self.dataset))):
                sample = self.dataset[i]
                
                # Prepare data
                multimodal_data = {}
                for key, tensor in sample['multimodal'].items():
                    multimodal_data[key] = tensor.unsqueeze(0).to(self.device)
                
                expert_action = sample['actions'].unsqueeze(0).to(self.device)
                
                # Forward pass
                state = self.encoder(multimodal_data)
                predicted_action = self.policy(state)
                reward = self.reward_net(torch.cat([state, predicted_action], dim=1))
                
                # Calculate metrics
                similarity = nn.functional.cosine_similarity(predicted_action, expert_action, dim=1).item()
                action_diff = torch.abs(predicted_action - expert_action).mean().item()
                consistency = 1.0 - min(action_diff, 1.0)  # Consistency score
                
                eval_metrics['action_similarity'].append(similarity)
                eval_metrics['reward_values'].append(reward.item())
                eval_metrics['consistency_scores'].append(consistency)
        
        # Calculate summary statistics
        avg_similarity = np.mean(eval_metrics['action_similarity'])
        avg_reward = np.mean(eval_metrics['reward_values'])
        avg_consistency = np.mean(eval_metrics['consistency_scores'])
        
        print(f"\\n📊 Evaluation Results:")
        print(f"  Action Similarity:  {avg_similarity:.4f} ± {np.std(eval_metrics['action_similarity']):.4f}")
        print(f"  Average Reward:     {avg_reward:.4f} ± {np.std(eval_metrics['reward_values']):.4f}")
        print(f"  Consistency Score:  {avg_consistency:.4f} ± {np.std(eval_metrics['consistency_scores']):.4f}")
        
        return eval_metrics
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'reward_net': self.reward_net.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'config': self.config,
            'train_metrics': self.train_metrics
        }, filepath)
        print(f"💾 Model saved to {filepath}")


def plot_training_metrics(metrics):
    """Plot training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🚀 Adversarial IRL Training Progress', fontsize=16)
    
    epochs = range(1, len(metrics['discriminator_losses']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, metrics['discriminator_losses'], 'r-', label='Discriminator', linewidth=2)
    axes[0, 0].plot(epochs, metrics['generator_losses'], 'b-', label='Generator', linewidth=2)
    axes[0, 0].plot(epochs, metrics['reward_losses'], 'g-', label='Reward', linewidth=2)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward evolution
    axes[0, 1].plot(epochs, metrics['expert_rewards'], 'orange', label='Expert Rewards', linewidth=2)
    axes[0, 1].plot(epochs, metrics['generated_rewards'], 'purple', label='Generated Rewards', linewidth=2)
    axes[0, 1].set_title('Reward Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss difference (discriminator vs generator)
    loss_diff = np.array(metrics['discriminator_losses']) - np.array(metrics['generator_losses'])
    axes[1, 0].plot(epochs, loss_diff, 'black', linewidth=2)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Loss Balance (D_loss - G_loss)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward gap (expert vs generated)
    reward_gap = np.array(metrics['expert_rewards']) - np.array(metrics['generated_rewards'])
    axes[1, 1].plot(epochs, reward_gap, 'darkgreen', linewidth=2)
    axes[1, 1].set_title('Reward Gap (Expert - Generated)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reward Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main training function."""
    print("🚀 Adversarial IRL Navigation System - Fixed Training")
    print("=" * 80)
    
    # Load configuration
    config = get_config()
    validate_config(config)
    
    print(f"\\n⚙️ Configuration loaded:")
    print(f"   Device: {config['device']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Epochs: {config['epochs']}")
    
    # Create trainer
    trainer = FixedAdversarialIRLTrainer(config)
    
    # Train the model
    metrics = trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    # Plot results
    plot_training_metrics(metrics)
    
    # Save the trained model
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    trainer.save_model(output_dir / "adversarial_irl_model.pth")
    
    print("\\n🎉 Training completed successfully!")
    print("   ✅ Model trained on synthetic expert demonstrations")
    print("   ✅ Performance metrics computed")
    print("   ✅ Training visualizations generated")
    print("   ✅ Model saved for future use")
    
    return trainer, metrics, eval_results


if __name__ == "__main__":
    trainer, metrics, eval_results = main()
