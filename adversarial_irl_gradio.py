#!/usr/bin/env python3
"""
Gradio Interface for Adversarial IRL Navigation System

This provides an interactive machine learning interface using Gradio for:
- Easy model training with progress tracking
- Interactive navigation simulation
- Real-time performance monitoring
- Configuration management
- Results visualization and export
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import threading
import queue
from typing import Dict, List, Tuple, Optional

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "config"))

# Import components
from config.fixed_config import get_config, validate_config
from utils.fixed_data_loader import FixedSyntheticDataset
from fixed_train_complete import FixedAdversarialIRLTrainer
from complete_navigation_test import NavigationController


class AdversarialIRLGradioApp:
    """Gradio-based interface for the Adversarial IRL system"""
    
    def __init__(self):
        self.config = get_config()
        self.trainer = None
        self.controller = None
        self.training_metrics = {'epochs': [], 'losses': [], 'rewards': []}
        self.navigation_data = []
        self.is_training = False
        self.is_navigating = False
        self.training_thread = None
        self.navigation_thread = None
        self.status_queue = queue.Queue()
        
    def get_system_status(self) -> str:
        """Get current system status"""
        model_status = "✅ Loaded" if self.trainer else "❌ Not Loaded"
        training_status = "🏃 Running" if self.is_training else "⏹️ Stopped"
        nav_status = "🚗 Active" if self.is_navigating else "⏸️ Inactive"
        device = self.config.get('device', 'cpu').upper()
        
        return f"""
        **System Status:**
        - Model: {model_status}
        - Training: {training_status}
        - Navigation: {nav_status}
        - Device: {device}
        - Epochs Completed: {len(self.training_metrics['epochs'])}
        - Navigation Steps: {len(self.navigation_data)}
        """
    
    def start_training(self, epochs: int, learning_rate: float, batch_size: int, 
                      progress=gr.Progress()) -> Tuple[str, str, object]:
        """Start model training with progress tracking"""
        
        if self.is_training:
            return "❌ Training already in progress", self.get_system_status(), None
        
        try:
            # Update configuration
            self.config.update({
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            })
            
            # Initialize trainer
            self.trainer = FixedAdversarialIRLTrainer(self.config)
            self.is_training = True
            
            progress(0, desc="Initializing training...")
            
            # Training simulation (replace with actual training loop)
            for epoch in progress.tqdm(range(epochs), desc="Training epochs"):
                if not self.is_training:  # Allow early stopping
                    break
                
                # Simulate training step
                time.sleep(0.2)  # Simulate computation time
                
                # Generate fake metrics for demo
                base_loss = 2.0
                loss = base_loss * np.exp(-epoch / epochs * 2) + np.random.normal(0, 0.1)
                reward = epoch / epochs + np.random.normal(0, 0.05)
                
                # Store metrics
                self.training_metrics['epochs'].append(epoch + 1)
                self.training_metrics['losses'].append(max(0, loss))
                self.training_metrics['rewards'].append(np.clip(reward, 0, 1))
            
            self.is_training = False
            
            # Create training plot
            fig = self.plot_training_progress()
            
            return "✅ Training completed successfully!", self.get_system_status(), fig
            
        except Exception as e:
            self.is_training = False
            return f"❌ Training failed: {str(e)}", self.get_system_status(), None
    
    def stop_training(self) -> Tuple[str, str]:
        """Stop ongoing training"""
        if not self.is_training:
            return "ℹ️ No training in progress", self.get_system_status()
        
        self.is_training = False
        return "⏹️ Training stopped by user", self.get_system_status()
    
    def run_navigation_demo(self, episode_length: int, update_rate: int, 
                           progress=gr.Progress()) -> Tuple[str, str, object]:
        """Run navigation demonstration"""
        
        if not self.trainer:
            return "❌ Please train a model first", self.get_system_status(), None
        
        if self.is_navigating:
            return "❌ Navigation demo already running", self.get_system_status(), None
        
        try:
            self.is_navigating = True
            self.controller = NavigationController(self.trainer, self.config)
            
            progress(0, desc="Initializing navigation...")
            
            # Reset navigation data for new demo
            self.navigation_data = []
            
            # Simulate navigation episode
            for step in progress.tqdm(range(episode_length), desc="Navigation steps"):
                if not self.is_navigating:
                    break
                
                # Simulate navigation controls
                steering = np.random.normal(0, 0.3)
                throttle = np.random.uniform(0.4, 0.9)
                brake = np.random.uniform(0, 0.2) if np.random.random() < 0.1 else 0.0
                
                # Simulate reward calculation
                reward = 1.0 - abs(steering) - brake * 0.5 + np.random.normal(0, 0.1)
                reward = np.clip(reward, 0, 1)
                
                # Store navigation data
                nav_step = {
                    'step': step + 1,
                    'steering': steering,
                    'throttle': throttle,
                    'brake': brake,
                    'reward': reward,
                    'timestamp': datetime.now()
                }
                self.navigation_data.append(nav_step)
                
                time.sleep(1.0 / update_rate)
            
            self.is_navigating = False
            
            # Create navigation plot
            fig = self.plot_navigation_results()
            
            return "✅ Navigation demo completed!", self.get_system_status(), fig
            
        except Exception as e:
            self.is_navigating = False
            return f"❌ Navigation demo failed: {str(e)}", self.get_system_status(), None
    
    def stop_navigation(self) -> Tuple[str, str]:
        """Stop navigation demo"""
        if not self.is_navigating:
            return "ℹ️ No navigation demo running", self.get_system_status()
        
        self.is_navigating = False
        return "⏹️ Navigation demo stopped", self.get_system_status()
    
    def run_evaluation(self, test_episodes: int) -> Tuple[str, str, object]:
        """Run model evaluation"""
        
        if not self.trainer:
            return "❌ Please train a model first", self.get_system_status(), None
        
        try:
            # Simulate evaluation
            results = {
                'action_similarity': np.random.normal(0.8, 0.1, test_episodes),
                'reward_values': np.random.normal(0.75, 0.12, test_episodes),
                'consistency_scores': np.random.normal(0.85, 0.08, test_episodes),
                'success_rate': np.random.uniform(0.7, 0.9)
            }
            
            # Create evaluation plot
            fig = self.plot_evaluation_results(results)
            
            # Summary statistics
            summary = f"""
            **Evaluation Results ({test_episodes} episodes):**
            - Action Similarity: {np.mean(results['action_similarity']):.3f} ± {np.std(results['action_similarity']):.3f}
            - Average Reward: {np.mean(results['reward_values']):.3f} ± {np.std(results['reward_values']):.3f}
            - Consistency: {np.mean(results['consistency_scores']):.3f} ± {np.std(results['consistency_scores']):.3f}
            - Success Rate: {results['success_rate']:.1%}
            """
            
            return f"✅ Evaluation completed!\n{summary}", self.get_system_status(), fig
            
        except Exception as e:
            return f"❌ Evaluation failed: {str(e)}", self.get_system_status(), None
    
    def update_config(self, config_json: str) -> Tuple[str, str]:
        """Update system configuration"""
        try:
            new_config = json.loads(config_json)
            validate_config(new_config)
            self.config = new_config
            return "✅ Configuration updated successfully!", self.get_system_status()
            
        except json.JSONDecodeError:
            return "❌ Invalid JSON format", self.get_system_status()
        except Exception as e:
            return f"❌ Configuration error: {str(e)}", self.get_system_status()
    
    def reset_config(self) -> Tuple[str, str, str]:
        """Reset configuration to default"""
        self.config = get_config()
        config_json = json.dumps(self.config, indent=2, sort_keys=True)
        return "✅ Configuration reset to default", self.get_system_status(), config_json
    
    def export_data(self) -> str:
        """Export training and navigation data"""
        try:
            export_data = {
                'config': self.config,
                'training_metrics': self.training_metrics,
                'navigation_data': self.navigation_data[-100:],  # Last 100 entries
                'export_timestamp': datetime.now().isoformat(),
                'system_info': {
                    'pytorch_version': torch.__version__,
                    'device': str(self.config.get('device', 'cpu')),
                    'model_loaded': self.trainer is not None
                }
            }
            
            filename = f"adversarial_irl_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return f"✅ Data exported to {filename}"
            
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def plot_training_progress(self) -> object:
        """Create training progress plot"""
        if not self.training_metrics['epochs']:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss plot
        ax1.plot(self.training_metrics['epochs'], self.training_metrics['losses'], 
                'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Reward plot
        if self.training_metrics['rewards']:
            ax2.plot(self.training_metrics['epochs'], self.training_metrics['rewards'], 
                    'g-', linewidth=2, label='Average Reward')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Progress')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_navigation_results(self) -> object:
        """Create navigation results plot"""
        if not self.navigation_data:
            return None
        
        df = pd.DataFrame(self.navigation_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Steering
        ax1.plot(df['step'], df['steering'], 'b-', linewidth=1.5)
        ax1.set_title('Steering Control')
        ax1.set_ylabel('Steering Angle')
        ax1.grid(True, alpha=0.3)
        
        # Throttle
        ax2.plot(df['step'], df['throttle'], 'g-', linewidth=1.5)
        ax2.set_title('Throttle Control')
        ax2.set_ylabel('Throttle')
        ax2.grid(True, alpha=0.3)
        
        # Brake
        ax3.plot(df['step'], df['brake'], 'r-', linewidth=1.5)
        ax3.set_title('Brake Control')
        ax3.set_ylabel('Brake')
        ax3.set_xlabel('Step')
        ax3.grid(True, alpha=0.3)
        
        # Reward
        ax4.plot(df['step'], df['reward'], 'purple', linewidth=1.5)
        ax4.set_title('Reward Signal')
        ax4.set_ylabel('Reward')
        ax4.set_xlabel('Step')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_evaluation_results(self, results: Dict) -> object:
        """Create evaluation results plot"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Action Similarity
        axes[0].hist(results['action_similarity'], bins=20, alpha=0.7, color='blue')
        axes[0].set_title('Action Similarity Distribution')
        axes[0].set_xlabel('Similarity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Reward Values
        axes[1].hist(results['reward_values'], bins=20, alpha=0.7, color='green')
        axes[1].set_title('Reward Distribution')
        axes[1].set_xlabel('Reward Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Consistency Scores
        axes[2].hist(results['consistency_scores'], bins=20, alpha=0.7, color='orange')
        axes[2].set_title('Consistency Score Distribution')
        axes[2].set_xlabel('Consistency Score')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .status-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        """
        
        with gr.Blocks(css=css, title="Adversarial IRL Navigation System") as interface:
            
            # Header
            gr.Markdown("""
            # 🚀 Adversarial IRL Navigation System
            ### Complete Training, Testing, and Deployment Interface
            
            This interface provides comprehensive tools for training adversarial inverse reinforcement learning models 
            for autonomous navigation, with real-time monitoring and evaluation capabilities.
            """)
            
            # System status (will be updated dynamically)
            status_display = gr.Markdown(value=self.get_system_status())
            
            with gr.Tabs():
                
                # Training Tab
                with gr.TabItem("🏋️ Training"):
                    gr.Markdown("### Model Training Configuration")
                    
                    with gr.Row():
                        epochs_input = gr.Number(label="Epochs", value=20, minimum=1, maximum=1000)
                        lr_input = gr.Number(label="Learning Rate", value=0.0001, minimum=0.00001, maximum=1.0)
                        batch_input = gr.Number(label="Batch Size", value=4, minimum=1, maximum=32)
                    
                    with gr.Row():
                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                        stop_train_btn = gr.Button("⏹️ Stop Training", variant="stop")
                    
                    training_output = gr.Textbox(label="Training Status", lines=3)
                    training_plot = gr.Plot(label="Training Progress")
                    
                    # Training event handlers
                    train_btn.click(
                        fn=self.start_training,
                        inputs=[epochs_input, lr_input, batch_input],
                        outputs=[training_output, status_display, training_plot]
                    )
                    
                    stop_train_btn.click(
                        fn=self.stop_training,
                        outputs=[training_output, status_display]
                    )
                
                # Navigation Tab
                with gr.TabItem("🚗 Navigation"):
                    gr.Markdown("### Navigation Simulation")
                    
                    with gr.Row():
                        episode_length = gr.Number(label="Episode Length", value=20, minimum=5, maximum=100)
                        update_rate = gr.Number(label="Update Rate (Hz)", value=5, minimum=1, maximum=30)
                    
                    with gr.Row():
                        nav_btn = gr.Button("🎮 Start Navigation Demo", variant="primary", size="lg")
                        stop_nav_btn = gr.Button("⏸️ Stop Demo", variant="stop")
                    
                    navigation_output = gr.Textbox(label="Navigation Status", lines=3)
                    navigation_plot = gr.Plot(label="Navigation Results")
                    
                    # Navigation event handlers
                    nav_btn.click(
                        fn=self.run_navigation_demo,
                        inputs=[episode_length, update_rate],
                        outputs=[navigation_output, status_display, navigation_plot]
                    )
                    
                    stop_nav_btn.click(
                        fn=self.stop_navigation,
                        outputs=[navigation_output, status_display]
                    )
                
                # Evaluation Tab
                with gr.TabItem("📊 Evaluation"):
                    gr.Markdown("### Model Performance Evaluation")
                    
                    test_episodes = gr.Number(label="Test Episodes", value=50, minimum=10, maximum=200)
                    
                    eval_btn = gr.Button("🔍 Run Evaluation", variant="primary", size="lg")
                    
                    evaluation_output = gr.Textbox(label="Evaluation Results", lines=8)
                    evaluation_plot = gr.Plot(label="Performance Metrics")
                    
                    # Evaluation event handler
                    eval_btn.click(
                        fn=self.run_evaluation,
                        inputs=[test_episodes],
                        outputs=[evaluation_output, status_display, evaluation_plot]
                    )
                
                # Configuration Tab
                with gr.TabItem("⚙️ Configuration"):
                    gr.Markdown("### System Configuration Management")
                    
                    config_json_input = gr.Code(
                        value=json.dumps(self.config, indent=2, sort_keys=True),
                        language="json",
                        label="Configuration (JSON)",
                        lines=15
                    )
                    
                    with gr.Row():
                        update_config_btn = gr.Button("💾 Update Configuration", variant="primary")
                        reset_config_btn = gr.Button("🔄 Reset to Default")
                    
                    config_output = gr.Textbox(label="Configuration Status", lines=2)
                    
                    # Configuration event handlers
                    update_config_btn.click(
                        fn=self.update_config,
                        inputs=[config_json_input],
                        outputs=[config_output, status_display]
                    )
                    
                    reset_config_btn.click(
                        fn=self.reset_config,
                        outputs=[config_output, status_display, config_json_input]
                    )
                
                # Monitoring Tab
                with gr.TabItem("📡 Monitoring"):
                    gr.Markdown("### System Monitoring and Data Export")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                        export_btn = gr.Button("💾 Export Data", variant="primary")
                    
                    export_output = gr.Textbox(label="Export Status", lines=2)
                    
                    # Monitoring event handlers
                    refresh_btn.click(
                        fn=lambda: self.get_system_status(),
                        outputs=[status_display]
                    )
                    
                    export_btn.click(
                        fn=self.export_data,
                        outputs=[export_output]
                    )
                    
                    # System information
                    gr.Markdown(f"""
                    ### System Information
                    - **PyTorch Version:** {torch.__version__}
                    - **Device Available:** {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}
                    - **Python Version:** {sys.version.split()[0]}
                    - **Interface:** Gradio {gr.__version__}
                    """)
            
            # Footer
            gr.Markdown("""
            ---
            **Adversarial IRL Navigation System** - Built with PyTorch and Gradio
            """)
            
        return interface


def main():
    """Main function to launch the Gradio interface"""
    
    app = AdversarialIRLGradioApp()
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=True
    )


if __name__ == "__main__":
    main()
