#!/usr/bin/env python3
"""
Universal Launcher for Adversarial IRL Navigation System

This script provides multiple interface options:
1. Desktop GUI (Tkinter) - Native desktop application
2. Web Interface (Streamlit) - Modern web-based interface  
3. Interactive ML Interface (Gradio) - Machine learning focused interface
4. Command Line Interface - Terminal-based interface
5. Jupyter Notebook - Interactive notebook environment

Choose your preferred interface for the complete Adversarial IRL system.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import importlib.util

def check_dependencies():
    """Check if all required dependencies are available"""
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'streamlit': 'Streamlit',
        'gradio': 'Gradio',
        'plotly': 'Plotly',
        'tkinter': 'Tkinter'
    }
    
    missing = []
    available = []
    
    for module, name in dependencies.items():
        try:
            if module == 'tkinter':
                import tkinter
            else:
                importlib.import_module(module)
            available.append(f"✅ {name}")
        except ImportError:
            missing.append(f"❌ {name}")
    
    print("🔍 Dependency Check:")
    for dep in available:
        print(f"  {dep}")
    for dep in missing:
        print(f"  {dep}")
    
    return len(missing) == 0

def launch_desktop_gui():
    """Launch the Tkinter desktop GUI"""
    print("🖥️ Launching Desktop GUI (Tkinter)...")
    try:
        subprocess.run([sys.executable, "adversarial_irl_gui.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch desktop GUI: {e}")
    except FileNotFoundError:
        print("❌ GUI file not found. Please ensure adversarial_irl_gui.py exists.")

def launch_streamlit_web():
    """Launch the Streamlit web interface"""
    print("🌐 Launching Streamlit Web Interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "adversarial_irl_web.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit: {e}")
    except FileNotFoundError:
        print("❌ Streamlit not installed or web file not found.")

def launch_gradio_interface():
    """Launch the Gradio ML interface"""
    print("🤖 Launching Gradio ML Interface...")
    try:
        subprocess.run([sys.executable, "adversarial_irl_gradio.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Gradio: {e}")
    except FileNotFoundError:
        print("❌ Gradio file not found. Please ensure adversarial_irl_gradio.py exists.")

def launch_cli():
    """Launch the command line interface"""
    print("💻 Launching Command Line Interface...")
    try:
        subprocess.run([sys.executable, "fixed_train_complete.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch CLI: {e}")
    except FileNotFoundError:
        print("❌ CLI file not found. Please ensure fixed_train_complete.py exists.")

def launch_jupyter():
    """Launch Jupyter notebook environment"""
    print("📓 Launching Jupyter Notebook...")
    try:
        # Create a comprehensive notebook
        create_demo_notebook()
        subprocess.run(["jupyter", "notebook", "adversarial_irl_demo.ipynb"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Jupyter: {e}")
    except FileNotFoundError:
        print("❌ Jupyter not installed or notebook file not found.")

def create_demo_notebook():
    """Create a demonstration Jupyter notebook"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 Adversarial IRL Navigation System Demo\n",
                    "\n",
                    "This notebook provides an interactive demonstration of the complete Adversarial Inverse Reinforcement Learning system for autonomous navigation.\n",
                    "\n",
                    "## Features:\n",
                    "- Complete model training pipeline\n",
                    "- Real-time navigation simulation\n",
                    "- Performance evaluation and visualization\n",
                    "- Configuration management\n",
                    "- Results analysis and export"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# System setup and imports\n",
                    "import sys\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import torch\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Add project paths\n",
                    "sys.path.append('src')\n",
                    "sys.path.append('config')\n",
                    "\n",
                    "# Import project components\n",
                    "from config.fixed_config import get_config, validate_config\n",
                    "from utils.fixed_data_loader import FixedSyntheticDataset\n",
                    "from fixed_train_complete import FixedAdversarialIRLTrainer\n",
                    "from complete_navigation_test import NavigationController\n",
                    "\n",
                    "print(\"✅ System initialized successfully!\")\n",
                    "print(f\"PyTorch version: {torch.__version__}\")\n",
                    "print(f\"Device available: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load and display configuration\n",
                    "config = get_config()\n",
                    "print(\"📋 System Configuration:\")\n",
                    "for key, value in config.items():\n",
                    "    print(f\"  {key}: {value}\")\n",
                    "\n",
                    "# Validate configuration\n",
                    "try:\n",
                    "    validate_config(config)\n",
                    "    print(\"\\n✅ Configuration is valid!\")\n",
                    "except Exception as e:\n",
                    "    print(f\"\\n❌ Configuration error: {e}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize the adversarial IRL trainer\n",
                    "print(\"🏋️ Initializing Adversarial IRL Trainer...\")\n",
                    "trainer = FixedAdversarialIRLTrainer(config)\n",
                    "print(\"✅ Trainer initialized successfully!\")\n",
                    "\n",
                    "# Display model architecture information\n",
                    "total_params = sum(p.numel() for p in trainer.encoder.parameters())\n",
                    "total_params += sum(p.numel() for p in trainer.policy.parameters())\n",
                    "total_params += sum(p.numel() for p in trainer.reward_net.parameters())\n",
                    "total_params += sum(p.numel() for p in trainer.discriminator.parameters())\n",
                    "\n",
                    "print(f\"📊 Model Statistics:\")\n",
                    "print(f\"  Total parameters: {total_params:,}\")\n",
                    "print(f\"  Estimated size: ~{total_params * 4 / 1024 / 1024:.1f} MB\")\n",
                    "print(f\"  Training device: {config['device']}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Training demonstration (quick demo with few epochs)\n",
                    "print(\"🚀 Starting training demonstration...\")\n",
                    "\n",
                    "# Create synthetic dataset\n",
                    "dataset = FixedSyntheticDataset(config, num_samples=100)\n",
                    "dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)\n",
                    "\n",
                    "# Training metrics storage\n",
                    "training_metrics = {'epochs': [], 'losses': []}\n",
                    "\n",
                    "# Quick training demo (5 epochs for demonstration)\n",
                    "demo_epochs = 5\n",
                    "for epoch in range(demo_epochs):\n",
                    "    epoch_loss = trainer.train_epoch(dataloader)\n",
                    "    \n",
                    "    training_metrics['epochs'].append(epoch + 1)\n",
                    "    training_metrics['losses'].append(epoch_loss)\n",
                    "    \n",
                    "    print(f\"  Epoch {epoch + 1}/{demo_epochs} - Loss: {epoch_loss:.4f}\")\n",
                    "\n",
                    "print(\"✅ Training demonstration completed!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize training progress\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "plt.plot(training_metrics['epochs'], training_metrics['losses'], 'b-', linewidth=2, marker='o')\n",
                    "plt.title('Training Progress - Loss Reduction', fontsize=14, fontweight='bold')\n",
                    "plt.xlabel('Epoch')\n",
                    "plt.ylabel('Loss')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"📊 Final training loss: {training_metrics['losses'][-1]:.4f}\")\n",
                    "print(f\"📈 Loss reduction: {((training_metrics['losses'][0] - training_metrics['losses'][-1]) / training_metrics['losses'][0] * 100):.1f}%\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Navigation demonstration\n",
                    "print(\"🚗 Starting navigation demonstration...\")\n",
                    "\n",
                    "# Initialize navigation controller\n",
                    "controller = NavigationController(trainer, config)\n",
                    "\n",
                    "# Run navigation demo\n",
                    "navigation_results = controller.run_demo_episode(episode_length=20)\n",
                    "\n",
                    "print(\"✅ Navigation demonstration completed!\")\n",
                    "print(f\"📊 Episode summary:\")\n",
                    "print(f\"  Steps completed: {len(navigation_results)}\")\n",
                    "print(f\"  Average reward: {np.mean([step['reward'] for step in navigation_results]):.3f}\")\n",
                    "print(f\"  Total distance: {sum([step['throttle'] for step in navigation_results]):.2f} units\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize navigation results\n",
                    "fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "\n",
                    "steps = [step['step'] for step in navigation_results]\n",
                    "steering = [step['steering'] for step in navigation_results]\n",
                    "throttle = [step['throttle'] for step in navigation_results]\n",
                    "brake = [step['brake'] for step in navigation_results]\n",
                    "rewards = [step['reward'] for step in navigation_results]\n",
                    "\n",
                    "# Steering control\n",
                    "ax1.plot(steps, steering, 'b-', linewidth=2)\n",
                    "ax1.set_title('Steering Control', fontweight='bold')\n",
                    "ax1.set_ylabel('Steering Angle')\n",
                    "ax1.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Throttle control\n",
                    "ax2.plot(steps, throttle, 'g-', linewidth=2)\n",
                    "ax2.set_title('Throttle Control', fontweight='bold')\n",
                    "ax2.set_ylabel('Throttle')\n",
                    "ax2.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Brake control\n",
                    "ax3.plot(steps, brake, 'r-', linewidth=2)\n",
                    "ax3.set_title('Brake Control', fontweight='bold')\n",
                    "ax3.set_ylabel('Brake')\n",
                    "ax3.set_xlabel('Step')\n",
                    "ax3.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Reward signal\n",
                    "ax4.plot(steps, rewards, 'purple', linewidth=2)\n",
                    "ax4.set_title('Reward Signal', fontweight='bold')\n",
                    "ax4.set_ylabel('Reward')\n",
                    "ax4.set_xlabel('Step')\n",
                    "ax4.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Performance evaluation\n",
                    "print(\"📊 Running performance evaluation...\")\n",
                    "\n",
                    "# Run multiple navigation episodes for evaluation\n",
                    "eval_episodes = 10\n",
                    "eval_results = []\n",
                    "\n",
                    "for episode in range(eval_episodes):\n",
                    "    episode_results = controller.run_demo_episode(episode_length=15)\n",
                    "    episode_reward = np.mean([step['reward'] for step in episode_results])\n",
                    "    eval_results.append(episode_reward)\n",
                    "    print(f\"  Episode {episode + 1}: Average reward = {episode_reward:.3f}\")\n",
                    "\n",
                    "# Calculate evaluation statistics\n",
                    "mean_reward = np.mean(eval_results)\n",
                    "std_reward = np.std(eval_results)\n",
                    "success_rate = sum(1 for r in eval_results if r > 0.6) / len(eval_results)\n",
                    "\n",
                    "print(f\"\\n📈 Evaluation Results:\")\n",
                    "print(f\"  Mean reward: {mean_reward:.3f} ± {std_reward:.3f}\")\n",
                    "print(f\"  Success rate: {success_rate:.1%}\")\n",
                    "print(f\"  Best episode: {max(eval_results):.3f}\")\n",
                    "print(f\"  Worst episode: {min(eval_results):.3f}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize evaluation results\n",
                    "plt.figure(figsize=(12, 5))\n",
                    "\n",
                    "# Episode performance\n",
                    "plt.subplot(1, 2, 1)\n",
                    "plt.plot(range(1, len(eval_results) + 1), eval_results, 'bo-', linewidth=2, markersize=8)\n",
                    "plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.3f}')\n",
                    "plt.title('Episode Performance', fontweight='bold')\n",
                    "plt.xlabel('Episode')\n",
                    "plt.ylabel('Average Reward')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "# Reward distribution\n",
                    "plt.subplot(1, 2, 2)\n",
                    "plt.hist(eval_results, bins=5, alpha=0.7, color='skyblue', edgecolor='black')\n",
                    "plt.axvline(x=mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.3f}')\n",
                    "plt.title('Reward Distribution', fontweight='bold')\n",
                    "plt.xlabel('Average Reward')\n",
                    "plt.ylabel('Frequency')\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎉 Demo Complete!\n",
                    "\n",
                    "You have successfully:\n",
                    "1. ✅ Initialized the Adversarial IRL system\n",
                    "2. ✅ Trained the model with synthetic data\n",
                    "3. ✅ Demonstrated navigation capabilities\n",
                    "4. ✅ Evaluated performance across multiple episodes\n",
                    "5. ✅ Visualized training and navigation results\n",
                    "\n",
                    "### Next Steps:\n",
                    "- Modify the configuration in `config/fixed_config.py` for different experiments\n",
                    "- Run longer training with more epochs\n",
                    "- Experiment with different navigation scenarios\n",
                    "- Try the GUI interfaces (`adversarial_irl_gui.py`, `adversarial_irl_web.py`, `adversarial_irl_gradio.py`)\n",
                    "\n",
                    "### For Production Use:\n",
                    "- Replace synthetic data with real driving data\n",
                    "- Integrate with actual vehicle control systems\n",
                    "- Implement safety constraints and monitoring\n",
                    "- Add comprehensive logging and telemetry"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    import json
    with open("adversarial_irl_demo.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print("📓 Demo notebook created: adversarial_irl_demo.ipynb")

def show_menu():
    """Display the main interface selection menu"""
    print("\n" + "="*60)
    print("🚀 ADVERSARIAL IRL NAVIGATION SYSTEM")
    print("   Complete Training, Testing & Deployment Platform")
    print("="*60)
    
    print("\n📋 Available Interfaces:")
    print("1. 🖥️  Desktop GUI (Tkinter)     - Native desktop application")
    print("2. 🌐 Web Interface (Streamlit)  - Modern web dashboard") 
    print("3. 🤖 ML Interface (Gradio)      - Interactive ML interface")
    print("4. 💻 Command Line Interface     - Terminal-based training")
    print("5. 📓 Jupyter Notebook          - Interactive notebook environment")
    print("6. 🔍 Dependency Check          - Verify system requirements")
    print("7. ❌ Exit")
    
    print("\n" + "-"*60)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Universal Launcher for Adversarial IRL Navigation System"
    )
    parser.add_argument(
        "--interface", "-i",
        choices=["gui", "web", "gradio", "cli", "jupyter"],
        help="Launch specific interface directly"
    )
    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check dependencies only"
    )
    
    args = parser.parse_args()
    
    # Direct interface launch
    if args.interface:
        if args.interface == "gui":
            launch_desktop_gui()
        elif args.interface == "web":
            launch_streamlit_web()
        elif args.interface == "gradio":
            launch_gradio_interface()
        elif args.interface == "cli":
            launch_cli()
        elif args.interface == "jupyter":
            launch_jupyter()
        return
    
    # Dependency check only
    if args.check:
        check_dependencies()
        return
    
    # Default behavior: launch desktop GUI directly
    launch_desktop_gui()

if __name__ == "__main__":
    main()
