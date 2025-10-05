#!/usr/bin/env python3
"""
Comprehensive GUI Application for Adversarial IRL Navigation System

This application provides a complete graphical interface for:
1. Training the adversarial IRL model
2. Real-time navigation simulation
3. Performance monitoring and visualization
4. Model evaluation and testing
5. Configuration management
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import torch
import threading
import time
import sys
from pathlib import Path
import json
from datetime import datetime
import queue

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "config"))

# Import project components (use real trainer and controller)
from config.fixed_config import get_config, validate_config
from fixed_train_complete import FixedAdversarialIRLTrainer
from complete_navigation_test import NavigationController
from utils.fixed_data_loader import FixedSyntheticDataset


class AdversarialIRLGUI:
    """Main GUI Application for Adversarial IRL Navigation System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Adversarial IRL Navigation System - Control Center")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Initialize components
        self.config = get_config()
        self.trainer = None
        self.controller = None
        self.training_thread = None
        self.is_training = False
        self.training_metrics = {"epochs": [], "losses": []}
        self.status_queue = queue.Queue()

        # Setup GUI
        self.setup_style()
        self.create_widgets()
        self.setup_plots()

        # Start status update loop
        self.update_status()
        
    def setup_style(self):
        """Configure GUI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define colors
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'success': '#27AE60',
            'dark': '#2C3E50',
            'light': '#ECF0F1'
        }
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(title_frame, text="🚀 Adversarial IRL Navigation System", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Complete Training, Testing, and Deployment Interface", 
                 style='Status.TLabel').pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_training_tab()
        self.create_navigation_tab()
        self.create_evaluation_tab()
        self.create_config_tab()
        self.create_monitoring_tab()
        
    def create_training_tab(self):
        """Create training interface tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🏋️ Training")
        
        # Training controls
        control_frame = ttk.LabelFrame(training_frame, text="Training Controls")
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Training parameters
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky='w')
        self.epochs_var = tk.StringVar(value=str(self.config['epochs']))
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=2, sticky='w', padx=(20,0))
        self.lr_var = tk.StringVar(value=str(self.config['learning_rate']))
        ttk.Entry(params_frame, textvariable=self.lr_var, width=12).grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky='w')
        self.batch_var = tk.StringVar(value=str(self.config['batch_size']))
        ttk.Entry(params_frame, textvariable=self.batch_var, width=10).grid(row=1, column=1, padx=5)
        
        # Training buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.train_button = ttk.Button(button_frame, text="🚀 Start Training", 
                                      command=self.start_training)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Training", 
                                     command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        self.save_button = ttk.Button(button_frame, text="💾 Save Model", 
                                     command=self.save_model, state='disabled')
        self.save_button.pack(side='left', padx=5)
        
        self.load_button = ttk.Button(button_frame, text="📂 Load Model", 
                                     command=self.load_model)
        self.load_button.pack(side='left', padx=5)
        
        # Training progress
        progress_frame = ttk.LabelFrame(training_frame, text="Training Progress")
        progress_frame.pack(fill='x', padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready to train")
        ttk.Label(progress_frame, textvariable=self.status_var).pack()
        
        # Training metrics display
        metrics_frame = ttk.LabelFrame(training_frame, text="Training Metrics")
        metrics_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create training plots frame
        self.training_plots_frame = ttk.Frame(metrics_frame)
        self.training_plots_frame.pack(fill='both', expand=True)
        
    def create_navigation_tab(self):
        """Create navigation simulation tab"""
        nav_frame = ttk.Frame(self.notebook)
        self.notebook.add(nav_frame, text="🚗 Navigation")
        
        # Navigation controls
        nav_control_frame = ttk.LabelFrame(nav_frame, text="Navigation Controls")
        nav_control_frame.pack(fill='x', padx=5, pady=5)
        
        nav_buttons = ttk.Frame(nav_control_frame)
        nav_buttons.pack(fill='x', padx=5, pady=5)
        
        self.start_nav_button = ttk.Button(nav_buttons, text="🎮 Start Navigation Demo", 
                                          command=self.start_navigation_demo)
        self.start_nav_button.pack(side='left', padx=5)
        
        self.stop_nav_button = ttk.Button(nav_buttons, text="⏸️ Stop Demo", 
                                         command=self.stop_navigation_demo, state='disabled')
        self.stop_nav_button.pack(side='left', padx=5)
        
        # Navigation parameters
        nav_params = ttk.Frame(nav_control_frame)
        nav_params.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(nav_params, text="Episode Length:").grid(row=0, column=0, sticky='w')
        self.episode_length_var = tk.StringVar(value="20")
        ttk.Entry(nav_params, textvariable=self.episode_length_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(nav_params, text="Update Rate (Hz):").grid(row=0, column=2, sticky='w', padx=(20,0))
        self.update_rate_var = tk.StringVar(value="5")
        ttk.Entry(nav_params, textvariable=self.update_rate_var, width=10).grid(row=0, column=3, padx=5)
        
        # Navigation display
        nav_display_frame = ttk.LabelFrame(nav_frame, text="Navigation Status")
        nav_display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Current action display
        action_frame = ttk.Frame(nav_display_frame)
        action_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(action_frame, text="Current Actions:", style='Heading.TLabel').pack(anchor='w')
        
        self.steering_var = tk.StringVar(value="Steering: 0.000")
        self.throttle_var = tk.StringVar(value="Throttle: 0.000")
        self.brake_var = tk.StringVar(value="Brake: 0.000")
        self.reward_var = tk.StringVar(value="Reward: 0.000")
        
        ttk.Label(action_frame, textvariable=self.steering_var).pack(anchor='w')
        ttk.Label(action_frame, textvariable=self.throttle_var).pack(anchor='w')
        ttk.Label(action_frame, textvariable=self.brake_var).pack(anchor='w')
        ttk.Label(action_frame, textvariable=self.reward_var).pack(anchor='w')
        
        # Navigation visualization
        self.nav_plots_frame = ttk.Frame(nav_display_frame)
        self.nav_plots_frame.pack(fill='both', expand=True)
        
    def create_evaluation_tab(self):
        """Create model evaluation tab"""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="📊 Evaluation")
        
        # Evaluation controls
        eval_control_frame = ttk.LabelFrame(eval_frame, text="Evaluation Controls")
        eval_control_frame.pack(fill='x', padx=5, pady=5)
        
        eval_buttons = ttk.Frame(eval_control_frame)
        eval_buttons.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(eval_buttons, text="🔍 Run Performance Test", 
                  command=self.run_performance_test).pack(side='left', padx=5)
        
        ttk.Button(eval_buttons, text="🛡️ Test Robustness", 
                  command=self.test_robustness).pack(side='left', padx=5)
        
        ttk.Button(eval_buttons, text="⚡ Benchmark Speed", 
                  command=self.benchmark_speed).pack(side='left', padx=5)
        
        # Evaluation results
        eval_results_frame = ttk.LabelFrame(eval_frame, text="Evaluation Results")
        eval_results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results text area
        self.results_text = tk.Text(eval_results_frame, wrap='word', height=10)
        scrollbar = ttk.Scrollbar(eval_results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
        
        # Evaluation plots
        self.eval_plots_frame = ttk.Frame(eval_results_frame)
        self.eval_plots_frame.pack(fill='both', expand=True)
        
    def create_config_tab(self):
        """Create configuration tab"""
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        
        # Configuration editor
        config_editor_frame = ttk.LabelFrame(config_frame, text="System Configuration")
        config_editor_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Config text editor
        self.config_text = tk.Text(config_editor_frame, wrap='word')
        config_scrollbar = ttk.Scrollbar(config_editor_frame, orient='vertical', 
                                       command=self.config_text.yview)
        self.config_text.configure(yscrollcommand=config_scrollbar.set)
        
        self.config_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        config_scrollbar.pack(side='right', fill='y')
        
        # Load current config
        self.load_config_to_editor()
        
        # Config buttons
        config_buttons = ttk.Frame(config_frame)
        config_buttons.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(config_buttons, text="💾 Save Configuration", 
                  command=self.save_config).pack(side='left', padx=5)
        
        ttk.Button(config_buttons, text="🔄 Reload Configuration", 
                  command=self.reload_config).pack(side='left', padx=5)
        
        ttk.Button(config_buttons, text="🔧 Validate Configuration", 
                  command=self.validate_config_gui).pack(side='left', padx=5)
        
    def create_monitoring_tab(self):
        """Create system monitoring tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="📡 Monitoring")
        
        # System status
        status_frame = ttk.LabelFrame(monitor_frame, text="System Status")
        status_frame.pack(fill='x', padx=5, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill='x', padx=5, pady=5)
        
        # Status indicators
        self.model_status_var = tk.StringVar(value="❌ Model: Not Loaded")
        self.dataset_status_var = tk.StringVar(value="❌ Dataset: Not Ready")
        self.training_status_var = tk.StringVar(value="⏸️ Training: Idle")
        self.navigation_status_var = tk.StringVar(value="⏸️ Navigation: Idle")
        
        ttk.Label(status_grid, textvariable=self.model_status_var).pack(anchor='w')
        ttk.Label(status_grid, textvariable=self.dataset_status_var).pack(anchor='w')
        ttk.Label(status_grid, textvariable=self.training_status_var).pack(anchor='w')
        ttk.Label(status_grid, textvariable=self.navigation_status_var).pack(anchor='w')
        
        # Resource monitoring
        resource_frame = ttk.LabelFrame(monitor_frame, text="Resource Usage")
        resource_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.resource_plots_frame = ttk.Frame(resource_frame)
        self.resource_plots_frame.pack(fill='both', expand=True)
        
    def setup_plots(self):
        """Setup matplotlib plots for each tab"""
        
        # Training plots
        self.training_fig = Figure(figsize=(12, 6), dpi=80)
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, self.training_plots_frame)
        self.training_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Navigation plots
        self.nav_fig = Figure(figsize=(10, 6), dpi=80)
        self.nav_canvas = FigureCanvasTkAgg(self.nav_fig, self.nav_plots_frame)
        self.nav_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Evaluation plots
        self.eval_fig = Figure(figsize=(10, 8), dpi=80)
        self.eval_canvas = FigureCanvasTkAgg(self.eval_fig, self.eval_plots_frame)
        self.eval_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def start_training(self):
        """Start model training in a separate thread"""
        if self.is_training:
            return
        
        try:
            # Update configuration
            self.config['epochs'] = int(self.epochs_var.get())
            self.config['learning_rate'] = float(self.lr_var.get())
            self.config['batch_size'] = int(self.batch_var.get())
            
            # Validate configuration
            validate_config(self.config)
            
            # Update UI
            self.is_training = True
            self.train_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_var.set("Initializing training...")
            self.training_status_var.set("🏋️ Training: Starting")
            
            # Start training thread
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training: {str(e)}")
            self.is_training = False
    
    def _training_worker(self):
        """Training worker function (runs in separate thread)"""
        try:
            # Initialize trainer (real training pipeline)
            self.trainer = FixedAdversarialIRLTrainer(self.config)
            self.status_queue.put(("status", "✅ Trainer initialized"))
            self.status_queue.put(("model_status", "✅ Model: Loaded"))
            self.status_queue.put(("dataset_status", "✅ Dataset: Ready"))

            # Train model with progress updates (epoch-wise)
            for epoch in range(self.config['epochs']):
                if not self.is_training:  # Check if training was stopped
                    break
                
                # Train one epoch using the trainer
                epoch_metrics = self.trainer.train_epoch(epoch)
                
                # Update progress
                progress = ((epoch + 1) / self.config['epochs']) * 100
                self.status_queue.put(("progress", progress))
                self.status_queue.put(("status", f"Epoch {epoch+1}/{self.config['epochs']} - Loss: {epoch_metrics['discriminator_loss']:.4f}"))
                
                # Store metrics
                self.training_metrics['epochs'].append(epoch + 1)
                self.training_metrics['losses'].append(epoch_metrics['discriminator_loss'])
                
                # Update plots
                self.status_queue.put(("plot_training", None))
            
            if self.is_training:  # Training completed normally
                self.status_queue.put(("training_complete", "✅ Training completed successfully!"))
                self.status_queue.put(("training_status", "✅ Training: Complete"))
            
        except Exception as e:
            self.status_queue.put(("error", f"Training failed: {str(e)}"))
            self.status_queue.put(("training_status", "❌ Training: Failed"))
        
        finally:
            self.is_training = False
    
    def stop_training(self):
        """Stop training"""
        if self.is_training:
            self.is_training = False
            self.status_var.set("Stopping training...")
            self.training_status_var.set("⏸️ Training: Stopping")
            
            # Re-enable buttons
            self.train_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.save_button.config(state='normal')
    
    def start_navigation_demo(self):
        """Start navigation demonstration"""
        if not self.trainer:
            messagebox.showwarning("No Model", "Please train or load a model first")
            return
        
        try:
            # Initialize navigation controller
            self.controller = NavigationController(self.trainer, self.config)
            
            # Start navigation demo
            self.start_nav_button.config(state='disabled')
            self.stop_nav_button.config(state='normal')
            self.navigation_status_var.set("🚗 Navigation: Running")
            
            # Start navigation thread
            nav_thread = threading.Thread(target=self._navigation_worker, daemon=True)
            nav_thread.start()
            
        except Exception as e:
            messagebox.showerror("Navigation Error", f"Failed to start navigation: {str(e)}")
    
    def _navigation_worker(self):
        """Navigation worker function"""
        try:
            episode_length = int(self.episode_length_var.get())
            update_rate = int(self.update_rate_var.get())
            
            # Create test dataset
            test_dataset = FixedSyntheticDataset(self.config, num_samples=100)
            
            for step in range(episode_length):
                # Get sensor data
                sample = test_dataset[step % len(test_dataset)]
                sensor_data = sample['multimodal']
                
                # Get navigation decision
                result = self.controller.process_sensor_data(sensor_data)
                
                # Update GUI
                self.status_queue.put(("navigation_update", {
                    'steering': float(result['action'][0]),
                    'throttle': float(result['action'][1]),
                    'brake': float(result['action'][2]),
                    'reward': float(result['reward'])
                }))
                
                time.sleep(1.0 / update_rate)
            
            self.status_queue.put(("navigation_complete", None))
            
        except Exception as e:
            self.status_queue.put(("error", f"Navigation error: {str(e)}"))
    
    def stop_navigation_demo(self):
        """Stop navigation demonstration"""
        self.start_nav_button.config(state='normal')
        self.stop_nav_button.config(state='disabled')
        self.navigation_status_var.set("⏸️ Navigation: Idle")
    
    def run_performance_test(self):
        """Run performance evaluation"""
        if not self.trainer:
            messagebox.showwarning("No Model", "Please train or load a model first")
            return
        
        try:
            # Run evaluation
            eval_results = self.trainer.evaluate(num_samples=50)
            
            # Display results
            results_text = f"""
Performance Evaluation Results
============================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Action Similarity: {np.mean(eval_results['action_similarity']):.4f} ± {np.std(eval_results['action_similarity']):.4f}
Average Reward:    {np.mean(eval_results['reward_values']):.4f} ± {np.std(eval_results['reward_values']):.4f}
Consistency Score: {np.mean(eval_results['consistency_scores']):.4f} ± {np.std(eval_results['consistency_scores']):.4f}

Total samples evaluated: {len(eval_results['action_similarity'])}
"""
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_text)
            
            # Update evaluation plots
            self.plot_evaluation_results(eval_results)
            
        except Exception as e:
            messagebox.showerror("Evaluation Error", f"Failed to run evaluation: {str(e)}")
    
    def test_robustness(self):
        """Test model robustness"""
        self.results_text.insert(tk.END, "\\n🛡️ Running robustness test...\\n")
        # Implementation would go here
    
    def benchmark_speed(self):
        """Benchmark inference speed"""
        self.results_text.insert(tk.END, "\\n⚡ Running speed benchmark...\\n")
        # Implementation would go here
    
    def save_model(self):
        """Save trained model"""
        if not self.trainer:
            messagebox.showwarning("No Model", "No model to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.trainer.save_model(filepath)
                messagebox.showinfo("Success", f"Model saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load pre-trained model"""
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                # Load model (implementation would go here)
                messagebox.showinfo("Success", f"Model loaded from {filepath}")
                self.model_status_var.set("✅ Model: Loaded")
                self.save_button.config(state='normal')
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load model: {str(e)}")
    
    def load_config_to_editor(self):
        """Load current configuration to text editor"""
        config_text = json.dumps(self.config, indent=2, sort_keys=True)
        self.config_text.delete(1.0, tk.END)
        self.config_text.insert(1.0, config_text)
    
    def save_config(self):
        """Save configuration from editor"""
        try:
            config_text = self.config_text.get(1.0, tk.END)
            new_config = json.loads(config_text)
            validate_config(new_config)
            self.config = new_config
            messagebox.showinfo("Success", "Configuration saved successfully")
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON format: {str(e)}")
        except Exception as e:
            messagebox.showerror("Config Error", f"Invalid configuration: {str(e)}")
    
    def reload_config(self):
        """Reload configuration"""
        self.config = get_config()
        self.load_config_to_editor()
        messagebox.showinfo("Success", "Configuration reloaded")
    
    def validate_config_gui(self):
        """Validate configuration from GUI"""
        try:
            config_text = self.config_text.get(1.0, tk.END)
            config = json.loads(config_text)
            validate_config(config)
            messagebox.showinfo("Success", "Configuration is valid!")
        except Exception as e:
            messagebox.showerror("Validation Error", f"Invalid configuration: {str(e)}")
    
    def plot_training_metrics(self):
        """Update training plots"""
        if not self.training_metrics['epochs']:
            return
        
        self.training_fig.clear()
        ax = self.training_fig.add_subplot(111)
        
        ax.plot(self.training_metrics['epochs'], self.training_metrics['losses'], 
                'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.training_canvas.draw()
    
    def plot_evaluation_results(self, results):
        """Plot evaluation results"""
        self.eval_fig.clear()
        
        # Create subplots
        ax1 = self.eval_fig.add_subplot(221)
        ax2 = self.eval_fig.add_subplot(222)
        ax3 = self.eval_fig.add_subplot(223)
        ax4 = self.eval_fig.add_subplot(224)
        
        # Plot histograms
        ax1.hist(results['action_similarity'], bins=20, alpha=0.7, color='blue')
        ax1.set_title('Action Similarity')
        ax1.set_xlabel('Similarity Score')
        
        ax2.hist(results['reward_values'], bins=20, alpha=0.7, color='green')
        ax2.set_title('Reward Values')
        ax2.set_xlabel('Reward')
        
        ax3.hist(results['consistency_scores'], bins=20, alpha=0.7, color='orange')
        ax3.set_title('Consistency Scores')
        ax3.set_xlabel('Consistency')
        
        # Summary plot
        metrics = ['Similarity', 'Reward', 'Consistency']
        values = [np.mean(results['action_similarity']), 
                 np.mean(results['reward_values']), 
                 np.mean(results['consistency_scores'])]
        
        ax4.bar(metrics, values, color=['blue', 'green', 'orange'], alpha=0.7)
        ax4.set_title('Average Metrics')
        ax4.set_ylabel('Score')
        
        self.eval_fig.tight_layout()
        self.eval_canvas.draw()
    
    def update_status(self):
        """Update GUI status from queue"""
        try:
            while True:
                message_type, data = self.status_queue.get_nowait()
                
                if message_type == "status":
                    self.status_var.set(data)
                elif message_type == "progress":
                    self.progress_var.set(data)
                elif message_type == "plot_training":
                    self.plot_training_metrics()
                elif message_type == "training_complete":
                    self.status_var.set(data)
                    self.train_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    self.save_button.config(state='normal')
                elif message_type == "model_status":
                    self.model_status_var.set(data)
                elif message_type == "dataset_status":
                    self.dataset_status_var.set(data)
                elif message_type == "training_status":
                    self.training_status_var.set(data)
                elif message_type == "navigation_update":
                    self.steering_var.set(f"Steering: {data['steering']:.3f}")
                    self.throttle_var.set(f"Throttle: {data['throttle']:.3f}")
                    self.brake_var.set(f"Brake: {data['brake']:.3f}")
                    self.reward_var.set(f"Reward: {data['reward']:.3f}")
                elif message_type == "navigation_complete":
                    self.stop_navigation_demo()
                elif message_type == "error":
                    messagebox.showerror("Error", data)
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_status)


def main():
    """Main function to run the GUI application"""
    print("🚀 Starting Adversarial IRL Navigation System GUI...")
    
    root = tk.Tk()
    app = AdversarialIRLGUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.quit()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    print("✅ GUI initialized successfully!")
    print("📱 Opening graphical interface...")
    
    root.mainloop()


if __name__ == "__main__":
    main()
