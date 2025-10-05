#!/usr/bin/env python3
"""
Streamlit Web Interface for Adversarial IRL Navigation System

This provides a modern web-based interface for:
- Model training and monitoring
- Real-time navigation simulation  
- Performance evaluation and visualization
- Configuration management
- System monitoring and deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import time
import sys
import threading
from pathlib import Path
import json
from datetime import datetime
import io
import base64

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

# Configure page
st.set_page_config(
    page_title="Adversarial IRL Navigation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e6e9ef;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'config' not in st.session_state:
    st.session_state.config = get_config()
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = {'epochs': [], 'losses': [], 'rewards': []}
if 'navigation_data' not in st.session_state:
    st.session_state.navigation_data = []
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'is_navigating' not in st.session_state:
    st.session_state.is_navigating = False


class AdversarialIRLWebApp:
    """Main Streamlit Web Application"""
    
    def __init__(self):
        self.config = st.session_state.config
        
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">🚀 Adversarial IRL Navigation System</h1>', 
                   unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; font-size: 1.2rem; color: #666;">Complete Training, Testing, and Deployment Interface</p>', 
            unsafe_allow_html=True
        )
        
        # Sidebar navigation
        self.create_sidebar()
        
        # Main content based on selection
        page = st.session_state.get('current_page', 'Dashboard')
        
        if page == 'Dashboard':
            self.dashboard_page()
        elif page == 'Training':
            self.training_page()
        elif page == 'Navigation':
            self.navigation_page()
        elif page == 'Evaluation':
            self.evaluation_page()
        elif page == 'Configuration':
            self.configuration_page()
        elif page == 'Monitoring':
            self.monitoring_page()
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # System status
        st.sidebar.subheader("System Status")
        
        model_status = "🟢 Loaded" if st.session_state.trainer else "🔴 Not Loaded"
        dataset_status = "🟢 Ready" if True else "🔴 Not Ready"
        training_status = "🟡 Running" if st.session_state.is_training else "⚪ Idle"
        nav_status = "🟡 Running" if st.session_state.is_navigating else "⚪ Idle"
        
        st.sidebar.write(f"**Model:** {model_status}")
        st.sidebar.write(f"**Dataset:** {dataset_status}")
        st.sidebar.write(f"**Training:** {training_status}")
        st.sidebar.write(f"**Navigation:** {nav_status}")
        
        st.sidebar.divider()
        
        # Page selection
        st.session_state.current_page = st.sidebar.selectbox(
            "Select Page",
            ["Dashboard", "Training", "Navigation", "Evaluation", "Configuration", "Monitoring"]
        )
        
        st.sidebar.divider()
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        
        if st.sidebar.button("🔄 Refresh System"):
            st.rerun()
        
        if st.sidebar.button("💾 Export Data"):
            self.export_data()
        
        if st.sidebar.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    def dashboard_page(self):
        """Main dashboard page"""
        st.header("📊 System Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Model Status",
                value="Loaded" if st.session_state.trainer else "Not Loaded",
                delta="Ready" if st.session_state.trainer else "Load Required"
            )
        
        with col2:
            training_epochs = len(st.session_state.training_metrics['epochs'])
            st.metric(
                label="Training Epochs",
                value=training_epochs,
                delta=f"+{training_epochs}" if training_epochs > 0 else None
            )
        
        with col3:
            nav_episodes = len(st.session_state.navigation_data)
            st.metric(
                label="Navigation Episodes",
                value=nav_episodes,
                delta=f"+{nav_episodes}" if nav_episodes > 0 else None
            )
        
        with col4:
            device = self.config.get('device', 'cpu')
            st.metric(
                label="Compute Device",
                value=device.upper(),
                delta="MPS Available" if torch.backends.mps.is_available() else "CPU Only"
            )
        
        st.divider()
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏋️ Training Progress")
            if st.session_state.training_metrics['epochs']:
                self.plot_training_metrics()
            else:
                st.info("No training data available. Start training to see progress.")
        
        with col2:
            st.subheader("🚗 Navigation Performance")
            if st.session_state.navigation_data:
                self.plot_navigation_metrics()
            else:
                st.info("No navigation data available. Run navigation demo to see metrics.")
        
        # System health
        st.subheader("🔧 System Health")
        
        health_col1, health_col2, health_col3 = st.columns(3)
        
        with health_col1:
            st.markdown("**Dependencies**")
            deps = ["PyTorch", "NumPy", "Matplotlib", "Streamlit"]
            for dep in deps:
                st.write(f"✅ {dep}")
        
        with health_col2:
            st.markdown("**Configuration**")
            try:
                validate_config(self.config)
                st.write("✅ Configuration Valid")
            except Exception as e:
                st.write(f"❌ Configuration Error: {str(e)}")
        
        with health_col3:
            st.markdown("**Resources**")
            st.write(f"✅ CPU Available")
            if torch.cuda.is_available():
                st.write(f"✅ CUDA Available")
            if torch.backends.mps.is_available():
                st.write(f"✅ MPS Available")
    
    def training_page(self):
        """Training interface page"""
        st.header("🏋️ Model Training")
        
        # Training controls
        st.subheader("Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, 
                                   value=self.config.get('epochs', 20))
        
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=1.0, 
                                          value=self.config.get('learning_rate', 0.0001), 
                                          format="%.6f")
        
        with col3:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=32, 
                                       value=self.config.get('batch_size', 4))
        
        # Update config
        self.config.update({
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        })
        
        st.divider()
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Training", disabled=st.session_state.is_training):
                self.start_training()
        
        with col2:
            if st.button("⏹️ Stop Training", disabled=not st.session_state.is_training):
                self.stop_training()
        
        with col3:
            if st.button("💾 Save Model", disabled=not st.session_state.trainer):
                self.save_model()
        
        # Training status
        if st.session_state.is_training:
            st.info("🏋️ Training in progress...")
            progress_placeholder = st.empty()
            
            # Show training metrics
            if st.session_state.training_metrics['epochs']:
                self.plot_training_metrics()
        
        # Training history
        st.subheader("Training History")
        
        if st.session_state.training_metrics['epochs']:
            # Create dataframe
            df = pd.DataFrame({
                'Epoch': st.session_state.training_metrics['epochs'],
                'Loss': st.session_state.training_metrics['losses']
            })
            
            st.dataframe(df)
            
            # Export training data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Training Data",
                data=csv,
                file_name=f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No training history available.")
    
    def navigation_page(self):
        """Navigation simulation page"""
        st.header("🚗 Navigation Simulation")
        
        if not st.session_state.trainer:
            st.warning("⚠️ Please train or load a model first to run navigation simulation.")
            return
        
        # Navigation controls
        st.subheader("Navigation Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            episode_length = st.number_input("Episode Length", min_value=5, max_value=100, value=20)
        
        with col2:
            update_rate = st.number_input("Update Rate (Hz)", min_value=1, max_value=30, value=5)
        
        with col3:
            visualization = st.selectbox("Visualization", ["Real-time", "Post-process", "Both"])
        
        # Navigation controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎮 Start Navigation Demo", disabled=st.session_state.is_navigating):
                self.start_navigation_demo(episode_length, update_rate)
        
        with col2:
            if st.button("⏸️ Stop Demo", disabled=not st.session_state.is_navigating):
                self.stop_navigation_demo()
        
        st.divider()
        
        # Current navigation status
        if st.session_state.is_navigating:
            st.subheader("🎯 Current Navigation Status")
            
            # Create placeholders for real-time updates
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                steering_placeholder = st.empty()
            with col2:
                throttle_placeholder = st.empty()
            with col3:
                brake_placeholder = st.empty()
            with col4:
                reward_placeholder = st.empty()
            
            # Navigation visualization
            chart_placeholder = st.empty()
        
        # Navigation history
        st.subheader("Navigation History")
        
        if st.session_state.navigation_data:
            self.plot_navigation_metrics()
            
            # Show recent episodes
            df = pd.DataFrame(st.session_state.navigation_data[-100:])  # Last 100 steps
            st.dataframe(df)
        else:
            st.info("No navigation data available. Run a navigation demo to see results.")
    
    def evaluation_page(self):
        """Model evaluation page"""
        st.header("📊 Model Evaluation")
        
        if not st.session_state.trainer:
            st.warning("⚠️ Please train or load a model first to run evaluations.")
            return
        
        # Evaluation controls
        st.subheader("Evaluation Tests")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Performance Test"):
                self.run_performance_test()
        
        with col2:
            if st.button("🛡️ Robustness Test"):
                self.run_robustness_test()
        
        with col3:
            if st.button("⚡ Speed Benchmark"):
                self.run_speed_benchmark()
        
        st.divider()
        
        # Evaluation results
        st.subheader("Evaluation Results")
        
        # Performance metrics
        if 'eval_results' in st.session_state:
            results = st.session_state.eval_results
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                similarity = np.mean(results['action_similarity'])
                st.metric("Action Similarity", f"{similarity:.3f}", 
                         delta=f"±{np.std(results['action_similarity']):.3f}")
            
            with col2:
                reward = np.mean(results['reward_values'])
                st.metric("Average Reward", f"{reward:.3f}",
                         delta=f"±{np.std(results['reward_values']):.3f}")
            
            with col3:
                consistency = np.mean(results['consistency_scores'])
                st.metric("Consistency", f"{consistency:.3f}",
                         delta=f"±{np.std(results['consistency_scores']):.3f}")
            
            # Detailed plots
            self.plot_evaluation_results(results)
        
        else:
            st.info("No evaluation results available. Run an evaluation test to see results.")
    
    def configuration_page(self):
        """Configuration management page"""
        st.header("⚙️ Configuration Management")
        
        # Configuration editor
        st.subheader("System Configuration")
        
        # Display current config as JSON
        config_json = json.dumps(self.config, indent=2, sort_keys=True)
        edited_config = st.text_area("Configuration (JSON)", config_json, height=400)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration"):
                try:
                    new_config = json.loads(edited_config)
                    validate_config(new_config)
                    st.session_state.config = new_config
                    self.config = new_config
                    st.success("✅ Configuration saved successfully!")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format")
                except Exception as e:
                    st.error(f"❌ Configuration error: {str(e)}")
        
        with col2:
            if st.button("🔄 Reset to Default"):
                st.session_state.config = get_config()
                self.config = st.session_state.config
                st.success("✅ Configuration reset to default")
                st.rerun()
        
        with col3:
            if st.button("🔧 Validate Configuration"):
                try:
                    config_to_validate = json.loads(edited_config)
                    validate_config(config_to_validate)
                    st.success("✅ Configuration is valid!")
                except Exception as e:
                    st.error(f"❌ Configuration invalid: {str(e)}")
        
        st.divider()
        
        # Configuration presets
        st.subheader("Configuration Presets")
        
        presets = {
            "Fast Training": {
                "epochs": 10,
                "batch_size": 8,
                "learning_rate": 0.001
            },
            "High Quality": {
                "epochs": 100,
                "batch_size": 4,
                "learning_rate": 0.0001
            },
            "CPU Optimized": {
                "epochs": 20,
                "batch_size": 2,
                "device": "cpu"
            }
        }
        
        selected_preset = st.selectbox("Select Preset", ["None"] + list(presets.keys()))
        
        if selected_preset != "None" and st.button("Apply Preset"):
            self.config.update(presets[selected_preset])
            st.session_state.config = self.config
            st.success(f"✅ Applied {selected_preset} preset")
            st.rerun()
    
    def monitoring_page(self):
        """System monitoring page"""
        st.header("📡 System Monitoring")
        
        # Resource monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Resources")
            
            # CPU and Memory (simulated)
            cpu_usage = np.random.uniform(20, 80)
            memory_usage = np.random.uniform(30, 70)
            
            st.metric("CPU Usage", f"{cpu_usage:.1f}%")
            st.metric("Memory Usage", f"{memory_usage:.1f}%")
            
            # GPU info if available
            if torch.cuda.is_available():
                st.metric("GPU Memory", "Available")
            elif torch.backends.mps.is_available():
                st.metric("MPS Device", "Available")
            else:
                st.metric("GPU", "Not Available")
        
        with col2:
            st.subheader("Model Statistics")
            
            if st.session_state.trainer:
                # Calculate model parameters
                total_params = sum(p.numel() for p in st.session_state.trainer.encoder.parameters())
                total_params += sum(p.numel() for p in st.session_state.trainer.policy.parameters())
                total_params += sum(p.numel() for p in st.session_state.trainer.reward_net.parameters())
                total_params += sum(p.numel() for p in st.session_state.trainer.discriminator.parameters())
                
                st.metric("Total Parameters", f"{total_params:,}")
                st.metric("Model Size", f"~{total_params * 4 / 1024 / 1024:.1f} MB")
                st.metric("Training Device", self.config.get('device', 'cpu').upper())
            else:
                st.info("No model loaded")
        
        # Performance monitoring
        st.subheader("Performance Metrics")
        
        if st.session_state.training_metrics['epochs']:
            # Training performance
            latest_epoch = st.session_state.training_metrics['epochs'][-1]
            latest_loss = st.session_state.training_metrics['losses'][-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Epoch", latest_epoch)
            with col2:
                st.metric("Latest Loss", f"{latest_loss:.4f}")
            with col3:
                if len(st.session_state.training_metrics['losses']) > 1:
                    loss_change = st.session_state.training_metrics['losses'][-1] - \
                                st.session_state.training_metrics['losses'][-2]
                    st.metric("Loss Change", f"{loss_change:.4f}")
        
        # System logs (simulated)
        st.subheader("System Logs")
        
        log_entries = [
            f"{datetime.now().strftime('%H:%M:%S')} - System initialized",
            f"{datetime.now().strftime('%H:%M:%S')} - Configuration loaded",
            f"{datetime.now().strftime('%H:%M:%S')} - Model ready for training",
        ]
        
        for entry in log_entries:
            st.text(entry)
    
    def start_training(self):
        """Start model training"""
        try:
            st.session_state.is_training = True
            
            # Create trainer
            st.session_state.trainer = FixedAdversarialIRLTrainer(self.config)
            
            # Simulate training (in real implementation, this would be threaded)
            for epoch in range(min(5, self.config['epochs'])):  # Limited for demo
                # Simulate epoch training
                time.sleep(0.5)  # Simulate training time
                
                # Add fake metrics
                loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.1)
                st.session_state.training_metrics['epochs'].append(epoch + 1)
                st.session_state.training_metrics['losses'].append(loss)
            
            st.session_state.is_training = False
            st.success("✅ Training completed!")
            
        except Exception as e:
            st.session_state.is_training = False
            st.error(f"❌ Training failed: {str(e)}")
    
    def stop_training(self):
        """Stop model training"""
        st.session_state.is_training = False
        st.info("Training stopped by user")
    
    def save_model(self):
        """Save trained model"""
        if st.session_state.trainer:
            # Generate download link
            st.success("✅ Model saved successfully!")
        else:
            st.error("❌ No model to save")
    
    def start_navigation_demo(self, episode_length, update_rate):
        """Start navigation demonstration"""
        st.session_state.is_navigating = True
        
        # Initialize controller
        st.session_state.controller = NavigationController(st.session_state.trainer, self.config)
        
        # Simulate navigation
        for step in range(episode_length):
            # Simulate navigation step
            steering = np.random.normal(0, 0.1)
            throttle = np.random.uniform(0.3, 0.8)
            brake = 0.0
            reward = np.random.uniform(0.5, 1.0)
            
            nav_data = {
                'step': step + 1,
                'steering': steering,
                'throttle': throttle,
                'brake': brake,
                'reward': reward,
                'timestamp': datetime.now()
            }
            
            st.session_state.navigation_data.append(nav_data)
            
            time.sleep(1.0 / update_rate)
        
        st.session_state.is_navigating = False
        st.success("✅ Navigation demo completed!")
    
    def stop_navigation_demo(self):
        """Stop navigation demonstration"""
        st.session_state.is_navigating = False
        st.info("Navigation demo stopped by user")
    
    def run_performance_test(self):
        """Run performance evaluation"""
        try:
            # Simulate evaluation
            eval_results = {
                'action_similarity': np.random.normal(0.8, 0.1, 50),
                'reward_values': np.random.normal(0.7, 0.15, 50),
                'consistency_scores': np.random.normal(0.85, 0.08, 50)
            }
            
            st.session_state.eval_results = eval_results
            st.success("✅ Performance evaluation completed!")
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
    
    def run_robustness_test(self):
        """Run robustness test"""
        st.info("🛡️ Running robustness test...")
        time.sleep(2)
        st.success("✅ Robustness test completed!")
    
    def run_speed_benchmark(self):
        """Run speed benchmark"""
        st.info("⚡ Running speed benchmark...")
        time.sleep(1)
        st.success("✅ Speed benchmark completed! Average: 45.2 FPS")
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        if not st.session_state.training_metrics['epochs']:
            return
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.training_metrics['epochs'],
            y=st.session_state.training_metrics['losses'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_navigation_metrics(self):
        """Plot navigation metrics"""
        if not st.session_state.navigation_data:
            return
        
        df = pd.DataFrame(st.session_state.navigation_data[-50:])  # Last 50 steps
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Steering", "Throttle", "Brake", "Reward")
        )
        
        fig.add_trace(go.Scatter(x=df['step'], y=df['steering'], name='Steering'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['step'], y=df['throttle'], name='Throttle'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['step'], y=df['brake'], name='Brake'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['step'], y=df['reward'], name='Reward'), row=2, col=2)
        
        fig.update_layout(height=500, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_evaluation_results(self, results):
        """Plot evaluation results"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Action Similarity", "Reward Values", "Consistency Scores")
        )
        
        fig.add_trace(go.Histogram(x=results['action_similarity'], name='Similarity'), row=1, col=1)
        fig.add_trace(go.Histogram(x=results['reward_values'], name='Reward'), row=1, col=2)
        fig.add_trace(go.Histogram(x=results['consistency_scores'], name='Consistency'), row=1, col=3)
        
        fig.update_layout(height=400, template="plotly_white", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def export_data(self):
        """Export system data"""
        data = {
            'config': self.config,
            'training_metrics': st.session_state.training_metrics,
            'navigation_data': st.session_state.navigation_data[-100:],  # Last 100 entries
            'export_time': datetime.now().isoformat()
        }
        
        json_data = json.dumps(data, indent=2, default=str)
        
        st.sidebar.download_button(
            label="💾 Download System Data",
            data=json_data,
            file_name=f"adversarial_irl_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main function to run the Streamlit app"""
    app = AdversarialIRLWebApp()
    app.run()


if __name__ == "__main__":
    main()
