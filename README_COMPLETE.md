# 🚀 Adversarial IRL Navigation System

A complete **Adversarial Inverse Reinforcement Learning** system for autonomous navigation with multiple interfaces and comprehensive training/evaluation capabilities.

## 🎯 Overview

This project implements a state-of-the-art adversarial inverse reinforcement learning system designed for autonomous vehicle navigation. The system learns optimal driving behaviors by combining adversarial training with inverse reinforcement learning, enabling robust and human-like navigation patterns.

## ✨ Key Features

- **🧠 Advanced ML Architecture**: Multi-modal encoder with adversarial discriminator
- **🎮 Multiple Interfaces**: Desktop GUI, Web Dashboard, Interactive ML Interface, CLI, Jupyter Notebooks
- **📊 Real-time Monitoring**: Live training metrics and navigation visualization
- **⚙️ Configuration Management**: Easy parameter tuning and model customization
- **🔧 Complete Pipeline**: End-to-end training, testing, evaluation, and deployment
- **💾 Data Export**: Comprehensive results export and analysis tools

## 🏗️ System Architecture

```
Adversarial IRL Navigation System
├── 🧠 Core ML Components
│   ├── MultimodalEncoder (Vision + LiDAR + GPS fusion)
│   ├── Policy Network (Action prediction)
│   ├── Reward Network (Reward estimation)
│   └── Discriminator (Adversarial training)
├── 🎮 Multiple Interfaces
│   ├── Desktop GUI (Tkinter)
│   ├── Web Dashboard (Streamlit)
│   ├── ML Interface (Gradio)
│   ├── Command Line Interface
│   └── Jupyter Notebooks
├── 📊 Training & Evaluation
│   ├── Synthetic Data Generation
│   ├── Real-time Metrics Tracking
│   ├── Performance Evaluation
│   └── Model Comparison Tools
└── ⚙️ Configuration & Deployment
    ├── Flexible Configuration System
    ├── Model Export/Import
    ├── Results Analysis
    └── Production Deployment Tools
```

## 🚀 Quick Start

### 1. **Universal Launcher** (Recommended)
```bash
python launcher.py
```
Choose from multiple interface options in the interactive menu.

### 2. **Direct Interface Launch**
```bash
# Desktop GUI
python adversarial_irl_gui.py

# Web Dashboard
streamlit run adversarial_irl_web.py

# ML Interface  
python adversarial_irl_gradio.py

# Command Line
python fixed_train_complete.py

# Jupyter Notebook
jupyter notebook adversarial_irl_demo.ipynb
```

### 3. **Command Line Arguments**
```bash
# Launch specific interface directly
python launcher.py --interface gui
python launcher.py --interface web
python launcher.py --interface gradio
python launcher.py --interface cli
python launcher.py --interface jupyter

# Check dependencies
python launcher.py --check
```

## 📋 System Requirements

### Required Dependencies
- **Python 3.8+**
- **PyTorch 2.0+** (with MPS/CUDA support recommended)
- **NumPy**
- **Matplotlib**
- **Pandas**

### Optional Dependencies (for full functionality)
- **Streamlit** (for web interface)
- **Gradio** (for ML interface)
- **Plotly** (for interactive plots)
- **Jupyter** (for notebook interface)
- **Seaborn** (for enhanced visualizations)

### Installation
```bash
# Install core dependencies
pip install torch numpy matplotlib pandas

# Install all dependencies for full functionality
pip install torch numpy matplotlib pandas streamlit gradio plotly jupyter seaborn pillow dash fastapi
```

## 🎮 Interface Options

### 1. 🖥️ Desktop GUI (Tkinter)
**File**: `adversarial_irl_gui.py`
- **Native desktop application**
- **Real-time training monitoring**
- **Interactive navigation simulation**
- **Configuration editor**
- **Performance evaluation tools**

**Features**:
- Live plotting with matplotlib integration
- Multi-threaded operation for responsive UI
- Comprehensive system monitoring
- Model save/load functionality

### 2. 🌐 Web Dashboard (Streamlit)
**File**: `adversarial_irl_web.py`
- **Modern web-based interface**
- **Interactive dashboards**
- **Real-time data visualization**
- **Multi-page application**
- **Data export capabilities**

**Features**:
- Dashboard with key metrics
- Interactive training controls
- Navigation simulation with live updates
- Configuration management
- System monitoring and health checks

### 3. 🤖 ML Interface (Gradio)
**File**: `adversarial_irl_gradio.py`
- **Machine learning focused interface**
- **Easy experiment management**
- **Interactive model evaluation**
- **Shareable interface**
- **Progress tracking**

**Features**:
- Tabbed interface for different functions
- Progress bars for long-running operations
- Interactive plots and visualizations
- Model comparison tools
- Results export and sharing

### 4. 💻 Command Line Interface
**File**: `fixed_train_complete.py`
- **Terminal-based operation**
- **Scripting and automation friendly**
- **Detailed logging**
- **Batch processing**

**Features**:
- Complete training pipeline
- Configurable parameters
- Progress monitoring
- Results logging

### 5. 📓 Jupyter Notebook
**File**: `adversarial_irl_demo.ipynb` (auto-generated)
- **Interactive experimentation**
- **Step-by-step analysis**
- **Educational demonstrations**
- **Reproducible research**

**Features**:
- Complete system walkthrough
- Interactive training and evaluation
- Visualization and analysis
- Documentation and explanations

## ⚙️ Configuration

### Configuration File: `config/fixed_config.py`
```python
{
    # Training Parameters
    "epochs": 20,
    "batch_size": 4,
    "learning_rate": 0.0001,
    "device": "mps",  # or "cuda" or "cpu"
    
    # Model Architecture
    "vision_input_dim": 256,
    "lidar_input_dim": 256,
    "gps_input_dim": 128,
    "hidden_dim": 64,
    
    # Training Configuration
    "use_adversarial": true,
    "discriminator_lr": 0.0002,
    "generator_lr": 0.0001,
    
    # Navigation Parameters
    "max_episode_length": 100,
    "reward_scale": 1.0,
    "action_space": ["steering", "throttle", "brake"]
}
```

### Key Configuration Options:
- **Device Selection**: Automatic detection of MPS/CUDA/CPU
- **Model Dimensions**: Configurable neural network architectures
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Navigation Settings**: Episode length, action spaces, reward scaling

## 🧠 Model Architecture

### MultimodalEncoder
- **Vision Branch**: Processes camera/image data
- **LiDAR Branch**: Handles point cloud data
- **GPS Branch**: Incorporates positional information
- **Fusion Layer**: Combines all modalities

### Adversarial Training
- **Generator**: Policy network that produces actions
- **Discriminator**: Distinguishes expert vs. generated actions
- **Reward Network**: Estimates reward signals
- **Training Loop**: Alternating adversarial optimization

## 📊 Training & Evaluation

### Training Features:
- **Real-time Loss Monitoring**: Live plots of training progress
- **Convergence Detection**: Automatic stopping criteria
- **Checkpoint Saving**: Model state preservation
- **Hyperparameter Tuning**: Easy configuration adjustment

### Evaluation Metrics:
- **Action Similarity**: Comparison with expert demonstrations
- **Reward Consistency**: Stability of reward estimation
- **Navigation Performance**: Success rates and path quality
- **Robustness Testing**: Performance under different conditions

## 🚗 Navigation Simulation

### Simulation Features:
- **Real-time Control**: Steering, throttle, brake prediction
- **Environment Interaction**: Physics-based simulation
- **Reward Calculation**: Real-time performance metrics
- **Visualization**: Live plotting of vehicle trajectory

### Control Outputs:
- **Steering**: Continuous steering angle prediction
- **Throttle**: Acceleration control
- **Brake**: Braking intensity
- **Metadata**: Confidence scores, uncertainty estimates

## 📈 Performance Monitoring

### Real-time Metrics:
- **Training Loss**: Generator and discriminator losses
- **Reward Signals**: Average episode rewards
- **Action Statistics**: Control input distributions
- **System Resources**: CPU, memory, GPU utilization

### Evaluation Reports:
- **Performance Summaries**: Statistical analysis
- **Comparison Charts**: Model comparison
- **Success Rates**: Task completion metrics
- **Error Analysis**: Failure mode identification

## 💾 Data Management

### Data Export:
- **Training Metrics**: Loss curves, learning progress
- **Navigation Data**: Trajectory information, control inputs
- **Model Checkpoints**: Trained model states
- **Configuration**: Experimental settings
- **Results**: Performance evaluations

### Export Formats:
- **JSON**: Configuration and metadata
- **CSV**: Numerical data and metrics
- **PNG/PDF**: Plots and visualizations
- **PyTorch**: Model checkpoints

## 🔧 Development & Extension

### Adding New Interfaces:
1. Create new interface file
2. Import core components from `src/` and `config/`
3. Add to `launcher.py` menu
4. Update documentation

### Custom Models:
1. Extend base classes in `fixed_train_complete.py`
2. Update configuration in `fixed_config.py`
3. Modify data loading in `fixed_data_loader.py`
4. Test with existing interfaces

### New Evaluation Metrics:
1. Add metric calculations to evaluation classes
2. Update visualization functions
3. Integrate with existing interfaces
4. Export capabilities

## 🐛 Troubleshooting

### Common Issues:

**Import Errors**:
```bash
# Ensure all paths are correctly set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/config"
```

**Device Issues**:
```bash
# Check PyTorch device availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

**Memory Issues**:
- Reduce batch size in configuration
- Use CPU instead of GPU for large models
- Monitor system resources

**Interface Issues**:
- Check dependency installation
- Verify file paths and permissions
- Review error logs in terminal

### Debug Mode:
```bash
# Run with detailed logging
python launcher.py --interface cli --debug

# Check system status
python launcher.py --check
```

## 📚 Documentation & Support

### File Structure:
```
📁 adversarial_irl_navigation/
├── 📄 launcher.py                 # Universal interface launcher
├── 📄 adversarial_irl_gui.py      # Desktop GUI interface
├── 📄 adversarial_irl_web.py      # Streamlit web interface
├── 📄 adversarial_irl_gradio.py   # Gradio ML interface
├── 📄 fixed_train_complete.py     # Core training system
├── 📄 complete_navigation_test.py # Navigation controller
├── 📁 config/
│   └── 📄 fixed_config.py         # System configuration
├── 📁 src/
│   └── 📁 utils/
│       └── 📄 fixed_data_loader.py # Data loading utilities
├── 📄 README.md                   # This documentation
└── 📄 adversarial_irl_demo.ipynb  # Demo notebook (auto-generated)
```

### Key Components:
- **Launcher**: Universal interface selection and management
- **Core Training**: Adversarial IRL implementation
- **Interfaces**: Multiple user interface options
- **Configuration**: Flexible parameter management
- **Data Loading**: Synthetic and real data support
- **Evaluation**: Comprehensive testing and metrics

## 🤝 Contributing

### Development Setup:
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Launch development interface: `python launcher.py`

### Code Style:
- Follow PEP 8 guidelines
- Use type hints where applicable
- Document functions and classes
- Write tests for new features

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent ML framework
- Streamlit and Gradio for intuitive interface libraries
- The autonomous driving and IRL research community

---

**🚀 Get Started**: Run `python launcher.py` to begin exploring the Adversarial IRL Navigation System!

**📧 Support**: For questions or issues, please refer to the troubleshooting section or create an issue in the repository.

**🌟 Enjoy**: Happy training and navigation! 🚗💨
