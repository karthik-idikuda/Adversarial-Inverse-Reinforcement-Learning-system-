# Project Structure

```
adversarial-irl-navigation/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── adversarial_irl.py   # Main IRL models
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   └── train_irl.py         # Training logic
│   ├── navigation/              # Navigation system
│   │   ├── __init__.py
│   │   └── navigation_controller.py
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── data_loader.py       # Data loading utilities
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting functions
├── configs/                     # Configuration files
│   ├── irl_config.yaml         # IRL training config
│   ├── navigation_config.yaml  # Navigation config
│   └── sensor_config.yaml      # Sensor fusion config
├── examples/                    # Example scripts
│   ├── train_example.py        # Training example
│   └── test_navigation.py      # Testing example
├── tests/                       # Unit tests
│   └── test_adversarial_irl.py
├── data/                        # Data directory (created during runtime)
├── checkpoints/                 # Model checkpoints (created during training)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── README.md                   # Project documentation
```

## Key Components

### 1. Adversarial IRL Model (`src/models/adversarial_irl.py`)
- **MultimodalEncoder**: Processes camera, LiDAR, radar, and GPS data
- **RewardNetwork**: Learns reward function from expert demonstrations
- **PolicyNetwork**: Generates actions from states
- **Discriminator**: Distinguishes expert from policy trajectories
- **AdversarialIRLAgent**: Complete system combining all components

### 2. Training System (`src/training/train_irl.py`)
- **AdversarialIRLTrainer**: Main training class
- Implements Maximum Entropy IRL with adversarial robustness
- Supports multimodal sensor fusion
- Includes safety checks and performance monitoring

### 3. Navigation Controller (`src/navigation/navigation_controller.py`)
- **NavigationController**: Real-time navigation system
- **NavigationSimulator**: Testing and evaluation platform
- Implements safety checks and emergency braking
- Performance monitoring and metrics collection

### 4. Data Pipeline (`src/utils/data_loader.py`)
- **MultimodalNavigationDataset**: Loads real driving data
- **SyntheticNavigationDataset**: Generates synthetic training data
- Supports multiple data formats and augmentation

### 5. Evaluation System (`src/utils/metrics.py`)
- Comprehensive metrics for IRL evaluation
- Behavioral cloning metrics
- Safety and robustness evaluation
- Performance analysis tools

## Configuration Files

### IRL Training (`configs/irl_config.yaml`)
- Training hyperparameters
- Model architecture settings
- Data paths and preprocessing options
- Adversarial training parameters

### Navigation (`configs/navigation_config.yaml`)
- Vehicle parameters and limits
- Safety thresholds and constraints
- Control system settings
- Performance monitoring options

### Sensor Fusion (`configs/sensor_config.yaml`)
- Individual sensor configurations
- Fusion methodology settings
- Data quality control parameters
- Augmentation and robustness options

## Example Usage

### Training a Model
```bash
cd examples
python train_example.py
```

### Testing Navigation
```bash
python test_navigation.py --model ../checkpoints/best_model.pth --single_test
```

### Running Full Simulation
```bash
python test_navigation.py --model ../checkpoints/best_model.pth --full_simulation
```

## Data Format

The system expects multimodal sensor data in the following structure:

```
trajectory_XXXX/
├── camera/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── lidar/
│   ├── frame_000000.pcd
│   ├── frame_000001.pcd
│   └── ...
├── radar/
│   ├── frame_000000.json
│   ├── frame_000001.json
│   └── ...
├── gps/
│   ├── frame_000000.json
│   ├── frame_000001.json
│   └── ...
└── actions/
    ├── frame_000000.json
    ├── frame_000001.json
    └── ...
```

## Key Features

1. **Multimodal Sensor Fusion**: Integrates camera, LiDAR, radar, and GPS data
2. **Adversarial Robustness**: Robust to sensor noise and adversarial attacks
3. **Inverse Reinforcement Learning**: Learns from expert demonstrations
4. **Safety Systems**: Built-in emergency braking and safety checks
5. **Real-time Performance**: Optimized for real-time navigation
6. **Comprehensive Evaluation**: Extensive metrics and visualization tools

## Research Applications

This project is suitable for research in:
- Autonomous vehicle navigation
- Multimodal sensor fusion
- Adversarial machine learning
- Inverse reinforcement learning
- Safety-critical AI systems

## Extensions

The framework can be extended with:
- Additional sensor modalities
- Different IRL algorithms
- Advanced safety systems
- Integration with simulators (CARLA, AirSim)
- ROS integration for real robots
