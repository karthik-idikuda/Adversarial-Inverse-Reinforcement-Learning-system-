# Inverse Reinforcement Learning with Adversarial Multimodal Data for Autonomous Navigation

## Overview

This project implements an advanced autonomous navigation system that combines Inverse Reinforcement Learning (IRL) with adversarial training on multimodal sensor data. The system learns optimal driving behaviors from expert demonstrations while being robust to adversarial attacks and sensor noise.

## Key Features

- **Inverse Reinforcement Learning**: Learn reward functions from expert driving demonstrations
- **Adversarial Training**: Robust learning against adversarial perturbations
- **Multimodal Fusion**: Integration of camera, LiDAR, radar, and GPS data
- **Real-time Navigation**: Efficient path planning and control
- **Safety Mechanisms**: Built-in safety checks and fallback systems

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Sensor Data   │────│  Multimodal      │────│   IRL Agent     │
│ (Cam/LiDAR/GPS) │    │     Fusion       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         │              ┌──────────────────┐              │
         └──────────────│  Adversarial     │──────────────┘
                        │    Training      │
                        └──────────────────┘
                                 │
                        ┌──────────────────┐
                        │   Navigation     │
                        │    Controller    │
                        └──────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train the IRL model
python src/training/train_irl.py --config configs/irl_config.yaml

# Run adversarial training
python src/training/adversarial_training.py --config configs/adversarial_config.yaml

# Test navigation
python src/navigation/test_navigation.py --model_path models/irl_model.pth
```

## Dataset Structure

```
data/
├── expert_demonstrations/
│   ├── trajectories/
│   ├── sensor_data/
│   └── annotations/
├── simulation/
│   ├── carla_data/
│   └── airsim_data/
└── real_world/
    ├── camera/
    ├── lidar/
    └── gps/
```

## Configuration

All configurations are stored in the `configs/` directory. Key configuration files:
- `irl_config.yaml`: IRL training parameters
- `adversarial_config.yaml`: Adversarial training settings
- `sensor_config.yaml`: Sensor fusion parameters
- `navigation_config.yaml`: Navigation controller settings

## License

MIT License - See LICENSE file for details
# Adversarial-Inverse-Reinforcement-Learning-system-
