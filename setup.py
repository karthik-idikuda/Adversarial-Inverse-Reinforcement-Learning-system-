"""
Setup script for Adversarial IRL Navigation System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'numpy>=1.21.0',
        'opencv-python>=4.6.0',
        'gymnasium>=0.26.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'tensorboard>=2.8.0',
        'wandb>=0.12.0',
        'scikit-learn>=1.1.0',
        'scipy>=1.8.0',
        'pillow>=9.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.64.0'
    ]

setup(
    name="adversarial-irl-navigation",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Inverse Reinforcement Learning with Adversarial Multimodal Data for Autonomous Navigation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/adversarial-irl-navigation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "sim": [
            "carla>=0.9.5",
            "msgpack-rpc-python>=0.4.1",  # For AirSim compatibility
            "msgpack>=1.0.0",
        ],
        "pointcloud": [
            "open3d>=0.15.0",  # For LiDAR point cloud processing
        ],
        "ros": [
            "rospkg>=1.3.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
        "all": [
            "carla>=0.9.5",
            "msgpack-rpc-python>=0.4.1",
            "msgpack>=1.0.0",
            "open3d>=0.15.0",
            "rospkg>=1.3.0",
            "wandb>=0.12.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ]
    },
    entry_points={
        "console_scripts": [
            "irl-train=training.train_irl:main",
            "irl-navigate=navigation.navigation_controller:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
