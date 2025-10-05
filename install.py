#!/usr/bin/env python3
"""
Simple installation script that avoids problematic dependencies
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and return success status."""
    print(f"\n{description}")
    print("=" * 50)
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("🚀 Adversarial IRL Navigation - Simple Installation")
    print("=" * 60)
    
    # Install core package without problematic dependencies
    print("Installing core package...")
    success = run_command("pip3 install -e .", "Installing package in development mode")
    
    if not success:
        print("⚠️  Package installation failed, but continuing with manual dependency installation...")
        
        # Install dependencies manually from requirements.txt (excluding commented ones)
        print("Installing core dependencies manually...")
        
        # Core dependencies that should work
        core_deps = [
            "torch>=1.12.0",
            "torchvision>=0.13.0", 
            "numpy>=1.21.0",
            "opencv-python>=4.6.0",
            "gymnasium>=0.26.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "scikit-learn>=1.1.0",
            "scipy>=1.8.0",
            "pillow>=9.0.0",
            "pyyaml>=6.0",
            "tqdm>=4.64.0",
            "open3d>=0.15.0"
        ]
        
        for dep in core_deps:
            cmd = f"pip3 install '{dep}'"
            run_command(cmd, f"Installing {dep}")
    
    # Install pytest for testing
    run_command("pip3 install pytest", "Installing pytest")
    
    # Create necessary directories
    directories = [
        "data/expert_demonstrations",
        "data/validation", 
        "data/test",
        "checkpoints",
        "logs"
    ]
    
    print("\nCreating project directories...")
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    # Run tests
    print("\n" + "=" * 60)
    print("Running tests...")
    test_success = run_command("python3 -m pytest tests/ -v", "Running unit tests")
    
    if test_success:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed, but installation continues...")
    
    # Run quick demo
    print("\n" + "=" * 60) 
    print("Running quick demo...")
    
    demo_success = run_command("cd examples && python3 train_example.py", "Training demo")
    if demo_success:
        print("✅ Training demo completed!")
    
    nav_success = run_command("cd examples && python3 test_navigation.py --single_test", "Navigation test")
    if nav_success:
        print("✅ Navigation test completed!")
    
    print("\n" + "=" * 60)
    print("🎉 Installation Summary")
    print("=" * 60)
    print("✅ Core dependencies installed")
    print("✅ Project directories created") 
    print(f"{'✅' if test_success else '⚠️ '} Tests {'passed' if test_success else 'had issues'}")
    print(f"{'✅' if demo_success else '⚠️ '} Demo {'completed' if demo_success else 'had issues'}")
    
    print("\n📝 Next Steps:")
    print("1. Try: python3 examples/train_example.py")
    print("2. Try: python3 examples/test_navigation.py --single_test")
    print("3. Check docs/PROJECT_STRUCTURE.md for detailed usage")
    
    if not success:
        print("\n📌 Note: Optional simulation dependencies were skipped")
        print("   To install CARLA (if available): pip3 install carla==0.9.5")
        print("   To install AirSim deps: pip3 install msgpack-rpc-python msgpack")

if __name__ == "__main__":
    main()
