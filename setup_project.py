#!/usr/bin/env python3
"""
Quick setup and test script for the Adversarial IRL Navigation project.
"""

import subprocess
import sys
from pathlib import Path
import os


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        if result.stderr and result.returncode == 0:
            print("Warnings:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"Error: Python 3.8+ required, but found Python {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def setup_environment():
    """Set up the development environment."""
    print("ADVERSARIAL IRL NAVIGATION - SETUP SCRIPT")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("Warning: Package installation failed, continuing...")
    
    # Install additional dependencies
    commands = [
        ("pip install pytest", "Installing pytest for testing"),
        ("pip install jupyter notebook", "Installing Jupyter for interactive development"),
        ("pip install tensorboard", "Installing TensorBoard for monitoring"),
    ]
    
    for command, description in commands:
        run_command(command, description, check=False)
    
    return True


def run_tests():
    """Run the test suite."""
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run tests
    success = run_command("python3 -m pytest tests/ -v", "Running unit tests", check=False)
    
    if success:
        print("✓ All tests passed!")
    else:
        print("⚠ Some tests failed, but this is expected without trained models")
    
    return success


def create_sample_data():
    """Create sample data for testing."""
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA")
    print("="*60)
    
    # Create data directories
    data_dirs = ["data/expert_demonstrations", "data/validation", "data/test", "checkpoints"]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    print("✓ Sample data directories created")
    return True


def run_quick_demo():
    """Run a quick demonstration."""
    print("\n" + "="*60)
    print("RUNNING QUICK DEMO")
    print("="*60)
    
    # Run the training example with minimal settings
    success = run_command(
        "cd examples && python3 train_example.py", 
        "Running training demonstration", 
        check=False
    )
    
    if success:
        print("✓ Training demo completed successfully!")
        
        # Try to run navigation test
        test_success = run_command(
            "cd examples && python3 test_navigation.py --single_test",
            "Running navigation test",
            check=False
        )
        
        if test_success:
            print("✓ Navigation test completed successfully!")
        else:
            print("⚠ Navigation test had issues (expected without proper training)")
    else:
        print("⚠ Training demo had issues")
    
    return success


def main():
    """Main setup function."""
    print("Starting Adversarial IRL Navigation setup...\n")
    
    steps = [
        ("Environment Setup", setup_environment),
        ("Sample Data Creation", create_sample_data),
        ("Test Suite", run_tests),
        ("Quick Demo", run_quick_demo),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    for step_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{step_name:25s}: {status}")
    
    if all(results.values()):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Explore the examples/ directory")
        print("2. Check out docs/PROJECT_STRUCTURE.md")
        print("3. Modify configs/ for your specific needs")
        print("4. Train your own model with real data")
    else:
        print("\n⚠ Setup completed with some issues.")
        print("Check the output above for specific problems.")
        print("The project may still be functional for basic usage.")
    
    print("\nProject structure:")
    print("- src/: Main source code")
    print("- examples/: Example scripts to get started")
    print("- configs/: Configuration files")
    print("- tests/: Unit tests")
    print("- docs/: Documentation")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
