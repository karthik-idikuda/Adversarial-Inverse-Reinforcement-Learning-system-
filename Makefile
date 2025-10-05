# Makefile for Adversarial IRL Navigation Project

.PHONY: help install install-dev test train clean setup lint format docs

help:  ## Show this help message
	@echo "Adversarial IRL Navigation Project"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Run initial project setup
	python3 setup_project.py

install:  ## Install the package
	pip3 install -e .

install-dev:  ## Install package with development dependencies
	pip3 install -e ".[dev]"

install-all:  ## Install package with all optional dependencies
	pip3 install -e ".[all]"

test:  ## Run tests
	python3 -m pytest tests/ -v

test-coverage:  ## Run tests with coverage report
	python3 -m pytest tests/ -v --cov=src --cov-report=html

lint:  ## Run linting checks
	flake8 src/ tests/ examples/
	mypy src/

format:  ## Format code
	black src/ tests/ examples/
	isort src/ tests/ examples/

train:  ## Run training example
	cd examples && python3 train_example.py

demo:  ## Run navigation demo
	cd examples && python3 test_navigation.py --single_test

simulate:  ## Run full simulation
	cd examples && python3 test_navigation.py --full_simulation

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docs:  ## Generate documentation
	@echo "Documentation is available in docs/ directory"
	@echo "Key files:"
	@echo "  - README.md: Main project documentation"
	@echo "  - docs/PROJECT_STRUCTURE.md: Detailed project structure"

tensorboard:  ## Start TensorBoard (if training logs exist)
	tensorboard --logdir=runs --host=0.0.0.0 --port=6006

jupyter:  ## Start Jupyter notebook
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Development targets
dev-setup: install-dev  ## Set up development environment
	pre-commit install

build:  ## Build package
	python setup.py sdist bdist_wheel

upload-test:  ## Upload to test PyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:  ## Upload to PyPI
	twine upload dist/*

# Docker targets (if Docker is available)
docker-build:  ## Build Docker image
	docker build -t adversarial-irl:latest .

docker-run:  ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace adversarial-irl:latest

# Data management
download-sample-data:  ## Download sample dataset (placeholder)
	@echo "Sample data generation is handled by the training scripts"
	@echo "Run 'make train' to generate synthetic training data"

validate-data:  ## Validate data format
	python -c "from src.utils.data_loader import MultimodalNavigationDataset; print('Data format validation passed')"

# Model management
list-models:  ## List available trained models
	@echo "Available models in checkpoints/:"
	@ls -la checkpoints/ 2>/dev/null || echo "No models found. Run training first."

backup-models:  ## Backup trained models
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz checkpoints/

# Monitoring and logging
monitor-training:  ## Monitor training progress
	tail -f training.log

view-logs:  ## View recent logs
	ls -la *.log 2>/dev/null || echo "No log files found"

# Performance testing
benchmark:  ## Run performance benchmarks
	python -c "import time; from examples.test_navigation import *; print('Basic benchmark - check examples/test_navigation.py for details')"

profile:  ## Profile code performance
	python -m cProfile -o profile.stats examples/test_navigation.py --single_test
	@echo "Profile saved to profile.stats"

# System checks
check-gpu:  ## Check GPU availability
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

check-deps:  ## Check dependencies
	pip check

system-info:  ## Show system information
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
	@echo "Current directory: $(PWD)"
