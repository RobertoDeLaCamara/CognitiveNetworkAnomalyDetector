# Contributing to Cognitive Anomaly Detector

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Prioritize project goals and quality

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/CognitiveNetworkAnomalyDetector.git
cd cognitive-anomaly-detector

# Add upstream remote
git remote add upstream https://github.com/RobertoDeLaCamara/CognitiveNetworkAnomalyDetector.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8
```

### 3. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style

We follow PEP 8 guidelines:

```bash
# Format code with black
black src/ tests/

# Check style
flake8 src/ tests/ --max-line-length=100
```

### Code Organization

- **src/**: Source code modules
- **tests/**: Test files (mirror src/ structure)
- **models/**: Trained models (gitignored)
- **data/**: Training data (gitignored)

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore()`

## Testing

### Writing Tests

All new features must include tests:

```python
# tests/test_your_feature.py
import pytest
from src.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self):
        obj = YourClass()
        result = obj.method()
        assert result == expected_value
    
    def test_error_handling(self):
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_your_feature.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Check coverage report
firefox htmlcov/index.html
```

### Test Coverage Requirements

- Minimum 80% coverage for new code
- All public methods should have tests
- Edge cases and error conditions must be tested

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def train_model(features, contamination=0.01):
    """Train Isolation Forest model.
    
    Args:
        features (np.ndarray): Feature matrix (n_samples, n_features)
        contamination (float): Expected anomaly proportion
    
    Returns:
        dict: Training statistics with metrics
    
    Raises:
        ValueError: If features have invalid shape
    
    Example:
        >>> trainer = ModelTrainer()
        >>> stats = trainer.train_model(features, contamination=0.01)
        >>> print(stats['training_time'])
    """
    pass
```

### API Documentation

Update [API.md](API.md) for any API changes:

- Add new methods with signatures
- Include parameter descriptions
- Provide usage examples
- Document return values and exceptions

### README Updates

Update [README.md](README.md) if you:

- Add new features
- Change usage patterns
- Modify configuration options
- Add new dependencies

## Pull Request Process

### 1. Before Submitting

Checklist:

- [ ] Code follows PEP 8 style
- [ ] All tests pass (`pytest tests/`)
- [ ] Coverage meets requirements (80%+)
- [ ] Documentation updated (docstrings, API.md, README.md)  
- [ ] MLflow integration tested (if applicable)
- [ ] Commit messages are clear and descriptive

### 2. Commit Messages

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add MinIO S3 storage support for MLflow artifacts"
git commit -m "Fix feature extraction bug for TCP packets"
git commit -m "Update API docs with new Detector methods"

# Bad examples
git commit -m "fix bug"
git commit -m "update"
git commit -m "wip"
```

### 3. Submit Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then on GitHub:

1. Create Pull Request from your fork
2. Fill in PR template:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this change needed?
   - **Testing**: How was it tested?
   - **Screenshots**: If UI changes
3. Link related issues
4. Request review

### 4. PR Requirements

Your PR will be reviewed for:

- **Functionality**: Does it work as intended?
- **Tests**: Are there sufficient tests?
- **Documentation**: Is it well documented?
- **Code Quality**: Is code clean and maintainable?
- **MLflow Integration**: Does it work with MLflow?
- **Backwards Compatibility**: Does it break existing functionality?

## Types of Contributions

### Bug Fixes

1. Create issue describing the bug
2. Create branch: `fix/issue-description`
3. Fix the bug and add regression test
4. Submit PR referencing the issue

### New Features

1. Discuss in an issue first
2. Get approval from maintainers
3. Create branch: `feature/feature-name`
4. Implement with tests and docs
5. Submit PR

### Documentation

1. Create branch: `docs/what-youre-documenting`
2. Make improvements
3. Submit PR

### Tests

1. Create branch: `test/what-youre-testing`
2. Add or improve tests
3. Submit PR

## Development Guidelines

### MLflow Integration

When adding MLflow features:

```python
# Check MLflow availability
if MLFLOW_AVAILABLE:
    # MLflow code here
    mlflow.log_metric('metric_name', value)
else:
    # Graceful degradation
    logger.warning("MLflow not available")
```

### Error Handling

```python
# Use specific exceptions
raise ValueError(f"Invalid feature shape: {features.shape}")

# Log errors appropriately
logger.error(f"Failed to load model: {e}")
```

### Configuration

```python
# Use config files for settings
from src.ml_config import CONTAMINATION,  N_ESTIMATORS

# Allow environment variable overrides
contamination = os.getenv('CONTAMINATION', CONTAMINATION)
```

## Project Structure

```
cognitive-anomaly-detector/
├── src/                    # Source code
│   ├── *.py               # Core modules
│   └── mlflow_config.py   # MLflow configuration
├── tests/                  # Test suite
│   └── test_*.py          # Test files
├── models/                 # Trained models (gitignored)
├── data/                   # Training data (gitignored)
├── *.md                    # Documentation
└── requirements.txt        # Dependencies
```

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making network anomaly detection better! 🎉
