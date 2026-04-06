# Contributing to cognitive-anomaly-detector

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/cognitive-anomaly-detector.git
   cd cognitive-anomaly-detector
   ```
3. Set up the development environment:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # if applicable
   ```
4. Generate baseline data and train the model:
   ```bash
   python generate_synthetic_data.py
   python train_model.py --from-file data/training/synthetic_baseline.csv
   ```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature` or `git checkout -b fix/issue-description`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Commit with a clear message: `git commit -m "feat: add new feature"`
5. Push and open a Pull Request

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

All new code should include tests. Aim to maintain or improve coverage.

## Running the Application

- **Packet capture** (requires root): `sudo python main.py`
- **Dashboard:** `python dashboard.py` (Streamlit UI)

## ML Tracking Experiment Tracking

Experiments are tracked with ML Tracking. When modifying models or the feature pipeline:
- Log parameters, metrics, and artifacts consistently
- Use meaningful experiment and run names
- Check `mlruns/` for existing experiment history

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Add docstrings to public functions and classes
- Keep the 18-feature extraction pipeline consistent when adding features
- Use Scapy best practices for packet handling
- Use clear, descriptive variable names

## Commit Messages

Use [conventional commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` adding or updating tests
- `refactor:` code restructuring

## Reporting Issues

- Use the issue templates (Bug Report or Feature Request)
- Include steps to reproduce for bugs
- Mention OS and Python version (root access may affect behavior)
- Check existing issues before creating a new one

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to learn and build.
