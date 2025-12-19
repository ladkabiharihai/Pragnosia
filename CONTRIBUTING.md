# Contributing to Pragnosia

Thank you for your interest in contributing to Pragnosia! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/pragnosia.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -e .`
6. Install dev dependencies: `pip install pytest black flake8 mypy`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `python tests/test_basic.py`
4. Format code: `black src/`
5. Lint: `flake8 src/`
6. Commit: `git commit -m "Description of changes"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions/classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Example

```python
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute cross-entropy loss.

    Args:
        predictions: Model predictions (batch, vocab_size)
        targets: Target labels (batch,)
        reduction: Reduction method ("mean", "sum", "none")

    Returns:
        loss: Computed loss value
    """
    return F.cross_entropy(predictions, targets, reduction=reduction)
```

## Testing

### Running Tests

```bash
# All tests
python tests/test_basic.py

# Specific test
python -c "from tests.test_basic import test_router; test_router()"
```

### Writing Tests

Add tests to `tests/test_basic.py` or create new test files:

```python
def test_new_feature():
    """Test description."""
    # Setup
    config = PragnosiaConfig()
    model = PragnosiaModel(config)

    # Test
    result = model.new_feature()

    # Assert
    assert result is not None
    print("✓ New feature working")
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(arg1: int, arg2: str = "default") -> bool:
    """
    Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
    """
    pass
```

### Documentation Files

- `README.md`: Overview, quick start, examples
- `docs/ARCHITECTURE.md`: Detailed technical documentation
- `docs/API.md`: API reference
- Code comments: Explain "why", not "what"

## Areas for Contribution

### High Priority

1. **Large-Scale Training**: Validate at 7B+ parameters
2. **RLHF Integration**: Combine with alignment methods
3. **Performance Optimization**: Speed up consolidation, routing
4. **Multi-GPU Support**: Distributed training
5. **Additional Benchmarks**: More continual learning tasks

### Medium Priority

1. **Improved Multimodal**: Better vision/audio encoders
2. **Hierarchical Experts**: Experts-of-experts
3. **Better Memory Management**: More efficient consolidation
4. **Monitoring Tools**: Better visualization
5. **Documentation**: Tutorials, examples

### Low Priority

1. **Code Cleanup**: Refactoring, optimization
2. **Additional Tests**: Edge cases, integration tests
3. **Type Annotations**: Complete coverage
4. **Logging**: More detailed logging options

## Pull Request Guidelines

### Before Submitting

- [ ] Tests pass
- [ ] Code is formatted (black)
- [ ] Code is linted (flake8)
- [ ] Documentation is updated
- [ ] Commit messages are clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted
- [ ] No breaking changes (or documented)
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Clear description of feature
- Use cases
- Proposed implementation (optional)
- Willingness to implement

## Code Review Process

1. Maintainer reviews PR
2. Feedback/changes requested
3. Update PR based on feedback
4. Approval and merge

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue with the "question" label
- Check existing issues and documentation first

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Acknowledgments in papers (for significant contributions)

Thank you for contributing to Pragnosia!
