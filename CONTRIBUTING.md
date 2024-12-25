# Contributing

We welcome contributions to mcp-agent framework!

## Guidelines

- **Pull Requests**: Fork the repo, create a feature branch, and submit a PR. Please add tests for new features in `tests/`.
- **Code Style**: Use `ruff` for formatting Python code. Follow PEP 8 conventions.
- **Plugins & Recipes**: Implement as separate packages and expose them via entry points. See `plugins_loader.py` for how we discover plugins.
- **Testing**: Run `pytest` in the `tests/` directory. Ensure all tests pass before submitting a PR.
- **Documentation**: Add docstrings and comments where appropriate. Keep code clear and modular.

Thank you for contributing!
