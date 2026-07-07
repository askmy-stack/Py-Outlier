# Contributing

Thank you for your interest in contributing to anomaly-detection!

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install in editable mode with dev dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Create a feature branch from `main`:

   ```bash
   git checkout -b your-branch-name
   ```

## Development workflow

1. Make your changes following existing project structure and conventions.
2. Run lint and tests locally before opening a pull request:

   ```bash
   ruff check src tests
   pytest tests/ -q
   ```

3. Open a pull request against `main` with a clear description of the change.

## Phase labels

Work is organized in stacked phases aligned with [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md). When opening issues or pull requests, apply the matching label when applicable:

| Label | Phase focus |
| --- | --- |
| `good first issue` | Small, well-scoped starter tasks |
| `phase-1` | Foundation — package skeleton, CI |
| `phase-2` | Detectors and model registry |
| `phase-3` | CLI and REST API |
| `phase-4` | Dataset registry |
| `phase-5` | Streaming detection |
| `phase-6` | Root-cause analysis |
| `phase-7` | Vision domain (UCF) |
| `phase-8` | Fairness metrics and mitigation |
| `phase-9` | Generative models and LLM explainability |
| `phase-10` | Tutorials, profiling, governance |

Prefer small PRs that map to a single phase when possible.

## Pull request requirements

- CI must pass (ruff + pytest).
- Keep changes focused; prefer small, reviewable PRs aligned with [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md).
- Do not commit secrets, large raw datasets, or local virtual environments.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and constructive in all interactions.
