# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-07

First community-ready release of the anomaly-detection framework, completing Phases 1–11 of the execution plan.

### Added

**Phase 1 — Foundation**
- Installable Python package at `src/anomaly_detection/`
- `pyproject.toml`, governance files (`LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`)
- CI workflow (ruff + pytest)
- Notebooks relocated to `examples/notebooks/`; legacy frontend deprecated

**Phase 2 — Detectors**
- `BaseDetector` ABC with `fit`, `score`, `predict`, `fit_predict`
- Statistical detectors: z-score, IQR
- ML detectors: Isolation Forest, LOF, One-Class SVM
- Deep detector: sklearn MLP autoencoder
- YAML-driven config and detector factory registry

**Phase 3 — CLI & REST API**
- `detect` CLI: config-driven detection with JSON report and plot output
- FastAPI service (`serve`): `/health`, `/detect`, `/detect/batch`, `/models`
- Evaluation metrics: precision, recall, F1, ROC-AUC

**Phase 4 — Dataset registry**
- `datasets/registry.yaml` with credit card fraud, NAB, and UCF crime entries
- Loaders for OpenML, CSV, and UCF fixtures
- `benchmark` CLI with markdown report output

**Phase 5 — Streaming**
- `BaseStreamingDetector` and sliding-window preprocessing
- Online z-score window detector; optional PySAD wrappers
- `stream` CLI and monitoring hooks (latency, memory, Slack alerts)

**Phase 6 — Root cause analysis**
- PyRCA-inspired causal graph and scorer modules
- `POST /root_cause` and `GET /root_cause/{anomaly_id}` API endpoints

**Phase 7 — Vision domain**
- UCF crime image/video classification module (supervised, 14 classes)
- `POST /vision/analyze/image` and `POST /vision/analyze/video` endpoints
- Optional `[vision]` extra (TensorFlow, OpenCV)

**Phase 8 — Fairness**
- AIF360-backed fairness metrics and mitigation (reweighing, threshold tuning)
- `docs/ETHICS.md` and fairness section in benchmark reports

**Phase 9 — Generative, LLM & multimodal**
- Diffusion-style reconstruction detector and GAN augmentation utilities
- LLM explainer with PII redaction (`POST /explain`)
- Multimodal tabular+text fusion module (experimental)

**Phase 10 — Governance & tutorials**
- Five domain tutorials in `docs/tutorials/`
- Performance profiler integrated into benchmark reports
- GitHub issue templates and phase labels

**Phase 11 — Release**
- Full verification matrix, anti-pattern cleanup, documentation audit
- `CHANGELOG.md`, `docs/RELEASE.md`, and comprehensive README

### Changed

- README rewritten to reflect implemented features (no aspirational claims)
- Package version bumped to 1.0.0

### Deprecated

- `examples/legacy-frontend/` static HTML site (broken `submit.php` reference removed from active code)

[1.0.0]: https://github.com/askmy-stack/anomaly-detection/compare/v0.5.0...v1.0.0
