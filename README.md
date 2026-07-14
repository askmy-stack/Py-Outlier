# Py-outlier

**Modular anomaly detection for tabular, time-series, and vision data — CLI, REST API, streaming, RCA, fairness, and optional LLM explanations.**

[![CI](https://github.com/askmy-stack/Py-Outlier/actions/workflows/ci.yml/badge.svg)](https://github.com/askmy-stack/Py-Outlier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

![Py-outlier demo](assets/py-outlier-demo.gif)

**Py-outlier** is a community-ready Python framework evolved from UCF crime-classification notebooks into a modular anomaly-detection toolkit. Ship detectors in production via CLI or FastAPI, benchmark across registered datasets, stream scores online, rank root causes, audit fairness, and optionally explain anomalies with an LLM.

> **Package name:** The installable Python package remains `anomaly_detection` (`pip install anomaly-detection`). **Py-outlier** is the project brand; imports use `import anomaly_detection`.

## Features

| Area | Capabilities |
| --- | --- |
| **Detectors** | z-score, IQR, Isolation Forest, LOF, One-Class SVM, autoencoder, diffusion reconstruction |
| **CLI** | `detect`, `benchmark`, `stream` |
| **REST API** | Tabular detection, batch CSV upload, RCA, LLM explain, optional vision classification |
| **Data** | Registry-driven loaders (OpenML, CSV, UCF fixtures) |
| **Streaming** | Online z-score window; optional PySAD wrappers |
| **RCA** | Causal graph scoring with ranked root causes |
| **Vision** | UCF 14-class image/video classification (supervised, separate from `/detect`) |
| **Fairness** | Demographic parity, equalized odds, reweighing mitigation (AIF360) |
| **LLM** | Opt-in anomaly explanations with PII redaction |
| **Multimodal** | Experimental tabular+text fusion |

See [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) for the phased roadmap and [CHANGELOG.md](CHANGELOG.md) for release history.

## Quick Start

```bash
git clone https://github.com/askmy-stack/Py-Outlier.git
cd Py-Outlier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
detect --config configs/default.yaml
benchmark --quick
serve    # REST API on http://localhost:8000
```

## Open Source Contributions

We welcome contributions across detectors, data, APIs, vision, streaming, fairness, docs, and evaluation. Pick an area below and open a draft PR — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

| Area | How to contribute | Module path | Skills |
| --- | --- | --- | --- |
| **Detectors** | Add a new algorithm or sklearn wrapper; register in the factory | `src/anomaly_detection/models/` | Python, ML |
| **Data loaders** | Add registry entries and loader adapters for new datasets | `src/anomaly_detection/data_ingestion/` | Python, pandas |
| **API & CLI** | Extend FastAPI routes or CLI flags (`detect`, `benchmark`, `stream`) | `src/anomaly_detection/api/`, `src/anomaly_detection/cli/` | Python, FastAPI |
| **Vision** | Improve UCF module, Grad-CAM, or TensorFlow model paths | `src/anomaly_detection/domains/vision/` | Python, CV |
| **Streaming** | Wrap PySAD detectors or extend the online z-score window | `src/anomaly_detection/streaming/` | Python, time-series |
| **RCA** | Improve causal graph scoring and metric ranking | `src/anomaly_detection/rca/` | Python, statistics |
| **Fairness & ethics** | Extend AIF360 metrics and bias mitigation | `src/anomaly_detection/fairness/` | ML fairness |
| **LLM** | Tune explainer prompts and PII redaction rules | `src/anomaly_detection/llm/` | Python, LLM APIs |
| **Docs & tutorials** | Add or update guides under `docs/tutorials/` | `docs/tutorials/` | Markdown |
| **Evaluation** | Extend benchmark harness, profiler, and metrics | `src/anomaly_detection/evaluation/` | Python, pytest |

Also see [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) for phased priorities and [docs/tutorials/](docs/tutorials/) for domain walkthroughs.

## Project structure

```
Py-Outlier/
├── src/anomaly_detection/     # installable package (import path unchanged)
│   ├── models/                # tabular & deep detectors
│   ├── api/                   # FastAPI routes
│   ├── cli/                   # detect, benchmark, stream
│   ├── data_ingestion/        # registry + loaders
│   ├── domains/vision/        # UCF classification (optional [vision] extra)
│   ├── streaming/             # online detectors
│   ├── rca/                   # root cause analysis
│   ├── fairness/              # bias metrics & mitigation
│   ├── llm/                   # anomaly explainer
│   └── multimodal/            # fusion (experimental)
├── configs/                   # YAML configuration
├── datasets/registry.yaml     # dataset metadata & licenses
├── docs/
│   ├── EXECUTION_PLAN.md      # phased roadmap
│   └── tutorials/             # step-by-step domain guides
├── tests/                     # pytest suite
├── examples/notebooks/        # original vision notebooks
└── assets/                    # README demo GIF and media
```

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/askmy-stack/Py-Outlier.git
cd Py-Outlier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Optional extras

| Extra | Install command | Enables |
| --- | --- | --- |
| `streaming` | `pip install -e ".[dev,streaming]"` | PySAD streaming detectors |
| `rca` | `pip install -e ".[dev,rca]"` | Root cause analysis (PyRCA) |
| `vision` | `pip install -e ".[dev,vision]"` | TensorFlow + OpenCV UCF endpoints |
| `fairness` | `pip install -e ".[dev,fairness]"` | AIF360 fairness metrics |
| `generative` | `pip install -e ".[dev,generative]"` | Diffusion reconstruction detector |
| `llm` | `pip install -e ".[dev,llm]"` | Anthropic LLM explainer |

Verify the install:

```bash
ruff check src tests
pytest tests/ -q
detect --config configs/default.yaml
benchmark --quick
serve    # REST API on http://localhost:8000
```

## CLI

| Command | Description |
| --- | --- |
| `detect --config CONFIG` | Run detection; writes JSON report and optional plot |
| `benchmark --quick` | Benchmark all registry datasets × detectors on fixtures |
| `benchmark --quick --profile` | Same, with wall-time and peak-memory profiling |
| `stream --config CONFIG` | Online streaming detection |

Example:

```bash
python -m anomaly_detection.cli.detect --config configs/default.yaml
python -m anomaly_detection.cli.benchmark --quick --profile
```

## REST API

Start the server with `serve` (or `uvicorn anomaly_detection.api.app:app`). Interactive docs at `/docs`.

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Liveness check |
| `/detect` | POST | Detect anomalies from a 2D numeric array + optional config override |
| `/detect/batch` | POST | Upload a CSV file for batch detection |
| `/models` | GET | List registered detector names |
| `/root_cause` | POST | Rank root causes for an anomaly given multivariate metrics |
| `/root_cause/{anomaly_id}` | GET | Retrieve a cached RCA result |
| `/explain` | POST | Generate plain-language anomaly explanation (LLM opt-in) |
| `/vision/analyze/image` | POST | Classify an image into 14 UCF crime categories (`[vision]` extra) |
| `/vision/analyze/video` | POST | Classify a video via frame sampling (`[vision]` extra) |

**Note:** Vision endpoints perform **supervised multi-class classification**, not unsupervised anomaly detection. They are intentionally separate from `/detect`.

### Authentication

All routes above except `/health`, `/docs`, `/redoc`, and `/openapi.json` require
an `X-API-Key` header matching `ANOMALY_API_KEY` when that env var is set. Leave
it unset for local/dev use (no auth required).

## Benchmark results (quick mode, fixtures)

Run `benchmark --quick` locally to reproduce. Representative results on test fixtures:

| Dataset | Detector | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| credit_card_fraud | isolation_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| nab | isolation_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ucf_crime | lof | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Profiling (isolation_forest on credit_card_fraud fixture): ~0.12 s wall time, ~194 MB peak memory on Apple Silicon (Python 3.13).

## Datasets

Registered in [datasets/registry.yaml](datasets/registry.yaml):

| ID | Domain | License |
| --- | --- | --- |
| `credit_card_fraud` | tabular | CC-BY-4.0 |
| `nab` | timeseries | AGPL-3.0 |
| `ucf_crime` | vision | Custom (academic use) |

## Tutorials

| Tutorial | Topic |
| --- | --- |
| [01-tabular-fraud](docs/tutorials/01-tabular-fraud.md) | Credit-card fraud (`configs/examples/fraud.yaml`) |
| [02-timeseries-iot](docs/tutorials/02-timeseries-iot.md) | NAB time-series and IoT streaming |
| [03-vision-surveillance](docs/tutorials/03-vision-surveillance.md) | UCF vision classification (supervised) |
| [04-streaming](docs/tutorials/04-streaming.md) | Online `stream` CLI |
| [05-fairness](docs/tutorials/05-fairness.md) | Fairness metrics and mitigation |

## Vision domain (legacy notebooks)

Pre-trained SavedModel artifacts remain at the repository root:

- `Image Anomaly Detection-2/` — image classifier
- `Video Anomaly Detection/` — video classifier

Configure paths in `configs/examples/vision.yaml`. To explore the original notebooks:

```bash
jupyter notebook examples/notebooks/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please read [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) before starting substantial work.

## License

MIT — see [LICENSE](LICENSE).

---

**Py-outlier** by [Abhinaysai Kamineni](https://github.com/askmy-stack) · [GitHub](https://github.com/askmy-stack/Py-Outlier)
