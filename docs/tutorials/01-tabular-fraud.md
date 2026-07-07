# Tutorial 01: Tabular Fraud Detection

Detect credit-card fraud on tabular transaction data using the registry-backed
`credit_card_fraud` dataset and the `detect` CLI.

## Prerequisites

- Python 3.11+ with the package installed: `pip install -e ".[dev]"`
- Network access (first run downloads the OpenML creditcard dataset)
- Config: [`configs/examples/fraud.yaml`](../../configs/examples/fraud.yaml)

## Run detection

From the repository root:

```bash
detect --config configs/examples/fraud.yaml
```

With verbose logging:

```bash
detect --config configs/examples/fraud.yaml --verbose
```

## Expected output

The CLI prints a JSON summary on stdout:

```json
{"report_path": "reports/fraud_run.json", "n_anomalies": <integer>}
```

The report file includes:

- `model`: `isolation_forest`
- `n_samples`: number of transactions processed
- `n_anomalies`: flagged anomalies (contamination ~5%)
- `scores` and `predictions` arrays
- `metrics` with precision, recall, F1, and ROC-AUC when labels are available

Inspect the report:

```bash
python -c "import json; print(json.dumps(json.load(open('reports/fraud_run.json')), indent=2)[:500])"
```

## Configuration highlights

The example config wires:

| Section | Setting |
| --- | --- |
| `dataset.id` | `credit_card_fraud` (OpenML creditcard) |
| `dataset.target_column` | `Class` |
| `model.name` | `isolation_forest` |
| `output.report_path` | `reports/fraud_run.json` |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| OpenML download timeout | Retry with stable network; dataset is cached after first fetch |
| `Unknown dataset` | Ensure `datasets/registry.yaml` lists `credit_card_fraud` |
| High memory usage | Use `benchmark --quick --datasets credit_card_fraud` for fixture-sized data |
| Import errors | Reinstall: `pip install -e ".[dev]"` |

## Next steps

- Compare detectors: `benchmark --datasets credit_card_fraud --detectors zscore isolation_forest --quick`
- Fairness-aware runs: [05-fairness.md](05-fairness.md)
