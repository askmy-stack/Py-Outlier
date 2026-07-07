# Tutorial 05: Fairness-Aware Detection

Run anomaly detection with fairness metrics and optional mitigation on
protected attributes using
[`configs/examples/fairness.yaml`](../../configs/examples/fairness.yaml).

## Prerequisites

- Python 3.11+ with fairness support: `pip install -e ".[dev,fairness]"`
- Fixture dataset: `tests/fixtures/fairness_sample.csv`
- Config: [`configs/examples/fairness.yaml`](../../configs/examples/fairness.yaml)

## Run fairness-aware detection

```bash
detect --config configs/examples/fairness.yaml
```

Verbose output:

```bash
detect --config configs/examples/fairness.yaml --verbose
```

## Benchmark with fairness metrics

```bash
benchmark --datasets credit_card_fraud --detectors isolation_forest --quick \
  --config configs/examples/fairness.yaml
```

Note: the fairness example config uses a local CSV path; for registry datasets
ensure protected attribute columns exist in the data.

## Expected output

CLI summary:

```json
{"report_path": "reports/fairness_run.json", "n_anomalies": <integer>}
```

Report includes `fairness_metrics` when protected attributes are configured:

```json
{
  "fairness_metrics": {
    "attributes": {
      "gender": {
        "demographic_parity_difference": 0.12,
        "equalized_odds_difference": 0.08
      }
    }
  }
}
```

Benchmark markdown adds a **Fairness metrics** table when applicable.

## Configuration highlights

[`configs/examples/fairness.yaml`](../../configs/examples/fairness.yaml):

| Section | Setting |
| --- | --- |
| `dataset.path` | `tests/fixtures/fairness_sample.csv` |
| `fairness.protected_attributes` | `gender`, `age_group` |
| `fairness.mitigation` | `reweighing` |
| `model.name` | `isolation_forest` |

Protected attributes must be **explicitly listed**; they are never inferred.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `aif360` import error | Install fairness extra: `pip install -e ".[fairness]"` |
| Protected attributes missing | Add columns to CSV or adjust `protected_attributes` list |
| Empty fairness metrics | Confirm `fairness` block is present and attributes exist in data |
| Mitigation warnings | Try `mitigation: null` to run metrics-only |

## Ethics

See [docs/ETHICS.md](../ETHICS.md) for responsible use of fairness tooling.

## Next steps

- Tabular fraud baseline: [01-tabular-fraud.md](01-tabular-fraud.md)
- Full roadmap: [docs/EXECUTION_PLAN.md](../EXECUTION_PLAN.md) Phase 8
