# Tutorial 04: Streaming Anomaly Detection

Run online anomaly scoring over CSV files or stdin using the `stream` CLI and
[`configs/examples/streaming.yaml`](../../configs/examples/streaming.yaml).

## Prerequisites

- Python 3.11+ with streaming support: `pip install -e ".[dev,streaming]"`
- Sample data: `data/sample.csv` (included in the repo)
- Config: [`configs/examples/streaming.yaml`](../../configs/examples/streaming.yaml)

## Stream from configured CSV

```bash
stream --config configs/examples/streaming.yaml
```

Override input path:

```bash
stream --config configs/examples/streaming.yaml --input data/sample.csv
```

## Stream from stdin

```bash
head -5 data/sample.csv | tail -n +2 | cut -d, -f1-3 | stream --config configs/examples/streaming.yaml
```

Each line should be comma-separated feature values matching `input.columns`.

## Expected output

Per-point scores (newline-delimited JSON):

```json
{"score": 1.23}
{"score": 0.98}
```

Final summary on the last line:

```json
{"summary": {"n_points": 200, "mean_latency_ms": 0.05, "max_score": 4.2}}
```

With `--verbose`, latency and memory tracking details appear in logs when
`monitoring.track_memory: true`.

## Configuration highlights

[`configs/examples/streaming.yaml`](../../configs/examples/streaming.yaml):

```yaml
stream:
  max_window: 100
  min_window: 5
model:
  name: zscore_window
input:
  source: csv
  path: data/sample.csv
  columns: [feature_1, feature_2, feature_3]
```

- `max_window` enforces a memory bound (raises if exceeded)
- `alerts.enabled` can notify Slack when scores exceed `threshold`

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Unsupported input source` | Set `input.source` to `csv` or `stdin` |
| Column mismatch | Align `input.columns` with CSV headers |
| `memory bound violated` | Increase `stream.max_window` or use a smaller window detector |
| Missing `pysad` | `pip install -e ".[streaming]"` |

## Next steps

- NAB time-series context: [02-timeseries-iot.md](02-timeseries-iot.md)
- Profile batch detectors: `benchmark --quick --profile`
