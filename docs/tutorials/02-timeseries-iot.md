# Tutorial 02: Time-Series IoT (NAB)

Work with Numenta Anomaly Benchmark (NAB) time-series data and streaming
windowed detectors for IoT-style sensor monitoring.

## Prerequisites

- Python 3.11+ with the package installed: `pip install -e ".[dev]"`
- Optional streaming extras: `pip install -e ".[streaming]"`
- NAB registry entry: `datasets/registry.yaml` (`id: nab`)
- Streaming config: [`configs/examples/streaming.yaml`](../../configs/examples/streaming.yaml)

## Benchmark NAB (batch)

Run a quick benchmark on the NAB fixture (no download required):

```bash
benchmark --datasets nab --detectors zscore iqr --quick
```

Profile wall time and memory:

```bash
benchmark --datasets nab --detectors zscore --quick --profile
```

## Run streaming detection

Process the sample CSV in a sliding window (IoT-style online scoring):

```bash
stream --config configs/examples/streaming.yaml
```

Pipe synthetic points on stdin:

```bash
echo "1.0,2.0,3.0" | stream --config configs/examples/streaming.yaml
```

## Expected output

**Benchmark:** Markdown report under `reports/benchmark_<timestamp>.md` with
dataset `nab`, detector metrics, and optional profiling table.

**Stream:** One JSON object per row on stdout:

```json
{"score": 0.42}
```

Final summary:

```json
{"summary": {"n_points": 200, "mean_latency_ms": ..., "max_score": ...}}
```

## Configuration highlights

| File | Purpose |
| --- | --- |
| `datasets/registry.yaml` | Registers `nab` as `domain: timeseries` |
| `configs/examples/streaming.yaml` | `zscore_window` detector, CSV input at `data/sample.csv` |

Streaming settings:

- `stream.max_window` / `min_window` — bounded memory for IoT deployments
- `model.name: zscore_window` — online z-score over a rolling window
- `input.path: data/sample.csv` — local demo sensor features

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `NAB fixture not found` | Ensure `tests/fixtures/nab_sample.csv` exists |
| Stream `CSV input requires` error | Pass `--input data/sample.csv` or set `input.path` in config |
| `pysad` import error | Install streaming extra: `pip install -e ".[streaming]"` |
| Window memory error | Lower `stream.max_window` in `streaming.yaml` |

## Next steps

- Deep-dive streaming: [04-streaming.md](04-streaming.md)
- Tabular baseline: [01-tabular-fraud.md](01-tabular-fraud.md)
