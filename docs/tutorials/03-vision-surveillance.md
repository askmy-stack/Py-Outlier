# Tutorial 03: Vision Surveillance (UCF)

Use the UCF crime vision module for image and video classification. **This is
supervised multi-class classification, not unsupervised anomaly detection.**

## Important caveat

The vision endpoints and SavedModels classify frames into **14 crime categories**
(e.g. Abuse, Arrest, Fighting). They do **not** produce unsupervised anomaly
scores. Do not conflate vision `/vision/analyze/*` results with tabular
`/detect` anomaly detection.

## Prerequisites

- Python 3.11+ with vision extras: `pip install -e ".[dev,vision]"`
- TensorFlow-compatible platform (macOS/Linux; GPU optional)
- Pre-trained SavedModels at repository root (not shipped in git by default):
  - `Image Anomaly Detection-2/`
  - `Video Anomaly Detection/`
- Config: [`configs/examples/vision.yaml`](../../configs/examples/vision.yaml)

## Start the API server

```bash
serve
```

The server listens on `http://127.0.0.1:8000` by default.

## Analyze an image

```bash
curl -X POST "http://127.0.0.1:8000/vision/analyze/image" \
  -F "file=@path/to/frame.jpg"
```

## Analyze a video

```bash
curl -X POST "http://127.0.0.1:8000/vision/analyze/video" \
  -F "file=@path/to/clip.mp4"
```

## Expected output

JSON response with top predicted class and confidence, for example:

```json
{
  "predictions": [
    {"label": "Normal", "confidence": 0.87}
  ],
  "model": "image",
  "n_frames": 1
}
```

Video responses include frame-level sampling (`max_frames` from config).

## Configuration highlights

[`configs/examples/vision.yaml`](../../configs/examples/vision.yaml):

| Key | Default |
| --- | --- |
| `vision.image_model_path` | `Image Anomaly Detection-2` |
| `vision.video_model_path` | `Video Anomaly Detection` |
| `vision.max_frames` | `16` |
| `vision.max_upload_mb` | `50` |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `503 Vision module not available` | Install vision extra: `pip install -e ".[vision]"` |
| Model path not found | Download or place SavedModels; update paths in `vision.yaml` |
| Large upload rejected | Reduce file size or raise `max_upload_mb` |
| Slow video inference | Lower `max_frames` for faster sampling |

## Explore notebooks

Original training notebooks live under `examples/notebooks/`:

```bash
jupyter notebook examples/notebooks/
```

## Next steps

- Tabular anomaly detection: [01-tabular-fraud.md](01-tabular-fraud.md)
- Roadmap context: [docs/EXECUTION_PLAN.md](../EXECUTION_PLAN.md) Phase 7
