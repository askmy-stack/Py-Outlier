# Anonmaly Detection

An anonymous, AI-powered tip-off platform for reporting suspicious activity. Users submit images or video; a DenseNet121 model classifies the anomaly type, Grad-CAM explains the AI's reasoning, and Claude synthesizes a structured incident report — all without collecting any user identity.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Next.js 14)                                           │
│  Landing · Form · About                                         │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP (rewrites to :8000)
┌────────────────────▼────────────────────────────────────────────┐
│  FastAPI  (uvicorn)                                             │
│  POST /analyze/image  →  DenseNet121 + Grad-CAM                 │
│  POST /analyze/video  →  enqueue Celery task                    │
│  GET  /analyze/job/:id→  poll Celery result                     │
└─────────┬───────────────────────────┬───────────────────────────┘
          │                           │
┌─────────▼──────────┐   ┌───────────▼───────────────────────────┐
│  Celery Worker     │   │  Claude API (claude-sonnet-4-6)        │
│  Video frame       │   │  Incident report synthesis             │
│  extraction +      │   │  from tip text + ML classification     │
│  inference         │   └───────────────────────────────────────┘
└─────────┬──────────┘
          │
┌─────────▼──────────┐   ┌───────────────────────────────────────┐
│  Redis             │   │  MLflow Tracking Server (:5000)        │
│  Task broker +     │   │  Experiment metrics, model registry    │
│  result backend    │   └───────────────────────────────────────┘
└────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, React Hook Form + Zod |
| Backend API | FastAPI, Python 3.11, Uvicorn |
| ML Inference | TensorFlow 2.16, DenseNet121, Grad-CAM |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Task Queue | Celery + Redis |
| Experiment Tracking | MLflow |
| Observability | structlog (JSON), Sentry |
| CI/CD | GitHub Actions (lint → test → model quality gate → Docker build → deploy) |
| Containers | Docker + Docker Compose |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- An Anthropic API key (optional — enables incident report synthesis)

```bash
git clone https://github.com/askmy-stack/Anonmaly-Detection
cd Anonmaly-Detection

cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY=sk-ant-...

docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| API health | http://localhost:8000/health |

---

## Local Development (without Docker)

### Backend

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
export MODEL_IMAGE_PATH="Image Anomaly Detection-2"
export MODEL_VIDEO_PATH="Video Anomaly Detection"

cd backend
uvicorn app.main:app --reload --port 8000

# Celery worker (separate terminal)
celery -A app.workers.tasks worker --loglevel=info -Q video
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

### Tests

```bash
# Backend
cd backend && pytest tests/ -v --cov=app

# Frontend
cd frontend && npm test -- --run
```

---

## ML Models

### Model 2 — DenseNet121 (Production, `Image Anomaly Detection-2/`)
- **8,095,054 parameters** (7.9M trainable)
- Input: 64×64×3 · Output: 14-class softmax
- Backbone: DenseNet121 (ImageNet weights, frozen) + Dense (256→1024→512→14)
- Trained on UCF Crime Dataset (1.26M images, 14 classes)

### Model 3 — Video Frame Classifier (`Video Anomaly Detection/`)
- Same architecture as Model 2, applied per-frame with averaged predictions
- Up to 16 evenly-sampled frames per video

### Training

```bash
python scripts/train.py \
    --data-dir /path/to/ucf-crime \
    --output-dir ./trained_models/run_001 \
    --epochs 50 --batch-size 64 \
    --mlflow-uri http://localhost:5000
```

### CI Quality Gate

```bash
python scripts/evaluate.py --threshold 0.70
# Exits non-zero → blocks merge if accuracy < 70%
```

---

## API Reference

### `POST /analyze/image`

```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "file=@photo.jpg" \
  -F "message=Two people fighting near the alley at night"
```

```json
{
  "submission_id": "uuid",
  "media_type": "image",
  "classification": {
    "predicted_class": "Fighting",
    "confidence": 0.91,
    "all_scores": { "Fighting": 0.91, "Normal": 0.04 },
    "heatmap_url": "data:image/png;base64,..."
  },
  "incident_report": {
    "summary": "Two individuals in physical altercation near alley.",
    "threat_level": 4,
    "location_indicators": ["alley"],
    "time_references": ["night"],
    "actor_count": "2",
    "ml_consistency": "Consistent — model detected Fighting.",
    "recommendation": "Dispatch patrol unit."
  },
  "processing_time_ms": 312.5
}
```

### `POST /analyze/video` → `GET /analyze/job/{id}`

```bash
curl -X POST http://localhost:8000/analyze/video -F "file=@clip.mp4"
# { "job_id": "uuid", "status": "queued" }

curl http://localhost:8000/analyze/job/{job_id}
# { "status": "complete", "result": { ... } }
```

---

## CI/CD Pipeline

```
push/PR → lint + type-check (backend + frontend)
       → unit tests (pytest + vitest)
       → model quality gate (accuracy ≥ 70%)
       → docker build (backend + frontend)
       → deploy to staging (main branch only)
```

---

## Project Structure

```
Anonmaly-Detection/
├── backend/app/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── core/{config,logging}.py
│   ├── ml/{model,gradcam,preprocess}.py
│   ├── llm/synthesizer.py   # Claude API integration
│   ├── workers/tasks.py     # Celery async video task
│   ├── api/{analyze,health}.py
│   └── models/schemas.py    # Pydantic schemas
├── frontend/
│   ├── app/{page,form,about}/
│   └── components/{AnalysisResultCard,Spinner}.tsx
├── scripts/{train,evaluate}.py
├── .github/workflows/ci.yml
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Security

- No PII collected at any point
- MIME type + size validation on all uploads
- `ANTHROPIC_API_KEY` and secrets loaded from environment — never hardcoded
- reCAPTCHA site key is intentionally public (client-side only by design)