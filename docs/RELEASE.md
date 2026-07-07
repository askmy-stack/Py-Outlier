# Release process

This document describes how to cut a release after the stacked phase PRs are merged.

## Prerequisites

1. All phase PRs (#3 → #5 → #7 → #9 → #11 → #13 → #15 → #17 → #19 → #21 → this PR) are merged into `main`.
2. CI is green on `main`.
3. Local verification passes:

   ```bash
   ruff check src tests
   pytest tests/ -v --cov=anomaly_detection
   python -m anomaly_detection.cli.detect --config configs/default.yaml
   python -m anomaly_detection.cli.benchmark --quick
   python -m anomaly_detection.cli.benchmark --quick --profile
   ```

## Version tagging

Releases follow [Semantic Versioning](https://semver.org/):

| Tag | Milestone |
| --- | --- |
| `v0.1.0` | Phase 3 — minimal viable framework (CLI + API) |
| `v0.5.0` | Phase 7 — vision + tabular domains |
| `v1.0.0` | Phase 11 — full framework (all phases complete) |

### Create the v1.0.0 tag (after merge)

```bash
git checkout main
git pull origin main

# Confirm version in pyproject.toml and src/anomaly_detection/api/app.py is 1.0.0
grep version pyproject.toml

git tag -a v1.0.0 -m "v1.0.0 — community-ready anomaly detection framework"
git push origin v1.0.0
```

### Create a GitHub release

```bash
gh release create v1.0.0 \
  --title "v1.0.0" \
  --notes-file CHANGELOG.md
```

## Post-release

- Update `CHANGELOG.md` with the release date if it differs from the draft.
- Open issues for any follow-up work discovered during verification.
