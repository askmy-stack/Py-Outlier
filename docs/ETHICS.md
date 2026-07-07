# Fairness, Privacy, and Responsible Use

This project provides tools to **detect** unusual patterns in data. Those patterns are
not always harmful, erroneous, or undesirable. Use human judgment and domain expertise
before treating flagged records as problems.

## Fairness and bias

Anomaly detectors can disproportionately flag members of protected groups when:

- Training or reference data under-represents some populations
- Features correlate with sensitive attributes
- Thresholds are tuned on homogeneous data

The optional fairness module (`pip install -e ".[fairness]"`) measures disparity using
metrics such as **demographic parity difference** and **equalized odds difference**.
Protected attributes must be listed explicitly in configuration; the framework never
infers sensitive columns automatically.

Fairness metrics describe model behavior on a dataset—they do not guarantee equitable
outcomes in production. Mitigation helpers (reweighing, threshold post-processing) are
starting points for experimentation, not certified compliance controls.

## Privacy

- Avoid logging raw sensitive attributes in shared reports unless policy allows it.
- Prefer aggregated fairness summaries over per-record protected-attribute exports.
- Review data retention for anomaly scores and explanations stored alongside identifiers.

## Responsible use

- **Anomalies are not always negative.** A spike may reflect legitimate growth, seasonality,
  or a newly launched product—not fraud or abuse.
- **Ground truth may be incomplete.** Rare events and label noise affect evaluation and
  fairness metrics.
- **Human review.** Automated flags should support investigation workflows, not replace them.
- **Context matters.** Domain-specific ethics (healthcare, finance, public safety) may impose
  requirements beyond this library.

When deploying detectors that affect people, document intended use, known limitations, and
escalation paths for disputed outcomes.
