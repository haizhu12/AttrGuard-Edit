# Changelog

## 0.2.2 — 2026-10-09

Code-only research archive before manuscript preparation. No new training, weight replacement or change to repository visibility.

- Archive opt-in `preservation_keep_mode=project|source-memory`, bounded frozen-weight `preservation_optimization=latent`, and `preservation_edit_anchor=rgb|attribute`.
- Preserve Full source-conditioned generation, paired training, CFG=1, permission schema and existing checkpoint compatibility. Optimization/keep routing remain off; the edit anchor remains RGB by default.
- Add self-contained keep-routing, latent-optimization and edit-anchor regression tests and portable usage guides.
- Retain explicit limitations: no general-purpose adapter or automatic permission predictor is shipped; internal object structure/identity and non-target semantic preservation are not established.
- Exclude models, adapters, datasets, source photographs, generated experiment outputs, private evaluation scripts, caches and machine-local provenance. Existing historical demonstration media stays historical.

The October 9 diagnostic resolved part of an RGB-anchor acceptance conflict, but the visible dog fur/face differences remained. Numerical proxy gains are not a general editing-quality claim.

## 0.2.1 — 2026-10-06

Integrate Full source-conditioned inference, explicit checkpoint export and portable paired training. Set CFG=1 and skip the negative DiT branch at that setting. No retraining or new generalization claim.
