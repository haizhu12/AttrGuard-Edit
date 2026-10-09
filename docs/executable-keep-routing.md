# Experimental executable keep routing

This is an opt-in local extension to the existing source-conditioned inference command. It changes inference, not weights or training. Default mode remains `off` pending quality validation.

## Select a mode

Add `--preservation_keep_mode project` or `--preservation_keep_mode source-memory` to the command in [README](../README.md#4-run-the-current-method). Keep the source image, source-bound permissions, adapter, seed, reference seed, CFG and step count identical when comparing modes. Use a fresh output name each time.

| Mode | Behavior | Additional DiT forward per step |
| --- | --- | --- |
| `off` | Existing source-conditioned inference | 0 |
| `project` | Retain fully protected latent cells along a source noising path | 0 |
| `source-memory` | Projection plus protected source features at each native double/single-stream block input | 1 when any protected patch exists |

`source-memory` requires `--text_guidance_scale 1` and the source-conditioned default empty negative instruction. Both enabled modes require inference without gradients. Legacy `--method baseline` rejects non-off modes. Existing cache/offload exclusions still apply. An empty eligible mask is an exact no-op, recorded as `effective_mode: off`.

No checkpoint conversion or training is necessary to try this switch. Keep your adapter selection explicit. The source-feature forward bypasses the adapter; the edited forward uses your existing adapter normally. Source features use the cached empty-instruction branch without CFG extrapolation. Therefore CFG=1 does not mean that source-memory performs only one DiT forward in total.

## What is protected

Permissions retain their existing five-attribute schema. A native patch is eligible only if its pooled pixel roles are exactly K=1, M=0, U=0, and every attribute has M=0. Unavailable source references and partially kept cells are not silently promoted to full protection. Structure/texture/glyph/color keep alone never implies a pixel lock.

The implementation follows the native patch order and does not resize masks. The source path is `(1-t) * initial_noise + t * source_latent`, with Boogu time `t=0` at noise and `t=1` at the source image. Projection occurs before denoising and after each actual Euler step. Source-memory replaces only eligible generated-token block inputs, leaving the text, native reference and free generated positions untouched at that replacement operation. Attention can still propagate their influence to the editable region.

The source path is an analytic intervention, **not exact inversion**. Block-input reuse is **not direct K/V caching**, nor an implementation of the KV-Edit paper. All tokens are still computed; this experiment makes no inference-speed improvement claim.

## Evaluate both editing and preservation

The output JSON records `preservation_keep_mode` and `keep_routing`, including eligible token coverage, effective mode, source-forward count per step, source-path convention and feature-routing strategy.

Do not judge success solely by protected-region pixel error. Verify that the requested edit occurred, important non-target objects and their relations remain, and no new seam, ghost or original-color residue appears. Completely unchanged objects and invariant attributes within an edited object are different problems: this first stage only provides executable pixel-region retention. It does not implement attribute-selective training or guarantee object identity.

The FP32 VAE still decodes the complete final latent. Preserved latent cells do not imply exact original RGB pixels, especially near mixed boundaries. Wrong keep permissions can suppress valid changes or retain unwanted content. Source features can also leak information about an object requested for removal.

## Local controlled diagnostic

The fixed protocol is `outputs/keep_routing_20261007/protocol.json` in the research workspace; datasets, weights, caches and experimental outputs are not part of the distributed package. It compares all three modes with the same frozen MagicBrush pilot adapter, Euler50 and CFG1. It uses four real photographs: motorcycle replacement at two seeds, ball removal, frisbee replacement, and dog recoloring plus a no-edit control, for 18 generations.

These are already inspected diagnostic examples, not an untouched generalization benchmark. The prior seed42 MagicBrush off-mode latent must replay exactly before modified results are interpreted. The dog runs use the same pilot adapter as the other cases, not the earlier garment adapter, so older dog outputs are not paired comparators. There is no paired target for the dog recolor task.

The fixed run completed at `outputs/keep_routing_20261007/run_v1`; the controller's image review and decision are in `outputs/keep_routing_20261007/RESULTS.md`. The immutable machine report still says visual review pending because that report was sealed before the separate visual assessment. Test-suite success alone does not establish preservation quality.

### Diagnostic outcome, October 7, 2026

All 18 generations finished, and the three historical seed42 off-mode endpoints replayed exactly. Inputs, code and weights were unchanged during the run. The controller independently rechecked 116 saved payload checksums and recomputed the reported outside/core metrics from the float32 RGB tensors.

Both enabled modes reduced protected-outside error. On the motorcycle seed42 failure, MAE fell from 0.126854 to approximately 0.01350, and the surfboard remained visible. Project-only already achieved this improvement. However, both constrained outputs had conspicuous vehicle/leg geometry artifacts; source-memory at seed123 retained the bicycle instead of completing the replacement. In ball removal, source-memory retained a clear ball. The frisbee case still had a detached ball, and dog recoloring still redrew internal fur/face details despite a more stable background.

**Neither mode passed the joint edit/preservation gate; default stays `off`.** Keep `project` as a diagnostic spatial-retention control and `source-memory` as an unpromoted experimental branch. The source-memory branch took approximately twice the measured denoising time without a consistent joint quality benefit. These fixed cases do not establish a general failure of all source-feature conditioning, nor justify a full architecture/weight replacement. They do show that this particular all-layer, all-step intervention should not be treated as a validated general solution.

The MagicBrush permissions in this run come from existing annotation alpha masks, not target RGB differences. Their use does not validate automatic permission prediction. Dog permissions were also reused without changes. No new masks were selected after viewing results.
