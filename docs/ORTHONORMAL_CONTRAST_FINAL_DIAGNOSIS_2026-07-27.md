```text
CM_FEATURE_IMMUTABILITY = pass
CM_BASIS_DEFAULT = cumulative
ORIGIN_CONTRAST_DEFAULT = inconclusive
INTERVAL_HESSIAN = correct_not_faster
EQUIVALENCE_VERIFIED = rank/inverse-transform/zero-sets/Delta*: PASS at D4 (both contrasts, all L)
  and D20 (both contrasts, L=50) -- see INTERVAL_VS_CUMULATIVE_FINAL_DIAGNOSIS Section 2-3 for the
  shared evidence (the transform/rank/zero-set checks are basis-focused but were run at both
  contrasts throughout, so they double as the contrast-equivalence check) | Hessian congruence:
  PASS analytically -- the production Hessian assembly code's own comments state the exact identity
  `H_final[block] = R' * H_raw[block] * R` (cm_hessian_architectures.jl:326-327,
  cm_hessian_architecture_interval.jl:140), i.e. anchored and orthonormal Hessians are related by an
  EXACT orthogonal congruence transform BY CONSTRUCTION, not merely observed to be numerically close
HIGHEST_PRIORITY_REMAINING_GAP = the actual wired production default (`run_cm_upper_checkpointed`,
  cm_checkpoint.jl:591, `contrasts::Symbol = :anchored`) contradicts three 2026-07-26 documents that
  claim orthonormal is "already the production default" -- see Section 2. This session did not
  change the code (no D20 evidence argues for paying any cost to change it, see Section 4), but the
  documentation/actual-default mismatch should be resolved explicitly, not left ambiguous.
```

# Anchored vs. Orthonormal Origin Contrasts — Final Diagnosis — 2026-07-27

Phase C, item 14 (contrast axis) + item 16's decision rule. **Headline finding: the D4-only
"orthonormal strictly dominates anchored, 2.6x-3.4x better conditioned" claim that three separate
2026-07-26 documents cite as an already-settled, already-production decision does NOT hold at real
D=20 — at D20 the two contrast schemes are conditioning-equivalent to within ~0.1%-1.7%, and
anchored is fractionally (not meaningfully) BETTER, not worse, at every point this session checked.**
This is the same D4-vs-D20 reversal pattern this exact codebase already knows well from the
cumulative-vs-interval basis axis (item 13) — and per CLAUDE.md's own standing feedback about this
project's history of exactly this trap (`feedback-verify-before-causal-claims`, `feedback-gravity-
elimination-zero-is-not-calibration`), it was worth checking rather than assuming the D4 claim
transfers. It did not transfer.

## 1. The D4 claim (2026-07-22 review, reconfirmed by this session at D4)

`docs/fullA_cm_conditioning_and_adaptive_grid_report.md` (D=4, W=8000): mean `cond(Hessian))` over 4
reference countries ranks, from best to worst, **interval < std_interval < orthonormal < anchored**
at every L in {10,20,50} tested, with orthonormal beating anchored by "2.6-3.4x" (that document's own
words) and the gap *widening* with L. This session's own fresh D4 run
(`c15_d4_four_arm_basis_contrast_comparison.jl`, `docs/key_results/four_arm_basis_contrast_summary_
2026-07-27.csv`) reconfirms this at calibration:

| L | basis | cond(anchored) | cond(orthonormal) | ratio (ortho/anchored) |
|---|---|---|---|---|
| 10 | cumulative | 3.6396e+04 | 1.6297e+04 | 0.448 (orthonormal 2.2x better) |
| 50 | cumulative | 1.4369e+05 | 4.2026e+04 | 0.293 (orthonormal 3.4x better) |
| 10 | interval | 1.2314e+04 | 1.1276e+04 | 0.916 (orthonormal marginally better) |
| 50 | interval | 1.1231e+04 | 1.1157e+04 | 0.993 (essentially tied) |

The D4/cumulative-basis numbers match the prior review's "2.6-3.4x, widening with L" characterization
closely. **This part of the prior finding is real and reconfirmed, not disputed.**

## 2. The documentation says this is already settled and already production. It is not.

Three 2026-07-26 documents state, unambiguously, that `contrasts = :orthonormal` is the current
production default:

- `docs/ORTHONORMAL_ORIGIN_CONTRAST_BENCHMARK_2026-07-26.md`: *"Contrary to the task's framing...
  `contrasts=:orthonormal` is **already** the approved, production-default contrast basis."*
- `docs/FIVE_FAMILY_OPTIMIZATION_COMPLETION_MASTER_REPORT_2026-07-26.md`: *"ORIGIN_CONTRAST_DEFAULT
  = orthonormal (pre-existing 2026-07-22 decision, reconfirmed not re-decided)."*
- `docs/FINAL_5X7_STATUS_MATRIX_2026-07-26.md`: all three restricted families listed as "cumulative +
  orthonormal (pre-existing decision, reconfirmed)."

**This session read the actual wired code on the current HEAD and found the opposite.** The real
production driver entry point is `run_cm_upper_checkpointed` (`cm_checkpoint.jl:589-591`):

```julia
function run_cm_upper_checkpointed(w0::Union{Nothing,Vector{Float64}} = nothing;
        W::Int = 80000, delta::Float64 = 1.0, draw_design::Symbol = :sobol_randomized, draw_seed::Int = 20260719,
        L::Int = 10, contrasts::Symbol = :anchored, ...)
```

`contrasts::Symbol = :anchored` — this single kwarg is threaded, unchanged, into whichever of
`build_cm_production_context` (flexible CM), `build_cm_frechet_production_context` (common Fréchet),
or `build_cm_meanzc_production_context` (CM+ZC) the `marginal_restriction`/`cm_extension` dispatch
selects (`cm_checkpoint.jl:891-895`). All three of those functions' own kwarg defaults are also
`contrasts::Symbol = :anchored` (`cm_production_bundle.jl:61`, `cm_frechet_level.jl:140,276`) — only
`cm_meanzc_production.jl:101`'s own function-level default says `:orthonormal`, but it is never
reached with that default from the actual driver, since `run_cm_upper_checkpointed` always passes
its own `contrasts` value explicitly. The unified `CMConfig` production surface (`cm_config.jl:57`)
also defaults to `contrasts::Symbol = :anchored`.

**So: every actual call path from the real production driver to a real CM inner solve defaults to
anchored, not orthonormal, on this HEAD, contradicting three documents that say the opposite.** This
is not a subtle inference — it is reading one `Base.@kwdef` default and one driver function
signature. It is exactly the "verify hot-path claims, don't trust [a document/label]" trap this
project's own CLAUDE.md flags as a recurring failure mode (`feedback-verify-hot-path-not-just-grep-
found.md`), just manifesting via a stale doc rather than a stale grep this time.

**This session did not change the code.** Per this task's own explicit instruction ("Do NOT flip
any production default... without real D=20 evidence meeting rule §16"), and given the D20 evidence
below shows no material difference either way, there is no basis this session found for preferring
either value strongly enough to justify a change — see Section 4. The discrepancy is reported here
so the next session (or the user) can make an explicit, informed choice between "the docs are
right and the code needs fixing" and "the code is right and the docs need correcting" — this session
takes neither position because neither is is strongly supported by new D20 evidence uncovered here.

## 3. New this session: real D=20/W=80,000, the check that was never run

`docs/fullA_cm_hessian_architecture_report.md` Section 10 explicitly labels the D20/L=50/W=80000
case for this exact question **"NOT run — qualitative projection only."** No document in this
codebase's history ran a real D20 anchored-vs-orthonormal conditioning comparison before this
session. `c15_d20_four_arm_basis_contrast_comparison.jl` (real D=20/W=80,000, seed 20260719,
`destination_sample=:exclude_row`, L=50, calibration + perturbed_2pct) closes that gap:

| point | basis | cond(anchored) | cond(orthonormal) | ratio (ortho/anchored) |
|---|---|---|---|---|
| calibration | cumulative | 1.1095e+06 | 1.1108e+06 | 1.0012 (anchored marginally better) |
| perturbed_2pct | cumulative | 1.4619e+06 | 1.4626e+06 | 1.0005 (anchored marginally better) |
| calibration | interval | 4.2165e+07 | 4.2314e+07 | 1.0035 (anchored marginally better) |
| perturbed_2pct | interval | 4.8028e+07 | 4.8838e+07 | 1.0169 (anchored marginally better) |

**The D4 finding does not survive to D20.** At D20, the two contrast schemes are conditioning-
equivalent to within 0.05%-1.7% — an order of magnitude smaller than run-to-run KNITRO iteration-
count noise, let alone the D4 gap (2.2x-3.4x). If anything, anchored is fractionally better at every
point checked (all four ratios are >1), the **opposite direction** of the D4 finding and of what the
2026-07-26 documents assert. `Delta_dual` agrees between contrasts to 1.6e-16-2.2e-16 at every point
(equivalence holds, as expected — this is an exact orthogonal reparameterization, proved by
construction in Section 5, not merely observed).

**Why the D4 finding likely does not transfer**: `docs/fullA_cm_conditioning_and_adaptive_grid_
report.md`'s own Section on orthonormal's structural cost notes `orthonormal_contrast_matrix(D)` is
a fully dense `D x D` matrix mixing all origins in a threshold's block, vs. anchored/interval's
per-origin-pair (2-origin) sparsity. At D=4 this rotation meaningfully reshapes a very
low-dimensional (`nO=3`) contrast subspace's conditioning. At D=20 (`nO=19`), the same fixed-shape
rotation is applied to a subspace an order of magnitude larger, and the D20 Hessian's conditioning
is dominated by a different mechanism entirely (compare the RAW magnitudes: D4 `cond(H)~1e4-2e5` vs.
D20 `cond(H)~1e6-1e6` for cumulative — three-to-four orders of magnitude larger baseline
conditioning at D20, likely from `W`/`D` scaling of the underlying bin-contingency tables, not from
the contrast choice at all). This is a plausible mechanism, not independently proven here — flagged
as a natural follow-up, not asserted as fact.

## 4. Decision rule (item 16) applied

- **Equivalence**: passes (Section 3, and the analytic congruence proof in Section 5).
- **Conditioning**: no meaningful difference at D20 (<1.7% either direction) — does not clear a bar
  for preferring orthonormal, and does not clear a bar for preferring anchored either. The D4-only
  "2.6-3.4x" advantage cited by three prior documents as grounds for an orthonormal default **does
  not survive to D20 and should not be relied on as a justification going forward.**
- **Complete inner solve / outer progress**: not separately measured for the contrast axis alone
  (the D20 timing numbers in Section 3's underlying log show cold/warm solve times statistically
  indistinguishable between contrasts, consistent with there being no real conditioning difference
  to produce a timing difference from).

**Verdict: `ORIGIN_CONTRAST_DEFAULT = inconclusive`.** Neither contrast scheme clears item 16's bar
for a confident default recommendation at D20 — the honest reading of this session's evidence is
"pick either; it does not matter numerically at production scale," not "orthonormal wins" (three
prior documents' claim) and not "anchored wins" (the actual wired default, which happens to already
be anchored, but not because of any documented D20 evidence supporting it — see Section 2). Given
this, and given the task's explicit instruction against flipping defaults without real D20 evidence
*for* a change, the actually-wired `contrasts=:anchored` default is left untouched. The prior
documents' "orthonormal is decided" claims should be treated as superseded by this document for any
future D20-scale decision-making.

## 5. Congruence is exact by construction, not merely close numerically

Anchored and orthonormal contrasts are **not** two independently-estimated quantities that happen to
agree — the orthonormal CM columns are built by literally right-multiplying the anchored block by
the closed-form orthonormal matrix `R` (`CM[:, cols] .= block * R`, `common_marginals_interval.jl`
and the cumulative equivalent in `common_marginals_moments.jl`), and the production Hessian assembly
code documents the resulting relationship as an explicit congruence transform in its own comments:

```julia
# H_CC_final[block_l,block_l'] = R' * H_CC_raw[block_l,block_l'] * R   (congruence, per threshold-block pair)
```

(`cm_hessian_architectures.jl:326-327`; `cm_hessian_architecture_interval.jl:140` implements the
identical transform for the interval basis). Since `R` is orthonormal (`R'R = I`, verified in
`docs/key_results/basis_conditioning_2026-07-25.txt`: `max|(CR)'*(CR) - I| ~ 2e-16` at D=4 and D=20),
this is a similarity transform under an orthogonal change of basis — `Delta_dual` (and every other
basis-independent economic quantity) is invariant by construction, and any residual "equivalence
check" is confirming the implementation matches its own specification, not testing an open
mathematical question. This is why Section 3's Delta_dual agreement is reported as strong (1e-16) as
it is, and why no numerically-observed disagreement beyond floating-point roundoff was ever a live
possibility here — the interesting empirical question was always conditioning, never equivalence.

## 6. Honest gaps

- As with the basis-axis document, common Fréchet and CM+ZC were not independently re-run at D20 for
  the contrast axis this session. The `cctx.R !== nothing` congruence mechanism is shared identically
  across all three CM-grid-bearing families (`production_backend_manifest.jl`'s
  `restriction_hessian_backend = cctx.R !== nothing ? :cm_bin_prefix_plus_congruence : :cm_bin_prefix`
  label is family-agnostic), making a differing verdict there unlikely, but unmeasured.
- The doc-vs-code default contradiction (Section 2) is reported, not resolved. Resolving it (either
  updating the three 2026-07-26 documents to say "anchored, per the actual code" or changing the
  code to match the documents' claim) is an explicit decision this session leaves to the user/next
  session, since neither direction is compelled by the D20 evidence gathered here.
