# Interval vs Cumulative CM Basis — 2026-07-26

## Status: NOT ATTEMPTED this session

Task §7 asks for a from-scratch implementation and benchmark of an **interval**-basis CM
restriction (one active bin index per origin/draw, interval-probability targets) as an alternative
to the current **cumulative** basis (`build_cm_augmented_obj`/`build_cm_production_context`,
confirmed the only basis wired into any production entry point this session touched — see
`ORTHONORMAL_ORIGIN_CONTRAST_BENCHMARK_2026-07-26.md`).

This is genuinely new numerical-kernel and moment-construction work, not a configuration flip: task
§7.2 itself asks for the interval-basis Hessian to be **"derived from scratch"** (not assumed
inherited from the cumulative Architecture-C derivation) before any benchmark is meaningful — get
that derivation wrong and every downstream comparison (arithmetic complexity, conditioning, KNITRO
iteration count) would be comparing against an incorrect baseline while looking numerically
plausible. That is exactly the failure mode this project's own standing feedback
(`feedback-verify-before-causal-claims`, `feedback-gravity-elimination-zero-is-not-calibration` —
both about trusting a plausible-looking numeric agreement without independently reconstructing and
diffing) warns against, and not a risk worth taking under this session's remaining time budget.

**A CM lookup kernel already exists with an `:interval` method variant**
(`cm_lookup_kernels.jl`, referenced in the inherited remediation's own Phase B1 writeup — that
session's first draft of the CM lookup port actually defaulted to `method=:interval` and had to be
corrected to `:suffix` because production's own moment builder uses the cumulative basis, not
interval). That existing `:interval` code path was built for a lookup-FG optimization, not
validated as a production moment-construction basis in its own right, and per the inherited
session's own account, using it against a cumulative-basis production context produced universal
KNITRO infeasibility (`nStatus=-400`) — i.e. it is **not** a drop-in interval-basis implementation
of the kind task §7 wants; it would need to be re-derived and gated as its own production context,
not merely re-enabled.

## Scoped follow-on

1. Derive the interval-basis moment construction and its Hessian from scratch (task §7.2's own
   instruction), independently of `cm_lookup_kernels.jl`'s existing `:interval` variant — reuse it
   only after confirming its targets/normalization match the production interval-basis
   specification, not by assumption.
2. Build the four arms (cumulative+anchored, cumulative+orthonormal, interval+anchored,
   interval+orthonormal) and verify exact feasible-set equivalence and dual-coordinate
   transformations (task §7.1) before any performance comparison.
3. Only then run the task §7.2/§7.3 comparison and decision rule.

See also `INTERVAL_BASIS_HESSIAN_OPTIMALITY_AUDIT_2026-07-26.md` for the Hessian-specific half of
this same deferred item.
