# Section 11 — algorithmic-parity mode

## Definition (task §11's own required settings)

`ab_mode=algorithmic_parity` requires: `JULIA_NUM_THREADS=1`, BLAS threads=1, outer-gradient
coordinate threading disabled in both arms, bandwidth cache disabled in both arms, same outer
algorithm/options, same decoded initial state, same delta/direction, same `lower_limit`, same
screens/verification, same evaluation/gradient budget, FULL powered coordinates, REDUCED
powered-relative coordinates (since powered mode passed its own gates in sections 6/9/10).

## REDUCED side — implemented and confirmed real

REDUCED's own `run_profiled_upper_constrained`/`run_profiled_upper_constrained_free_nu` already
expose every needed knob directly, no new code required:

- `JULIA_NUM_THREADS=1` at process launch (`-t 1`)
- `OPENBLAS_NUM_THREADS=1` (this repo's own standing rule, already followed throughout this
  session)
- `threaded_gradient=false` (disables the coordinate-loop threading gated in section 7)
- `cache=nothing` (the default — bandwidth cache is opt-in via an explicit `cache=`/`bwcache`
  argument that production callers do not currently pass; confirmed by direct read, `cache`
  defaults to `nothing` in both drivers)

Real confirmation runs (D20/W=20,000, `-t 1`, `threaded_gradient=false`, `a_coordinate_mode=
:profiled_powered_relative_A`): flexible_cm and origin_zc both complete cleanly under these
settings (see `key_results/section11_algorithmic_parity/`) — confirms the REDUCED side of an
algorithmic-parity harness is genuinely available and functional, not merely configured on paper.

## FULL side — NOT built this session (precise blocker)

FULL's own production wrapper (`run_cm_upper_checkpointed`, confirmed by direct read in the prior
session's own FINAL_CLOSEOUT, `cm_checkpoint.jl:1344-1365`) calls its analytic gradient
(`cm_production_gradient_cplus`/family equivalents) with `threaded=true, h_mode=:cached`
**unconditionally** — there is no existing kwarg on the production wrapper itself to disable
either. The underlying gradient function (`composite_gradient_at_fast` and its family-specific
analogues) DOES accept `threaded`/`h_mode` as explicit kwargs (confirmed: section 7's own
`instrumented_composite_gradient_at_fast` calls it exactly this way), so a diagnostic adapter is
possible in principle — but constructing one requires replicating FULL's own KNITRO problem
setup (bounds, `lower_limit`, screens, checkpoint/manifest wiring) outside the trusted, already-
production-tested `run_cm_upper_checkpointed` path, in FULL's own ~90-file dependency graph that
this session did not have remaining budget to trace safely. Given this repo's own standing
caution against under-verified changes near production code, and the explicit task constraint
"do not change FULL production defaults," building this adapter under further time pressure this
session was judged a real risk not worth taking casually — reported as a precise, honest blocker
rather than rushed.

## Verdict

```
ALGORITHMIC_PARITY_MODE =
    REDUCED_side: implemented_and_confirmed (flexible_cm, origin_zc; common_frechet/cm_meanzc/
        unrestricted not independently run)
    FULL_side: not_built_this_session (precise blocker: run_cm_upper_checkpointed hardcodes
        threaded=true/h_mode=:cached; a diagnostic adapter bypassing it is possible in principle
        but was not constructed)
    ALGORITHMIC_PARITY_AB_W20K = fail_full_side_adapter_not_built (all 5 families -- a genuine A/B
        under this mode requires BOTH arms, and only REDUCED's is available)
```

This is real remaining work for a follow-up session, not a silent gap.
