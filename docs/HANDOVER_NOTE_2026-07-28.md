# Handover note — structured CM/ZC Hessian optimization (2026-07-28)

For whoever picks this up next. Full detail is in
`docs/PRODUCTION_STRUCTURED_HESSIAN_OPTIMIZATION_MASTER_2026-07-28.md` — read that first. This note
is just orientation and the things most likely to trip you up.

## Where things are

- Branch: `optimize/production-structured-CM-ZC-hessian-2026-07-28`, HEAD `9a020bc`.
- Base: `origin/production/fullA-exact@5b4f9da` (the CM/common-Fréchet harmonization merge).
- **Not merged, tagged, or pushed to `production/fullA-exact`.** The user was asked explicitly and
  had not yet said go/no-go when this session ended — that decision is still open.
- Three sub-branches already merged in (kept for history, don't need separate attention):
  `agent/d4-gates-zc-centering-2026-07-28`, `agent/d20-profile-flexcm-frechet-2026-07-28`,
  `agent/d20-profile-cmzc-originzc-hzz-2026-07-28`.
- Worktrees for all of the above still exist under `/bbkinghome/edav/gravity_robustness/worktrees/`
  if you want to inspect individual sub-branch history — not required, everything relevant is
  already merged into the main branch above.
- Dropbox: `dropbox:Gravity robustness/Analysis/Server Output/structured_cm_zc_hessian_production_session_2026-07-28`.

## What's actually done and validated

- A real, previously-latent bug fixed: `WinnerBinCrossScratch`'s constructor (`winner_pair_cross_hessian.jl`)
  had two arguments swapped, silently breaking the `:winner_bin` H_EC backend for any context with a
  real `CompressedFactual`. Found independently twice, fixed, confirmed at D=4 and D=20.
- Threaded H_EC/H_EZ/H_CZ: opt-in, bit-exact at D=4 and real D=20 for all four families.
  flexible_cm/common_frechet: 1.42x complete-callback. origin_zc: H_EZ 4.5x (peak 5.3x@t=10).
- H_ZZ backend: `:blas_gemm` at BLAS-threads≥8 is the right default at true production width
  (nx=210) — **this reverses an earlier same-day session's `:threaded_packed` recommendation**,
  which was measured at a 10x-narrower config (nx=20) and does not generalize. Don't re-adopt
  `:threaded_packed` without re-checking width first.
- ZC-centering cache (Section 10 of the original task brief): implemented opt-in, D=4-validated,
  confirmed rebuilds-per-callback drops from `n_callbacks` to 1-per-outer-point.
- D=4 gates: all four families pass (flexible_cm 6/6, common_frechet 6/6 — newly added this
  session, cm_meanzc 39/39 — was 0/40 before the constructor fix, origin_zc 26/26 + one documented
  pre-existing synthetic-data infeasibility at a specific K config, not a bug).
- Everything above is **opt-in** — no production default was flipped. Zero behavior change unless
  someone explicitly sets the new flags/Refs.

## The one thing you most need to know: KNITRO concurrency on this host

Mid-session, a different Claude session reported cm_meanzc's real driver working fine on this exact
commit, directly contradicting an earlier finding in this branch's history that it was "broken by a
production regression." That earlier finding was investigated live and found to be **wrong** — it
was KNITRO resource contention from running multiple concurrent solves on this shared host, not a
code bug. Confirmed via a direct A/B: the same unmodified script run alone always succeeds; run as
5 simultaneous processes, ~40% fail with the identical error signature that looked like a "regression."

**Practical implication for you**: if you see a KNITRO callback error (`nStatus=-500`/`-502`) on
this host, check whether multiple KNITRO solves were running at the same time before concluding
it's a real bug — rerun in isolation first. This appears to affect cm_meanzc's inner solve more
than the other three families (it's the widest/most compute-heavy of the four), but whether the
other three are also susceptible under concurrent load was **not tested** — don't assume they're
immune just because they didn't show it this session.

**Meta-lesson, not just a code finding**: when a report from another session contradicts your own
evidence, the right move is to re-run it live and get a third data point, not to just pick a side
or average the two claims. That's what resolved this.

## Other gotchas from this session

- `CS` in `full_aod_diag/d4_exact/*.jl` diagnostic scripts is the `CounterfactualSensitivity`
  module (loaded via `include("context.jl")`), not a data field — don't reassign it, don't confuse
  it with `ctx.so.counters`.
- Don't call `archC_verified_state`/`archC_meanzc_base_state`/`archOZ_base_state` (or the
  `build_cm_*_production_context` constructors) directly for a "warmed" state — go through the real
  public drivers (`run_cm_upper_checkpointed`, `run_originzc_upper_checkpointed`) instead. See
  `smoke_delta1_*.jl` for known-working call patterns per family.
- `run_cm_upper_checkpointed` hardcodes real D=20 data internally — it cannot be pointed at a D=4
  synthetic context. Don't try to use it as a "warm-up" trick for a D=4 test.
- Threading/backend selection is via global `Ref`s (`CROSS_HESSIAN_THREADED_DEFAULT[]`,
  `CROSS_HESSIAN_WORKERS_DEFAULT[]`, `ZC_GRAM_BACKEND_DEFAULT[]`, `ZC_GRAM_THREADED_WORKERS_DEFAULT[]`,
  `ZC_CENTERED_CACHE_ACROSS_CALLBACKS[]`), not driver kwargs.
- Always `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` except explicit BLAS-thread sweeps.

## Open items, not started or not finished

1. **The actual go/no-go decision on merging to `production/fullA-exact`** — everything is
   validated and ready; this just needs an explicit yes from the user.
2. Whether flexible_cm/common_frechet/origin_zc share cm_meanzc's KNITRO-concurrency sensitivity —
   untested.
3. H_CZ's threaded gain (cm_meanzc-only block) is weaker/less reliable than H_EZ's — loses to
   serial at t=4/t=8, only wins at t=20. Worth a closer look if cm_meanzc perf matters going
   forward.
4. flexible_cm's own pre-existing D=4 gate has a coverage gap — it never actually exercises the
   `:winner_bin` path (builds `core_cf_ref` as `nothing`). Found this session, not fixed (was out
   of scope). It's why the constructor bug above was invisible to that gate for as long as it was.
5. Section 9's "realistic five-family process-parallel resource plan" benchmark (an explicit
   "if you have time" item in the original task brief) was not attempted.
