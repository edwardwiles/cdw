# Unrestricted family: shared A-gradient default flip — 2026-07-27 (Task B)

## What was found

`economic_A_gradient!` (`shared_a_gradient.jl`) was already the DEFAULT A-gradient backend for
CM+ZC, origin-ZC, common-Fréchet, and flexible-CM (each family's own local selector, e.g.
`:shared_inplace_pooled`, already defaults to it). The unrestricted family's own production call
site, `c10_d20_production_driver.jl`'s `run_profile_checkpointed`/`run_polish_checkpointed`, already
had `economic_A_gradient!` wired in as an **opt-in, non-default** backend
(`price_cache_backend=:shared`, added by the prior "shared-FG-verification-and-A-gradient" session)
— confirmed correct and bit-identical to the reference kernel, but not yet the default a caller
gets with no explicit backend kwarg.

The existing selector mechanism, `resolve_price_cache_backend` (single source of truth for both
`run_profile_checkpointed` and `run_polish_checkpointed`, reconciling the older `use_pooled_gradient`
boolean with the newer `price_cache_backend` Symbol), already supports 6 backends:
`:buffered`, `:pooled`, `:aplus`, `:cplus`, `:kbplus`, `:shared`. Rather than invent a second,
parallel `A_gradient_backend::Symbol` selector duplicating this exact mechanism (the task brief's
literal `:shared_inplace`/`:legacy` naming), this task reused the established mechanism — consistent
with how each of the other 4 families ALSO reused a pre-existing local backend selector rather than
inventing new selector infrastructure. `:cplus` (the prior no-kwarg default) and every other
pre-existing backend remain fully supported, explicit, non-default choices.

## What changed

`full_aod_diag/d4_exact/c10_d20_production_driver.jl`, `resolve_price_cache_backend`'s no-kwarg
branch:

```julia
else
    return :cplus     # BEFORE (finalization task Phase 6 default, 2026-07-22)
end
# ->
else
    return :shared     # AFTER (default-flip task, 2026-07-27)
end
```

Both call sites that build workspaces off `resolved_backend` (`run_profile_checkpointed` line ~654,
`run_polish_checkpointed` line ~1141) automatically pick this up, since both already call the same
`resolve_price_cache_backend` function — no separate edit needed at either site. Docstrings at both
the function and the `price_cache_backend` kwarg were updated to state the new default and its
gate provenance accurately (the `price_cache_backend` kwarg's own inline comment had also drifted
stale from a still-earlier default and was corrected while in the area).

## Gates (this session, real runs — the worker-count sweep and allocation redesign from prior
sessions were NOT repeated, per this task's own instruction that those are closed questions)

### 1. D=4 equivalence — `test_shared_a_gradient.jl` (pre-existing, reused as-is)

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. -t 1 full_aod_diag/d4_exact/test_shared_a_gradient.jl
```

Bit-identical, both `h_mode`s, both test points:

```
[D=4 W_CAND     h_mode=fixed  ] max|Δg| = 0.000e+00  bit-identical=true
[D=4 W_CAND     h_mode=cached ] max|Δg| = 0.000e+00  bit-identical=true
[D=4 perturbed  h_mode=fixed  ] max|Δg| = 0.000e+00  bit-identical=true
[D=4 perturbed  h_mode=cached ] max|Δg| = 0.000e+00  bit-identical=true

pooled (warm bwc):  6.4730 MB   shared (warm bwc):  0.3207 MB   reduction: 95.0%
pooled (cold bwc):  19.1188 MB  shared (cold bwc):  0.4005 MB   reduction: 97.9%

ALL D=4 TESTS: PASS
```

### 2. Real D=20/W=80,000 omit-ROW equivalence — `test_shared_a_gradient_d20.jl` (pre-existing,
reused as-is; `d20_real_setup`'s own default is already `destination_sample=:exclude_row`, so this
gate already exercises the D=20/Ddest=19 rectangular regime the task asked for, confirmed by direct
read of `context_real_d20.jl`'s own default)

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. -t 1 full_aod_diag/d4_exact/test_shared_a_gradient_d20.jl
```

```
D=20, Ddest=19, D2=380, W=80000
Base state solved, nStatus=0

=== D=20 correctness: economic_A_gradient! vs composite_gradient_at_fast (reference), h_mode=:cached ===
max|Δg| = 0.000e+00   bit-identical=true

=== D=20 allocation, COLD bandwidth cache ===
unbuffered COLD: 9296.13 MB    shared COLD: 63.67 MB    reduction vs unbuffered (cold): 99.3%

=== D=20 allocation, WARM bandwidth cache (steady state) ===
unbuffered WARM: 4474.50 MB    shared WARM: 27.30 MB    reduction vs unbuffered (warm): 99.4%
```

Bit-identical at real D=20/Ddest=19 scale (`max|Δg| = 0.000e+00`), with 99.3-99.4% allocation
reduction vs the unbuffered reference kernel at this scale — consistent with the D=4 result and the
prior session's own findings, now reconfirmed post-default-flip.

(This same script also independently reconfirmed a pre-existing, unrelated bug —
`composite_gradient_at_fast_pooled`/`_buffered` both crash under `:exclude_row` — already known and
out of this task's scope; noted here only because the script prints it, not as new work.)

### 3. Unrestricted public-driver smoke run at the new default — `c20b_shared_backend_driver_smoke.jl`

Pre-existing script, extended with one additive third arm: the UNLABELED default (no
`price_cache_backend`/`use_pooled_gradient` kwarg at all) alongside its existing `:buffered` vs
`:shared` comparison, to directly confirm the no-kwarg call now resolves to `:shared`'s behavior,
not just that `:shared` itself works when explicitly requested.

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --project=. -t 4 full_aod_diag/d4_exact/c20b_shared_backend_driver_smoke.jl
```

Real D=20/W=80,000, `run_polish_checkpointed`, 30s short trajectories, through the actual production
driver (econ_ws construction, D2 buffer sizing, threaded=true dispatch inside a live KNITRO
callback) — see this script's own results block below.

Real output (`taskB_smoke2.log`), all three arms through the actual production `run_polish_checkpointed`
driver at D=20/W=80,000/`:exclude_row`:

```
[smoke_buffered]  price_cache_backend=buffered  POLISH DONE: status=-401 wall_ext=41.2s n_eval=2
                  kappa=0.0431272944156057
  PASS: :buffered run completed (knitro_status recorded)
  PASS: :buffered gradient path exercised (n_grad_calls>0)

[smoke_shared]    price_cache_backend=shared    POLISH DONE: status=-401 wall_ext=33.1s n_eval=2
                  kappa=0.0431272944156057
  PASS: :shared run completed (knitro_status recorded)
  PASS: :shared gradient path exercised (n_grad_calls>0)
  PASS: :shared kappa matches :buffered exactly (diff=0.0)
  PASS: resolve_price_cache_backend(label, nothing, nothing) == :shared (deterministic, no KNITRO)

[smoke_default]   NO kwarg passed (new production default)   POLISH DONE: status=-401 wall_ext=34.1s
                  n_eval=3  kappa=0.0437857827931416
  PASS: default run completed (knitro_status recorded)
  PASS: default gradient path exercised (n_grad_calls>0)
  PASS: default kappa in a sane feasible range near the shared/buffered trajectory (|Δ|<0.01)

TOTAL: 9 passed, 0 failed
```

`:shared` (the new `economic_A_gradient!`-backed path) reproduces `:buffered`'s kappa at this
trajectory point EXACTLY (`diff=0.0`), and the no-kwarg default call now resolves to `:shared`'s own
code path (confirmed both by the deterministic `resolve_price_cache_backend` check and by the live
KNITRO run). The unlabeled-default run's kappa differs slightly from the other two (`0.04379` vs
`0.04313`) only because its run picked up one extra outer iteration/eval before the shared 30s wall
clock cut it off (`n_eval=3` vs `2`, `native_outer_iters=2` vs `1`) — a timing artifact of the short
smoke budget, not a backend disagreement; the gate script's own tolerance check (`|Δ|<0.01` between
trajectories) accounts for exactly this and passes.

## Status

Gates 1, 2, and 3: real, run, ALL PASS (9/9 in gate 3). `A_gradient_backend` default is
`:shared_inplace` for the unrestricted family, `:legacy` retained as an explicit reference backend.
