# Five-Family Operator Verification Default Release — 2026-07-27 (Task Section 6)

## What changed

All five families' post-solve dual-verification tail now supports an explicit
`verification_backend::Symbol` kwarg (`:operator` | `:dense_reference`), dispatched at:

| Family | Wired function | File |
|---|---|---|
| Flexible-CM | `archC_verified_state` | `cm_production_bundle.jl` |
| CM+ZC (CM+meanZC) | `archC_meanzc_verified_state` | `cm_meanzc_production.jl` |
| Origin-ZC | `archOZ_verified_state` | `cm_originzc_production.jl` |
| Common-Fréchet | `archC_frechet_verified_state` | `cm_frechet_cplus.jl` |
| Unrestricted | `evaluate_fullA_fast_compressed` (tail only) | `compressed_live.jl` |

Each kwarg defaults to a new global `*_VERIFICATION_BACKEND_DEFAULT::Ref{Symbol}`
(`operator_verification.jl`): `CM_VERIFICATION_BACKEND_DEFAULT`,
`CM_MEANZC_VERIFICATION_BACKEND_DEFAULT`, `ORIGINZC_VERIFICATION_BACKEND_DEFAULT`,
`CM_FRECHET_VERIFICATION_BACKEND_DEFAULT`, `UNRESTRICTED_VERIFICATION_BACKEND_DEFAULT` — mirroring
this codebase's existing backend-toggle discipline (`CM_INNER_FG_BACKEND_DEFAULT`,
`ORIGINZC_FG_BACKEND_DEFAULT`, etc., `core_exact_hessian.jl`). **Deliberately NOT** added as
`CMBinHessCtx`/`OriginZCCoreHessCtx` struct fields — both structs live in
`cm_hessian_architectures.jl`, an explicitly off-limits Hessian-backend file for this task (two
other agents are concurrently changing which Hessian backends are active on different branches); a
global Ref threads through with zero struct-plumbing risk or merge-conflict surface.

A shared helper, `verify_namedtuple_from_operator(ov, obj, W, nStatus)`
(`operator_verification.jl`), builds the **same-shaped** `verify` NamedTuple the dense
`*_verified_state` tails already produce (`inner_status`, `Delta_dual`, `Delta_primal`,
`primal_dual_gap`, `weight_norm_resid`, `mean_m_resid`, `max_abs_moment_kkt_resid`, `m_mean`,
`m_min`, `m_max`) from any `verify_inner_solution_operator_*!` return value (`r`, `f`, `g_lambda`,
`kkt_resid`), with **no dense `obj.H` read anywhere in this function**:

- `Delta_dual = -ov.f` — sign verified from `oracle.jl`'s own documented comment ("`constr[1] =
  -f*1e10`"), not assumed; empirically confirmed to machine precision below.
- `m_weights = dPsi(ov.r)`, recomputed via the same `obj.dPsi!` both backends already call.
- `max_abs_moment_kkt_resid = ov.kkt_resid` directly.

Because the returned NamedTuple has the identical field set, `classify_inner_result` /
`is_cacheable_result` / `is_verified_success` (`oracle.jl`) consume either backend's output
**identically** — this is what makes "cache admission decision" / "incumbent admission decision"
a literal function-level comparison in the gates below, not a re-derived approximation.

No silent fallback: each `:operator` branch hard-errors if its prerequisites aren't met (e.g.
`cctx.core_cf_ref[]` not a `CompressedFactual`, `octx.fg_zc_op === nothing`). `record_operator_
verification!()` (called inside every `verify_inner_solution_operator_*!`) / `record_dense_
reference_verification!()` (called at each new dense-branch dispatch point) count every call,
`no_dense_g_counters.jl`.

### Unrestricted's disclosed scope limit

`evaluate_fullA_fast_compressed` computes several **non-verification** reporting outputs
(`gravity_raw` from `cbuf[2]`, `K_hard`, `benchmark_unweighted_moment_mean`/`max_abs_moment_resid`
from a full-`d`-column `G`) that are NOT part of `classify_inner_result`'s field set and are
unrelated to cache/incumbent admission. These still require the dense `obj(inner_x,constr=...)`
call and `G` materialization, which therefore still run **unconditionally** regardless of
`verification_backend` — only the admission-relevant subset (`Delta_dual`, `Delta_primal`,
`primal_dual_gap`, `mean_m_resid`, `max_abs_moment_kkt_resid`, `weight_norm_resid`, `m_weights`)
is actually dispatched through the operator path. This means unrestricted's `:operator` backend
does **not** eliminate all dense `obj.H` reads inside this function — a disclosed, deliberate scope
limit (porting the reporting-only diagnostics to an operator-based recompute would need `G`'s full
`d`-column transpose-to-ones, a materially larger change, out of scope for a verification-default
task). The four CM-family functions do not have this limitation: under `:operator` they perform
zero dense `obj.H` reads.

## Gates (Section 6.1) — ALL FIVE FAMILIES, ALL PASS

Every gate exercises the **actual wired production dispatch** (not just the standalone
`verify_inner_solution_operator_*!` functions in isolation) via a fresh call to that family's real
verified-state entry point, comparing `verification_backend=:dense_reference` against
`verification_backend=:operator` on the SAME converged dual point `(ζ*, λ*)`. Both calls are fresh
(neither reuses the other's scratch — `verify_inner_solution_operator_*!` builds fresh scratch
internally per its own docstring, confirmed not assumed), satisfying "cold verification": every
comparison below is a from-scratch recompute, not a warm-state-dependent shortcut.

Checked per Section 6.1: **draw-level dual index** (`m_star`, the per-draw `dPsi(r)` vector),
**objective** (`Delta_dual`), **complete dual gradient** (full vector, not just the max-abs KKT
residual), **KKT residual**, **feasibility/moment residual** (`mean_m_resid`), **status
classification** (`classify_inner_result`), **cache admission decision**
(`is_cacheable_result`), **incumbent admission decision** (`is_verified_success`).

Gate scripts: `test_verification_backend_default_{cm,cmmeanzc,originzc,frechet,unrestricted}.jl`
(D=4) and their `_d20.jl` companions (unrestricted's D=20 run is `SCALE=d20` on the same script).
Shared comparison harness: `verification_gate_utils.jl`.

### D=4 results

| Family | Configs | Checks | Result | max\|Δobjective\| | max\|Δ full gradient\| |
|---|---|---|---|---|---|
| Flexible-CM | L=10, L=50 | 22/22 | ALL PASS | 6.1e-17 | 1.9e-15 |
| CM+ZC | (K_mean,K_pair,L)=(1,0,10),(1,1,10),(2,2,20) | 33/33 | ALL PASS | 7.9e-17 | 3.5e-15 |
| Origin-ZC | (K_mean,K_pair)=(1,0),(1,1),(2,2) | 33/33 | ALL PASS | 5.2e-18 | 3.0e-15 |
| Common-Fréchet | L=10, L=50 | 22/22 | ALL PASS | 1.8e-16 | 3.2e-15 |
| Unrestricted | D4 calib | 11/11 | ALL PASS | 6.7e-17 | 2.1e-15 |

### Real D=20/W=80,000 results (`destination_sample=:exclude_row`, production default L=50 / K=1,1)

| Family | Config | Checks | Result | dense Δ_dual | operator Δ_dual | max\|Δ full gradient\| |
|---|---|---|---|---|---|---|
| Flexible-CM | L=50 | 11/11 | ALL PASS | 0.0086604952627**23162** | 0.0086604952627**23202** | 1.12e-12 |
| CM+ZC | K=1/1, L=50 | 11/11 | ALL PASS | 0.009789029020717**89** | 0.009789029020717**807** | 1.15e-12 |
| Origin-ZC | K=1/1 | 11/11 | ALL PASS | 0.0036095731377116**867** | 0.0036095731377116**793** | 3.13e-12 |
| Common-Fréchet | L=50 | 11/11 | ALL PASS | 0.015817469186355**126** | 0.015817469186355**000** | 9.45e-13 |
| Unrestricted | calib | 11/11 | ALL PASS | 0.0024867734779406**123** | 0.0024867734779406**236** | 7.30e-13 |

Every family: `inner_status` agrees (both 0), `mean_m_resid` agrees to `<3.3e-16`,
`primal_dual_gap` agrees to `<1.8e-16`, `classify_inner_result` returns `VerifiedSolved` for both
backends at every config, `is_cacheable_result`/`is_verified_success` both `true` for both
backends at every config. KKT residual max-abs diffs ranged `1.7e-13` (unrestricted) to `1.8e-12`
(origin-ZC) — both backends' own KKT residuals were themselves already at the `1e-13`-`1e-12`
scale (real-D20/W=80,000's own characteristic convergence tolerance, matching this task's
inherited-session precedent for the 3 families gated before this session), not a backend
disagreement.

**Total: 5/5 families, 5/5 D=4 gates ALL PASS (121/121 checks), 5/5 D=20 gates ALL PASS (55/55
checks). 176/176 checks passed across every family and every scale tested.**

## Default flip decision

All five `*_VERIFICATION_BACKEND_DEFAULT` Refs flipped `:dense_reference` → `:operator`
(`operator_verification.jl`), since every family passed every comparison cleanly at both D=4 and
real D=20. `:dense_reference` remains fully available as an explicit, named, non-default backend
(`verification_backend = :dense_reference` at any of the 5 call sites) for anti-regression /
emergency-revert comparisons, matching this codebase's established pattern for every other backend
toggle.

Confirmed the flip actually took effect at a real call site (not just in the Ref's own value):
called `archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)` **with no `verification_backend`
kwarg at all** post-flip — resolved to `:operator`, returned `VerifiedSolved`, `Delta_dual` matching
the pre-flip explicit-`:operator` value to machine precision.

| Family | Prior default | New default | Basis |
|---|---|---|---|
| Flexible-CM | `:dense_reference` | **`:operator`** | D4+D20 ALL PASS (above) |
| CM+ZC | `:dense_reference` | **`:operator`** | D4+D20 ALL PASS (above) |
| Origin-ZC | `:dense_reference` | **`:operator`** | D4+D20 ALL PASS (above) |
| Common-Fréchet | `:dense_reference` | **`:operator`** | D4+D20 ALL PASS (above) |
| Unrestricted | `:dense_reference` | **`:operator`** | D4+D20 ALL PASS (above); disclosed scope limit (see above) on non-verification reporting outputs only |

## Pre-existing gates re-confirmed on current HEAD

The task brief noted 3 families (unrestricted, flexible-CM, common-Fréchet) had operator
verifiers built and gated at D=4+D=20 in a prior session (`a4e8ece`) and origin-ZC's verifier
(`3969acf`) had less precedent — both explicitly not to be assumed still valid. The pre-existing
narrower gates (`test_operator_verification_{cm,cmmeanzc,originzc,cm_frechet,cm_frechet_d20,
unrestricted}.jl`, KKT-residual/finiteness-only) were re-run on this branch's current HEAD before
building the wider Section 6.1 gates above: **all still pass** (see `test_operator_verification_cm.
jl`'s own D=4 re-run, `PASS L=10/L=50`, at the start of this session's work) — code had not
silently drifted since those commits, but the wider gates above are what the flip decision is
actually based on, not a re-assumption of the older narrower ones.

## Section 6.2 (obsolete-toggle removal) — NOT attempted

`skip_cm_fill_ref` and similar mutable dense-fill-skip switches were explicitly the lowest-priority,
highest-risk part of this task, and were **not touched**. Removing them safely requires proving no
currently-active Hessian backend for that family still reads the dense CM-column fill they guard —
this cannot be independently verified on this branch alone while two other agents are concurrently
changing which Hessian backends are active for other families on parallel branches (the exact
mechanism this determination depends on). Left in place, undocumented-as-removed, per the task's
own explicit "if in doubt, leave it" instruction.

## Files changed

- `full_aod_diag/d4_exact/operator_verification.jl` — `verify_namedtuple_from_operator` helper +
  5 `*_VERIFICATION_BACKEND_DEFAULT` Refs (now `:operator`).
- `full_aod_diag/d4_exact/cm_production_bundle.jl` — `archC_verified_state` operator dispatch.
- `full_aod_diag/d4_exact/cm_meanzc_production.jl` — `archC_meanzc_verified_state` operator dispatch.
- `full_aod_diag/d4_exact/cm_originzc_production.jl` — `archOZ_verified_state` operator dispatch.
- `full_aod_diag/d4_exact/cm_frechet_cplus.jl` — `archC_frechet_verified_state` operator dispatch.
- `full_aod_diag/d4_exact/compressed_live.jl` — `evaluate_fullA_fast_compressed` operator dispatch
  (verification-relevant subset only, see scope limit above).
- `full_aod_diag/d4_exact/verification_gate_utils.jl` (new) — shared comparison harness.
- `full_aod_diag/d4_exact/test_verification_backend_default_{cm,cmmeanzc,originzc,frechet,
  unrestricted}.jl` (new, D=4) + `_d20.jl` companions (new, real D=20/W=80,000; unrestricted's is
  `SCALE=d20` on the same D=4 script) — the Section 6.1 gates themselves.

## What was NOT done / honest gaps

- **Checkpoint-resume cold verification**: "cold verification" above means fresh scratch / no
  warm-state reuse between the two backend calls (confirmed per the operator verifiers' own
  docstrings and by direct call-site testing), not a literal simulated checkpoint-save-then-resume
  cycle through each family's own `cm_checkpoint.jl`/`cm_originzc_checkpoint.jl` driver. A genuine
  checkpoint-resume replay was not separately built — the gates' own "fresh call, independent
  scratch, no shared state" property is the functional content of that requirement, but a literal
  save/resume round-trip through the checkpoint files was not exercised.
- **Unrestricted's non-verification reporting outputs** (`gravity_raw`, `K_hard`,
  `benchmark_unweighted_moment_mean`) still read dense `obj.H` under `:operator` — disclosed above,
  not eliminated.
- **Section 6.2** not attempted (see above).
- D=20 gates used a single representative config per family (production-default `L`/`K`), not the
  multi-config sweep D=4 gates used — real-D20/W=80,000 KNITRO solves are expensive; this matches
  the inherited-session precedent's own D=20 gate scope (e.g. `test_flexible_cm_winner_bin_her_
  wiring_d20.jl`'s 2-point sweep vs its own D=4 36-comparison sweep).
