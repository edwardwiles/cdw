# Handover prompt: apply Variant D to OZC-CROSS

Paste everything below this line as your task prompt in a fresh Claude Code session.

---

## Context

You're continuing work in `/bbkinghome/edav/cdw_worktrees/ozc-cross-2026-08-09`, on branch
`feature/ozc-cross-2026-08-09` (based off `origin/production/fullA-exact`@`4df5254` in the canonical
repo `/bbkinghome/edav/cdw`). Nothing on this branch is committed yet. A prior session built and
verified a new restriction family, **OZC-CROSS**, and the user has now asked for one specific
follow-up: apply the existing **Variant D** fix to it. This document gives you everything you need
to do that without re-deriving it — read it fully before touching any code.

**Read `/bbkinghome/edav/gravity_robustness/CLAUDE.md` first** (project-wide operational rules —
no-defaults-on-scientific-params, no-dense-G, Dropbox push requirement, early-check-in-on-jobs, and
importantly the two "do not reach for these as blanket explanations" warnings about KNITRO
warm/cold-start and Δ\*-bimodality). Also load auto-memory as usual; two memories from the prior
session are directly relevant and referenced throughout below:
[[ozc-cross-kpair2-grid-build-2026-08-09]] and
[[feedback-control-against-base-family-before-bug-hunting]].

## What OZC-CROSS is (confirmed with the user, do not re-derive)

The base origin-specific-ZC family imposes `E[z_o^k * z_p^k] = nu_{o,k}*nu_{p,k}` only for the SAME
power level `k` on both origins in a pair (`K_pair` restrictions per unordered origin pair
`(o,p)`, `o<p`). OZC-CROSS generalizes this to the FULL ordered grid: for every unordered origin
pair and every `(k1,k2) in {1..K_pair}^2` (`K_pair^2` restrictions per pair, not `K_pair` — 9 for
K=3), `E[z_o^k1 * z_p^k2] = nu_{o,k1}*nu_{p,k2}`, power `k1` on the LOWER-indexed origin, `k2` on
the higher. No new outer parameters — same `nu_{o,k}` as the base family. Confirmed with the user
this is non-duplicative (pairs stay canonically ordered `o<p`, never revisited reversed).

## What's already done and verified (do not redo — the evidence is real)

New files, all in `full_aod_diag/d4_exact/`:
- `cm_originzc_cross_moments.jl` — `cross_pair_level_index`, `n_originzc_cross_moments`,
  `build_raw_cross_pair_matrix_levels` (raw feature builder, reuses already-built Frechet powers).
- `cm_originzc_cross_target_layout.jl` — `OriginByPowerCrossLayout`, a new `MeanZCTargetLayout`
  subtype with its own `pair_targets` method (ordinary Julia multiple dispatch).
- `cm_originzc_cross_production.jl` — `build_originzc_cross_augmented_obj`/
  `build_originzc_cross_production_context` (context construction), plus the outer-gradient
  extension `d_delta_dual_d_eta_origin_cross_vec`/`originzc_cross_fixed_contribution`/
  `cm_originzc_cross_production_gradient` (hand-derived envelope-formula extension to the
  `(k1,k2)` grid).
- `originzc_cross_outer_driver_2026-08-09.jl` — `run_originzc_cross_upper`, a lightweight outer
  KNITRO loop mirroring the established `run_cm_upper` pattern (`cm_outer_driver.jl`), NOT the
  full 500+-line `run_originzc_upper_checkpointed` (that driver's config/layout validation is
  hardcoded to the two pre-existing layouts and would need real integration work — deliberately
  out of scope both last session and probably this one too, unless the user asks for it).

**Key architectural finding, load-bearing for everything below:** the shared inner-loop machinery
(`zc_restriction_operator.jl`'s FG callback, H_ZZ gram, H_EZ cross-block) is genuinely
column-count-agnostic — confirmed by direct reading, not assumed. It needed zero changes for
OZC-CROSS. The new files above supply ONLY the raw cross-power features, the cross target formula,
and (since the accumulation shape genuinely changes) the outer-gradient envelope extension.

**Verification, all phases PASS** (full detail + raw logs in
`dropbox:Gravity robustness/Analysis/Server Output/ozc_cross_session_2026-08-09/`, `MASTER.md`
there is the fullest account): D4 synthetic (K=1/1,2/2,3/3, real KNITRO convergence, residuals
~1e-12 to 1e-14) → D4 outer-gradient (fixed-contribution vs independent operator recompute:
machine precision; analytic-vs-FD: matches as well as or better than the base family's own formula
at identical points) → D20 real Brazil-Korea data, `destination_sample=:exclude_row` (actual
production default), W=100,000 (K=1/1 22s, K=2/2 43s, K=3/3 297s, all converged, residuals ~1e-13
to 3e-11; D20 gradient checks pass) → outer-search smoke test (confirmed genuinely searching
multiple points, correct feasible/infeasible classification, clean KNITRO exit).

## The open item this task is about: the 7x Delta* jump at K=3, root-caused

At D20/W=100,000, K_mean=K_pair=3, OZC-CROSS gives Delta_dual=0.0667 vs the base family's 0.00949
at the identical point — a 7x jump. This was root-caused (NOT a bug) via a targeted ablation: a
restriction set keeping the cross grid but excluding every `(k1,k2)` combo touching level
`k*=sigma-1=2` (leaving only `{(1,1),(1,3),(3,1),(3,3)}`) gives Delta*=0.0122, only 1.29x the base
family's — a sane, proportionate increase for a genuinely new restriction. The full 9-combo grid
(5 of which touch level 2) gives the other 5.5x on top of that. This isolates the cause to the
**already-known, already-documented `k*=sigma-1` autarky-moment collinearity** (see
[[fulla-zc-profiled-focal-sigmaminus1-mean-production-ready]] / `cmzc-k2-singularity-root-cause` in
memory, and CLAUDE.md's own history of this exact issue) — present in BOTH families (neither the
base-family control run nor OZC-CROSS applied Variant D last session), just hit far more often by
the cross grid (5 of 9 combos touch level 2, vs 1 of 3 for the base family).

**Your job:** apply Variant D (focal `k*=sigma-1` mean-row omission) to OZC-CROSS, then re-run the
SAME K=3/3 D20/W=100k comparison with `kstar=2` profiled (for both the base family and OZC-CROSS)
to confirm Delta* comes back down toward a sane multiple, closing this open item.

## Everything you need for Variant D — what's reusable UNCHANGED vs what needs a mirrored extension

The base family's Variant D machinery is in `cm_originzc_target_layout.jl`, `cm_originzc_moments.jl`,
`autarky_cf.jl`, `zc_restriction_operator_ragged.jl`, `cm_originzc_production.jl`, and the driver
wiring in `cm_originzc_checkpoint.jl`. A prior-session investigation (this handover's author) read
ALL of this in full before writing this document — the following inventory is exact, not guessed.

**Reusable completely unchanged (no edit needed anywhere, just call these directly):**
- `ActiveMeanLayout(base, focal_origin, kstar, D)`, `scatter_nu_eff`, `gather_active_grad`,
  `mean_offset_from_aml` — `cm_originzc_target_layout.jl` / `cm_originzc_moments.jl` line ~359.
  All pure functions of `(base_layout, focal_origin, kstar)` — no K_pair-diagonal-vs-cross
  assumption anywhere in them. `mean_offset_from_aml(aml) = vcat(0, cumsum(length.(aml.
  mean_active_origins)))`.
- `ZCRestrictionOperator(Zraw_all_full, Zpairraw_all, D, aml::ActiveMeanLayout)` — the ragged
  constructor, `zc_restriction_operator_ragged.jl`. CONFIRMED (read in full): it only touches the
  MEAN block (`Zraw_all` compaction via `aml.mean_active_origins`); `Zpairraw_all` is passed
  through completely untouched regardless of its length/meaning. Works for OZC-CROSS's 9-block
  `Zpairraw_all` with zero changes.
- `build_originzc_core_hess_ctx` (`cm_hessian_architectures.jl`) — ALREADY aml-aware (dispatches to
  the ragged constructor above when `aug.aml !== nothing && aug.aml.active`, for BOTH `fg_zc_op`
  and `hzz_zc_op`). No edit needed; already confirmed working for OZC-CROSS's non-aml case last
  session, and its aml branch doesn't care about K_pair meaning either.
- `nu_star_value_and_dgrad(θ_full, ctx)`, `build_focal_kstar_derivative_info(ctx, pe)`,
  `apply_focal_kstar_chain_rule!(gfull, θ_full, ctx, info, D2_econ, coeff)` — `autarky_cf.jl` lines
  255-339. Pure economic-parameter functions (gp/A_dd only) — zero dependence on the ZC pair
  structure. `apply_focal_kstar_chain_rule!` operates on `g_econ` (the economic gradient block)
  before any A-coordinate rescale; `coeff` is `d_delta_d_nu_star` from whatever family's own
  `..._active_and_nustar` gradient function you call.
- `originzc_profiled_nu_value(xf, ctx)` — `cm_originzc_checkpoint.jl` lines 32-46. Also pure
  economic (`autarky_cf_scalars(ctx.obj, AodPow, σ, γ_prime_bi)`), no ctx setup beyond normal
  needed (does NOT require `enable_autarky_cf!` — that's a different, unrelated mechanism).

**Already extended this session (in `cm_originzc_cross_production.jl`, done, NOT yet re-verified
by a test run):**
- `build_originzc_cross_augmented_obj(ctx, CS, layout; aml=nothing)` — now accepts `aml`, uses
  `mean_offset_from_aml(aml)[end]` for `n_mean` when active, returns `aml=aml` in the NamedTuple.
  Mirrors `build_originzc_augmented_obj`'s own aml handling (`cm_originzc_moments.jl` lines
  ~223-298) exactly.
- `build_originzc_cross_production_context(ctx, CS, layout; aml=nothing, ...)` — threads `aml`
  through to the function above; `build_originzc_core_hess_ctx` needs no change (see above).

**NOT yet done — this is the actual remaining work, each one mirrors an exact existing base-family
counterpart:**

1. **A new `d_delta_dual_d_eta_active_and_nustar_cross(λstar, aug, aml, nu_eff; mean_m)`** in
   `cm_originzc_cross_production.jl`, mirroring `d_delta_dual_d_eta_active_and_nustar`
   (`cm_originzc_moments.jl` lines 383-428) EXACTLY, with two changes:
   - Mean-block loop: copy the base function's ragged-aware version verbatim (uses
     `aml.mean_active_origins[k]`/`mean_offset_from_aml(aml)` to locate each level's flat
     `λstar` slice and target subset — this part is IDENTICAL between families, the mean block
     never changes for OZC-CROSS).
   - Pair-block loop: use the ALREADY-WRITTEN cross-grid logic from
     `d_delta_dual_d_eta_origin_cross_vec` (this file, the non-aml version — copy its pair loop
     body verbatim: `levels = cross_pair_level_index(K_pair)`, iterate `(klin,(k1,k2))`, use
     `target_index(layout,o,k1)`/`target_index(layout,p,k2)` for the two contributions), but with
     `pair_start0 = ncore_econ + n_mean_active` (ragged-aware offset, NOT dense `K_mean*D` — copy
     this exact substitution from the base function's own `pair_start0` line).
   - Return `(eta_grad_active, d_delta_d_nu_star)` via `gather_active_grad(aml, eta_grad_dense)`
     / `gather_active_grad(aml, d_nu)` respectively (identical to the base function's own final
     two lines).

2. **Extend `originzc_cross_fixed_contribution` in place** (same function name/signature — the
   base family's `originzc_fixed_contribution` is a SINGLE function handling both aml/non-aml
   cases internally, not two separate functions, because its return shape doesn't change). Copy
   the base function's aml-aware mean-block subsetting logic verbatim
   (`cm_originzc_moments.jl` lines 459-471: `aml_local = hasproperty(aug,:aml) ? aug.aml :
   nothing`, `mean_offset = aml_local !== nothing && aml_local.active ? mean_offset_from_aml
   (aml_local) : collect(0:D:K_mean*D)`, subset `Zk_active`/`targets_active` via
   `aml_local.mean_active_origins[k]`). The pair-block loop (this file's own cross-grid version)
   needs NO change beyond using `n_mean_active = mean_offset[end]` for `pair_start0` instead of
   the current hardcoded `ncore_econ + K_mean*D`.

3. **Extend `cm_originzc_cross_production_gradient`'s aml branch**, mirroring
   `cm_originzc_production_gradient` (`cm_originzc_production.jl` lines 267-278) EXACTLY: after
   computing `g_econ`, check `aml = hasproperty(pcx.aug,:aml) ? pcx.aug.aml : nothing`; if
   `aml !== nothing && aml.active`, call your new function 1 above, then
   `apply_focal_kstar_chain_rule!(g_econ, θ_full, ctx, info, D2_econ, d_delta_d_nu_star)` (needs
   `θ_full = CS.reconstruct_full(x_free0, ctx.m)` and
   `info = build_focal_kstar_derivative_info(ctx, pe)`), and `return vcat(g_econ,
   eta_grad_active), meta`; else fall through to the existing non-aml path.

4. **Extend `run_originzc_cross_upper`** (`originzc_cross_outer_driver_2026-08-09.jl`) for the
   ragged `w = [gp; zfree; eta_active]` (one shorter than dense). Mirror
   `run_originzc_upper_checkpointed`'s `cb_F!`/`cb_G!` EXACTLY (`cm_originzc_checkpoint.jl` lines
   859-935, already read in full — the pattern is identical for both families since it's purely
   about economic-parameter/nu-vector plumbing, not restriction-block structure):
   - Accept an `aml` argument; `n_eta_total = aml === nothing ? n_eta(layout) : aml.n_eta_active`.
   - In `cb_F!`/`cb_G!`: `νvec_active = exp.(w[D2econ+1:end])`; `νvec = aml === nothing ?
     νvec_active : scatter_nu_eff(aml, νvec_active, originzc_profiled_nu_value(xf, ctx))`; pass
     `νvec` (the DENSE, scattered vector) into `archOZ_verified_state`/
     `cm_originzc_cross_production_gradient` — NOT `νvec_active`.
   - Bounds: `bounds = aml === nothing ? bounds_dense : [bounds_dense[d] for d in
     1:length(bounds_dense) if d != aml.dense_omit_idx]` (mirrors the checkpoint driver's own
     line 782 exactly) — you'll need `originzc_default_nu_bounds` (`cm_originzc_config.jl`) or
     just build a manual bounds vector matching your test script's own `eta_halfwidth` convention,
     whichever is simpler for a smoke test.
   - `w0` construction at the call site (your launcher script): `eta0_active` has length
     `aml.n_eta_active` (one shorter than dense `nu0`) — build it by computing the DENSE
     theoretical `nu0` as before, then removing the one entry at `aml.dense_omit_idx` (or just
     construct it directly for origins/levels other than `(ctx.bi, kstar)`).

## Hard-won lessons from this session — do not repeat these

1. **[[feedback-control-against-base-family-before-bug-hunting]] — the single most useful
   technique this session.** When anything about the NEW code looks wrong or surprising
   (infeasibility, a timing/magnitude anomaly, an FD mismatch), before reading your own new code
   for bugs, reproduce the IDENTICAL test against the UNMODIFIED base family under the exact same
   conditions. This resolved four separate apparent "bugs" in minutes each, every single time: a
   bad `nu0` seed, a low-W D20 infeasibility, an FD-conditioning gap at high K, and (critically for
   THIS task) it's what proved the 7x Delta* jump was real economics, not a bug.
2. **D20's `x_free` needs the pivot-elimination (`powered_aspace`) encoding, NOT
   `theta0_up[free_idx]` directly.** `pe = build_pivot_elimination(ctx)`; `Aod_theta_natural =
   ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest]`; `zfree0 = pivot_reduce(reshape(log.
   (Aod_theta_natural), D, Ddest), pe)`; `gp0 = ctx.θ0_up[3+D]`; `x_free_calib = vcat(gp0, vec(exp.
   (pivot_expand(zfree0, pe))))`. (This session's first attempt used the D4 convention naively at
   D20 and got `nStatus=-300`; this is also a pre-existing, separately-documented memory —
   `feedback-campaign-seed-w0-encoding-not-raw-theta-free` — that should have been checked before
   making the mistake once.)
3. **`nu0` must be the theoretical `Gamma(1 - mu*k)` population mean** (via
   `SpecialFunctions.gamma`), never an arbitrary placeholder value copied from an unrelated test
   file — even a value that's harmless in ITS original context can be actively wrong once reused
   somewhere the number's economic meaning actually matters.
4. **Julia top-level `for`-loop scoping gotcha**: assigning to a variable declared OUTSIDE a
   top-level `for` loop, from INSIDE that loop, in a script run via `julia file.jl`/`include`
   (not the REPL), needs an explicit `global` keyword or it throws
   `UndefVarError: ... not defined in local scope` on first use.
5. **A verification script that calls `d_delta_dual_d_eta_origin_fd` (or any FD-sweep helper) and
   ALSO wants to check `octx.fg_lookup_st`'s raw state must do the state-dependent check FIRST.**
   The FD helper re-solves at many perturbed points, leaving shared solver-context objects
   (`core_cf_ref`, `zc_ws` targets) at the LAST perturbed point, not the point you think you're
   checking.
6. Two full D20/W=100k KNITRO jobs launched concurrently by the SAME session died silently with no
   Julia error trace and no dmesg OOM/segfault entry — cause never conclusively identified (the
   user has ~10 OTHER such jobs running fine elsewhere, ruling out a general concurrency/license
   cap), but re-running alone worked cleanly every time. If you hit an unexplained silent process
   death, don't assume a code bug — try running it alone first.
7. K=3/3 costs ~5min/inner-solve at D20/W=100k for OZC-CROSS (~2.8x the base family's ~58s,
   tracking the ~2.8x bigger restriction block) — budget wall-clock accordingly, and always use
   the "check in within ~30-60s" discipline from CLAUDE.md on every background job (this session
   caught two real script bugs — a missing include, and the pivot-encoding mistake — within a
   minute each, instead of burning a 5-minute solve to discover them).

## When you're done

Re-run the same verification shape as last session (D4 first for speed, then D20 at production
scale), PLUS the specific comparison this task exists for: base family with `kstar=2` profiled vs
OZC-CROSS with `kstar=2` profiled, at K_mean=K_pair=3, D20/W=100,000/`:exclude_row` — does Delta*
come back down to a sane multiple of the base family's own (Variant-D-fixed) value? Update
[[ozc-cross-kpair2-grid-build-2026-08-09]] with the outcome, and push a session package to Dropbox
per CLAUDE.md's standing requirement (new subfolder,
`dropbox:Gravity robustness/Analysis/Server Output/ozc_cross_variant_d_<date>/`, don't overwrite
last session's).
