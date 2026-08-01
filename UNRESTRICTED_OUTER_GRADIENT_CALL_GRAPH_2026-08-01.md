# Unrestricted outer evaluator / outer gradient — exact call-graph map (2026-08-01)

Scope: the CURRENT (production/full, non-profiled) `:unrestricted` family outer pipeline, mapped
end-to-end with exact `file:line` references, so a `economic_parameterization=:profiled_destination_scales`
branch can be spliced into the existing evaluator instead of forked into a parallel pipeline.

All paths below are relative to
`/bbkinghome/edav/gravity_robustness/worktrees/diagnostic-profiled-scales-unrestricted-outer-ab-2026-08-01/full_aod_diag/d4_exact/`
unless stated otherwise.

**The single most important entry point** for the whole pipeline (production driver, all of
Sections 1-10 below hang off it) is:

```
run_profile_checkpointed()      c10_d20_production_driver.jl:531
  -> screened_eval()             c10_d20_production_driver.jl:353   (called from cb_F!/cb_G!/cb_newpt!)
       -> evaluate_fullA_screened_ranged()   fast_range_screen.jl:868
```
and the outer-gradient entry point is
```
cb_G!() (closure inside run_profile_checkpointed)   c10_d20_production_driver.jl:872
  -> composite_gradient_at_Cplus()  (default backend is actually :shared, see §6)  lfix_factorized_workspace.jl:354
```

---

## 0. Which driver is "production" for `:unrestricted`

`c10_d20_production_driver.jl` is explicitly documented (header, lines 1-10) as "the file the
coordinating session should use for Section 10's real frontier runs." `run_profile_checkpointed`
(c10_d20_production_driver.jl:531) is the profile-stage driver (fixed `g`/gp, minimize `Delta_dual`
over the A-block only — this is the stage the fixed-theta `:unrestricted` family actually runs in
production). A second, near-identical `run_polish_checkpointed` exists further down the same file
(header comment at c10_d20_production_driver.jl:1012 says "direct extension of
run_profile_checkpointed above") for the polish stage (`g` free too). Everything below focuses on
`run_profile_checkpointed`; `run_polish_checkpointed` reuses the exact same `screened_eval`/
`x_free_from_w`/gradient-backend machinery, just optimizes over `w=[g;zfree]` instead of `zfree`
alone.

---

## 1. Outer coordinate decode (fixed-theta, production driver's own path)

The production driver does **not** use the generic `decode_outer_unified` machinery described in
§2 below for its live KNITRO callbacks — it uses a much smaller, fixed-theta-only inline decoder:

```julia
# c10_d20_production_driver.jl:323
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
```

- `w = [gp; zfree]` where `zfree` (length `D*Ddest-1`) is the KNITRO outer variable vector
  (`n = length(zfree_start)`, c10_d20_production_driver.jl:666).
- `pe::PivotGravityElim` is built ONCE per run via `build_pivot_elimination(ctx)`
  (c10_d20_production_driver.jl:658 — NOT the theta-invariant "cheap" cache variant; this driver
  is fixed-theta, so the plain, non-cached pivot struct from `gravity_elimination.jl` suffices).
- `pivot_expand(zfree, pe)` (gravity_elimination.jl:81-89) reinserts the gravity-eliminated pivot
  cell and reshapes to `(D, Ddest)` in log(Aod_theta) space (`z`); `exp.(...)` converts to
  `Aod_theta` levels; `vcat(gp, Aod_levels)` produces `xf`, the exact shape
  `evaluate_fullA_screened_ranged`/`CS.reconstruct_full` expect (`[gp; Aod_levels]`, matching
  `ctx.m`'s `free_idx` convention for a fixed-theta ctx).

**A separate, more general decoder exists** (not used by the live production driver's KNITRO
callbacks, but reused by several diagnostic/flexible-theta drivers and worth knowing about for a
profiled-parameterization splice):

```
outer_coordinate_layout.jl:130  decode_outer_unified(w, ctx, layout, pgc, xy, gs) -> NamedTuple
outer_coordinate_layout.jl:174  reduce_to_w_unified(theta, gp, logA_full, pgc, xy, layout, gs)
outer_coordinate_layout.jl:194  gradient_transform_unified(gfull_reduced_z, theta, gp_raw, layout, gs)
outer_coordinate_layout.jl:45   struct OuterCoordinateLayout(trade_elasticity_mode, A_coordinate_mode, gp_coordinate_mode)
```
This is a THREE-axis abstraction (`:fixed`/`:flexible` theta × `:legacy_z`/`:powered_aspace` A-coordinate
× `:raw`/`:scaled_log` gp-coordinate) built specifically so new coordinate conventions can be added
as a new `A_coordinate_mode`/axis without touching `gravity_elimination.jl`. **This is the natural
place to add a fourth `A_coordinate_mode` value (e.g. `:profiled_destination_scales`) if the new
parameterization is to be exposed through the SAME generic decoder** rather than only through a
standalone `outer_coordinate_layout_profiled_2026-07-31.jl`-style parallel file (see §2).
`decode_outer_unified` is NOT currently wired into `c10_d20_production_driver.jl`'s live callbacks
at all — only into other drivers (`flexible_theta.jl:222`, `flexible_theta_aspace_production.jl:142,153`).

---

## 2. Gravity-pivot expansion (`gravity_elimination.jl` + the 2026-07-31 profiled-scales layer)

Core file `gravity_elimination.jl` (254 lines) — verified exactly linear-in-z gravity constraint,
two elimination schemes:

- `PivotGravityElim` / `build_pivot_elimination(ctx; μ=...)` (gravity_elimination.jl:53-78):
  argmax-|coefficient| pivot cell chosen once; `pivot_expand`/`pivot_reduce`
  (gravity_elimination.jl:81-92) map `z_free (D*Ddest-1)` ↔ full `z (D,Ddest)`, O(D·Ddest) each,
  re-derives `c`/`g0`/pivot from scratch every call — this is what the production driver uses
  (fixed theta, called once per run to build `pe`, then `pivot_expand` is called every callback).
- `PivotGravityElimCache` / `build_pivot_elimination_cheap(ctx; mu_probe1, mu_probe2)`
  (gravity_elimination.jl:153-224): theta-invariant pivot/other_idx/slope computed ONCE, affine
  `g0(μ)=a+b·μ` fit from two probes; `pivot_expand_cheap`/`pivot_reduce_cheap`
  (gravity_elimination.jl:210-236) are O(1) per θ-move. Used by the flexible-theta path
  (`outer_coordinate_layout.jl`) and by the profiled-scales files below, NOT by
  `run_profile_checkpointed`'s live callbacks.

**The 2026-07-31 profiled-destination-scales layer already exists in this worktree, additive-only,
layered strictly ON TOP of the above (none of it modifies gravity_elimination.jl):**

1. `relative_a_coordinate_2026-07-31.jl` — `AnchorSpec` (one anchor origin per destination),
   `decode_relative_A`/`encode_relative_A` (lines 115, 142): full `z (D,Ddest)` ↔ retained vector
   `r` (length `n_retained = D*Ddest - Ddest`), an ADDITIVE shift `z[o,d] = r[k] + gauge[d]`
   relative to a per-destination `gauge[d]` (built from a GENUINE calibration z-matrix via
   `build_anchor_gauge`, line 100 — **never from the `zfree=0` pivot reference point**, per this
   repo's standing CLAUDE.md warning about `A_od≡1` not being calibration).
2. `gravity_pivot_on_retained_2026-07-31.jl` — composes the gravity pivot with the anchor
   reduction: `build_pivot_elimination_on_retained(ctx, spec, gauge; μ=...)` (line 52) selects its
   pivot from the RETAINED cells only (anchor cells have exactly-zero gravity coefficient
   contribution when shifted, theory doc §2.1(c)); `pivot_expand_on_retained`/`pivot_reduce_on_retained`
   (lines 66, 79) mirror `pivot_expand`/`pivot_reduce`'s contract one level up, on `r` instead of `z`;
   `decode_full_z_on_retained(r_free, pe)` (line 92) is the full composition `r_free -> z (D,Ddest)`.
3. `outer_coordinate_layout_profiled_2026-07-31.jl` — the KNITRO-facing shape: `outer_dim_profiled`
   (line 32) = `1 + n_retained-1` (vs `D*Ddest` full — 361 vs 380 at real D=20); `decode_outer_profiled`
   (line 44) produces `(xf, gp, Aod_levels, z_full)` with `xf` in the **exact same shape**
   `decode_outer_unified`/`screened_eval` already expect, so a screened-evaluation entry point
   needs **no changes** to consume it; `reduce_to_w_profiled` (line 62) is the inverse (seed a
   profiled run from a calibrated/recovered full-A point).

This means the profiled decode/encode/pivot machinery is **already built and tested** (see the
paired `test_*_2026-07-31.jl` files, not read in full for this doc) — the splice point for a new
`economic_parameterization=:profiled_destination_scales` branch is almost certainly a small dispatch
added at `x_free_from_w` (c10_d20_production_driver.jl:323) and inside `screened_eval`'s callers
(cb_F!/cb_G!/cb_newpt!), swapping in `decode_outer_profiled` in place of the current
`x_free_from_w(w, pe)` call, not a rebuild of the decode/pivot layer itself.

---

## 3. Full A reconstruction (working-gauge, and gamma-normalized recovery)

- Working-gauge full A: `x_free_from_w` (§1) produces `xf = [gp; Aod_levels]`; the actual full
  θ-vector reconstruction from `xf` happens inside the evaluator via
  `CS.reconstruct_full(x_free, ctx.m)` (called at oracle_fast.jl:263, compressed_live.jl:518,
  fast_range_screen.jl — `ctx.m`'s own `free_idx`/`fixed_idx` convention, not redefined in this doc's
  scope; `CS` is the `cc_algo` module).
- Gamma-normalized full-A recovery (used for reporting/anchoring, NOT inside the hot per-eval
  path): `recover_full_a_2026-07-31.jl::recover_gamma_normalized_full_A(θ_working, ctx; d_list)`
  (line 81) — per-destination `c[d] = gamma_tilde[d]^(-1/(mu*(sigma-1)))` rescale via
  `build_compressed_factual`'s own `wval`/`denom` fields (rebuilt 2026-07-31 specifically to use the
  no-H `OperatorPsiBundle`/compressed representation, not the legacy dense `moments!`/G path — see
  that file's header for why the earlier dense-G version was wrong).

---

## 4. Inner Δ* evaluation (KNITRO inner solve → outer objective value)

Production per-point evaluator (called from `screened_eval`, always with
`moment_representation = :compressed`):

```
fast_range_screen.jl:868   evaluate_fullA_screened_ranged(x_free, ctx, rsc; ...)
  -> (screens, §9 below, then, on pass:)
  fast_range_screen.jl:572  evaluate_fullA_screened_compressed_with_cf(x_free, θ_full, ctx, cf; ...)
    -> compressed_live.jl:477  evaluate_fullA_fast_compressed(x_free, ctx; ...)
         -> compressed_live.jl:364  inner_loop_internal_compressed(obj, θ_full, ctx)
              -> compressed_live.jl:310  inner_loop_KNITRO_compressed(obj, st::CompressedCBState)
                   -> KNITRO.KN_solve(kc)   (compressed_live.jl:335)
                      callbacks: _callbackEvalFG_inner_compressed! (compressed_live.jl:210)
                                 _callbackEvalH_inner_compressed!  (compressed_live.jl:257)
```

`inner_loop_KNITRO_compressed` returns `(nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)`.
Feasible status codes throughout this codebase are `FEASIBLE_CODES = (0, -100, -101, -103)`
(c10_d20_production_driver.jl:124, mirrored as a local `const` in ~15 other files). On a feasible
status, `evaluate_fullA_fast_compressed` (compressed_live.jl:477) computes `Delta_dual` (either via
the `:operator` or `:dense_reference` verification backend, §5), `gravity_value`, moment/KKT
residuals, `zeta`/`lambda`, and returns a large `NamedTuple` `result` whose `Delta_dual` field is
literally what `cb_F!` (c10_d20_production_driver.jl:805) assigns to `evalResult.obj[1]` — i.e. the
**outer objective value IS `result.Delta_dual`**, set at c10_d20_production_driver.jl:828-829.

The dense (non-compressed) mirror of the same evaluator chain — `oracle.jl:340 evaluate_fullA` /
`oracle_fast.jl:209 evaluate_fullA_fast` — still exists and is used by many diagnostic scripts, but
is explicitly NOT on the production hot path (`moment_representation=:dense` is the default of
`evaluate_fullA_fast` but the production driver always passes `:compressed`, and
`evaluate_fullA_screened_ranged`'s own `:dense` branch at fast_range_screen.jl calls
`evaluate_fullA_fast` directly, bypassing screening's compressed `cf` reuse).

---

## 5. Verified incumbent handling

Two-layer verification:

1. **Per-eval "solved vs verified" classification** (independent of the compressed/operator
   plumbing above): `oracle.jl:290 classify_inner_result(result; tol=DEFAULT_VERIFIED_SUCCESS_TOL)`
   returns one of `@enum InnerResultClass VerifiedSolved ApproximateSolved ExactInfeasible
   ConfirmedNumericalNegative TransientFailure` (oracle.jl:273). Gate: KNITRO status feasible AND
   `primal_dual_gap`, `mean_m_resid`, `max_abs_moment_kkt_resid` all finite and within
   `VerifiedSuccessTolerances` (oracle.jl:249, defaults `primal_dual_gap_tol=1e-3`,
   `mean_m_resid_tol=1e-6`, `max_abs_moment_kkt_resid_tol=1e-3`, `m_min_floor=0.0`) AND `m_min >
   m_min_floor`. `is_verified_success(result) = classify_inner_result(result) == VerifiedSolved`
   (oracle.jl:310). **Only a `VerifiedSolved` result may become the outer incumbent** — the
   production driver's own `cb_F!` gate is explicit:
   `is_new_best = is_verified_success(r) && is_better_profile(...)` (c10_d20_production_driver.jl:849).
   `is_cacheable_result(result) = classify_inner_result(result) in (VerifiedSolved, ExactInfeasible)`
   (oracle.jl:324) — this is the ONLY gate on entering the exact-point cache (§8).

2. **Independent residual recomputation ("operator verification")**, orthogonal to (1) — this is
   what actually PRODUCES the `Delta_dual`/residual fields (1) classifies:
   `operator_verification.jl:286 verify_inner_solution_operator_unrestricted!(zeta, lambda, cf, obj, W)`
   recomputes `r`, `f` (⇒ `Delta_dual = -f`), and `g_lambda` (⇒ `kkt_resid`) from FRESH scratch via
   `economic_forward!`/`economic_transpose!` (not the live FG callback's own `st.arg0`/`st.arg1`),
   then `operator_verification.jl:344 verify_namedtuple_from_operator(ov, obj, W, nStatus)` builds
   the same-shaped `verify` NamedTuple the dense path's tail produces, so `classify_inner_result`
   consumes either backend's output identically. Default backend for unrestricted:
   `UNRESTRICTED_VERIFICATION_BACKEND_DEFAULT = Ref{Symbol}(:operator)` (operator_verification.jl:382,
   flipped from `:dense_reference` on 2026-07-27 after D=4 + real D=20/W=80,000 gates passed both
   backends agreeing bit-for-bit on classification). `:dense_reference` remains available as an
   explicit non-default backend (`verification_backend=:dense_reference` kwarg on
   `evaluate_fullA_fast_compressed`, compressed_live.jl:480).

---

## 6. Custom "C+" outer gradient — exact mechanism (NOT a pure analytic gradient)

**Important correction to the task's framing**: production's "C+" outer gradient is a **hybrid**,
not a single analytic formula:

- `w[1]` (gp) component: a genuine closed-form exact derivative,
  `composite_gradient.jl:64 gamma_component_analytic(cache, base, g)` — derived by hand from
  `cf_contrib_at`'s closed form (the only piece of `L_fix` depending on gp), cross-validated against
  `ForwardDiff.derivative`/central FD in `test_composite_gradient.jl`. This is exact, not a small-h
  approximation (see the derivation comment at composite_gradient.jl:30-45: it uses the
  fixed-dual/envelope-theorem identity `dPsi(q0_s) == base.m_star[s]` at the base point, no
  perturbation needed).
- `w[2:end]` (A-block, i.e. `z_nonpivot`) components: **fixed-dual coordinatewise CENTRAL finite
  differences**, one KNITRO-free "L_fix" evaluation pair per coordinate, using an O(1)-per-probe
  incremental winner-update mechanism (NOT a fresh inner KNITRO solve per probe — that is exactly
  what makes this tractable: "requires exactly ONE inner dual solve per outer iterate... vs
  eval_grad_central_fd's 2·n_free FULL inner re-solves", composite_gradient.jl:18-20).

Production entry point (the `:cplus` backend, currently used only when a caller explicitly passes
`price_cache_backend=:cplus` — see the default-flip note below):
```
lfix_factorized_workspace.jl:354  composite_gradient_at_Cplus(x_free0, ctx, pe, pool, ws; base, threaded, h_mode, bandwidth_cache, ...)
  g[1] = gamma_component_analytic(cache, base, w0[1])              # exact, line 371
  g[k] = a_block_fd_component_Cplus!(tws, cache, ctx, pe, w0, k, h_used[k])   # central FD, line 401
           -> lfix_factorized_workspace.jl:326  a_block_fd_component_Cplus!
                Lp = lfix_incremental_at_Cplus!(ws, cache, ctx, pe, w0, k, w0[k]+h)
                Lm = lfix_incremental_at_Cplus!(ws, cache, ctx, pe, w0, k, w0[k]-h)
                return (Lp - Lm) / (2h)      # central difference (lfix_factorized_workspace.jl:332)
  h_used[k] via select_bandwidth_C (lfix_factorized.jl:326, h0=0.01 default, h_floor=1e-4, h_ceil=0.1)
```
Winner-switch handling: `lfix_incremental_at_Cplus!`'s O(1) tier relies on
`winner_certificate.jl:556 coord_winner_update!(winner_out, ref, ctx, x_free', changed_cells)` — an
incremental winner-matrix update over a SMALL `changed_cells::Vector{(o,d)}` set (the FD-perturbed
cell(s) only), rather than a full winner rescan; `count_winner_flips`/`count_winner_flips_multi_top3`
(composite_gradient.jl, shared_a_gradient.jl:106,163) track how many draws actually flip winner
under the ±h perturbation — this is the "winner-switch/changed-cell correction" the task asked
about: it is a performance/exactness mechanism internal to the FD-probe evaluation (make each probe
O(touched draws) not O(W·D)), not a separate additive correction term applied to the gradient value.

**Default backend is actually `:shared`, not `:cplus`**: `resolve_price_cache_backend`
(c10_d20_production_driver.jl:511, called at c10_d20_production_driver.jl:668) resolves to `:shared`
(⇒ `shared_a_gradient.jl::economic_A_gradient!`, wired at c10_d20_production_driver.jl:913-916) when
neither `use_pooled_gradient` nor `price_cache_backend` is passed — this was flipped 2026-07-27 (see
docstring at c10_d20_production_driver.jl:482-500) from a prior default of `:cplus`. `:shared` is
gated bit-identical to `composite_gradient_at_fast` (the reference kernel `:cplus` also mirrors) at
D=4 and real D=20/W=80,000. `:cplus` remains fully supported, non-default
(`price_cache_backend=:cplus`). **Both `:shared` and `:cplus` use the SAME fixed-dual
central-finite-difference-with-incremental-winner-update mechanism for the A-block** — `:shared`
just uses a persistent `LFixBaseWorkspace`-backed implementation instead of `LFixFactorizedWorkspace`.
Six backends total are valid: `VALID_PRICE_CACHE_BACKENDS = (:buffered, :pooled, :aplus, :cplus,
:kbplus, :shared)` (c10_d20_production_driver.jl:473); all six compute the identical mathematical
gradient (validated bit-identical against each other), differing only in allocation/threading
strategy.

`gradient_transform_unified` (outer_coordinate_layout.jl:194) is the ONE place that would rescale
this raw z-space gradient into a different `A_coordinate_mode`'s coordinates
(`g[2:end] .*= (-theta)` for `:powered_aspace`) — **a `:profiled_destination_scales` A-coordinate
mode would need an analogous chain-rule rescale here** (or inside a new wrapper) mapping
`d(Delta)/d(z_nonpivot)` → `d(Delta)/d(r_free)`, since `decode_relative_A`/`pivot_expand_on_retained`
(§2) are both affine, not identity, maps.

---

## 7. "Finite-difference fallback" — there isn't one; FD central-differencing IS the production A-block gradient

Per §6: this codebase does **not** have a separate "exact analytic gradient, with FD as a
degraded fallback" structure for the A-block. The fixed-dual central finite difference (adaptive
per-coordinate bandwidth `h`, `select_bandwidth_C`/`select_bandwidth`, default range
`h_floor=1e-4` to `h_ceil=0.1`, seed `h0=0.01`) **is** the production A-block gradient computation,
in every backend (`:buffered`/`:pooled`/`:aplus`/`:cplus`/`:kbplus`/`:shared`). It is fast only
because each probe reuses `lfix_incremental_at(...)`'s O(1)-per-changed-cell winner update instead
of a full inner KNITRO re-solve — not because it is a rare fallback path.

Genuine finite-difference **cross-checks against this production gradient** (used only in
diagnostics/tests, never in the live outer loop) include: `c8_nestedw_gradcheck.jl` (central FD of
the OPTIMIZED-value objective, i.e. re-solving the inner problem at each probe — the "slow ground
truth" this whole FD-secant machinery is designed to avoid paying for on every outer iterate),
`h_sweep.jl` (bandwidth-sensitivity sweep), `directional_sign_audit.jl` (sign-consistency audit).
`derivative_methods.jl:27` documents "the cheap fixed_dual_L gradient (finite-differenced)" — same
terminology used here.

---

## 8. Cache keys

Two independent key structs, both requiring `ctx_fingerprint` (AUD-08 remediation) so no two
distinct contexts can alias:

- `oracle.jl:178 struct FullAEvalKey(x_free::Vector{Float64}, δ::Float64, find_smallest::Bool,
  inner_loop_opt::String, mode::Symbol, ctx_fingerprint::String)` — equality/hash both defined
  (oracle.jl:186-189) over ALL SIX fields. `x_free` is the exact-shape `xf` vector from §1/§3 (i.e.
  `[gp; Aod_levels]` for a fixed-theta ctx) — **NOT the reduced outer vector `w`/`zfree`**. This
  means a profiled-parameterization's cache key does NOT need to change shape at all as long as it
  keeps decoding to the same `xf` shape (which `decode_outer_profiled`, §2, already guarantees) —
  two different outer coordinate systems that decode to the identical `xf` are, correctly,
  cache-equivalent.
- `oracle.jl:111 context_fingerprint(ctx)::String` — versioned SHA-256 over draws (`ctx.draw_meta`
  checksums), shapes (D, W, D_dest), `row_idx`/destination-sample tag, fixed trade data
  (`wHat`/`L`/`LPrime`/`τ`/`τPrime`), CM config, loaded KNITRO release, and option-file CONTENTS
  (not just path). Memoized per-`ctx.U` identity in `_CTX_FINGERPRINT_CACHE::IdDict` (oracle.jl:95),
  lock-guarded (oracle.jl:96). **Explicitly does NOT include the outer coordinate system/layout at
  all** — a profiled-scales ctx and a full-A ctx built from the SAME draws/trade-data/option-file
  would currently fingerprint IDENTICALLY. If the new parameterization changes anything
  `context_fingerprint` doesn't already cover but that changes the mathematical answer (unlikely,
  since §2's layer is purely a coordinate change on top of the same `ctx`) it would need a new field
  written into the digest buffer at oracle.jl:111-155; more likely, no change is needed here at all
  since `FullAEvalKey`'s `x_free` (the decoded `xf`) already disambiguates by construction.
- Storage: `oracle.jl:214 struct SafeExactCache{K}(d::Dict{K,NamedTuple}, lock::ReentrantLock)`,
  generic in the key type; `oracle.jl:222-228 _cache_lookup`/`_cache_store!` dispatch over
  `Nothing`/`Dict`/`SafeExactCache`. `CrossDeltaExactCache` (cross_delta_cache.jl, not read in full
  for this doc) is a second key/cache pairing threaded through the SAME `_cache_lookup`/`_cache_store!`
  generic dispatch — confirms this mechanism is already designed for more than one key type.

---

## 9. Screens

Invoked, in this exact fixed order, inside `evaluate_fullA_screened_ranged`
(fast_range_screen.jl:868-1012), called from every `screened_eval` (§0's entry point):

1. **Pairwise never-wins certificate** (fast_range_screen.jl:893-903) — `pairwise_certificate(a, pc,
   Pmat)` where `pc = pairwise` kwarg, falling back to `ctx.pairwise` (precomputed once at ctx
   construction, `context_real_d20.jl`/`draw_design.jl`) or, failing that, a fresh
   `precompute_pairwise_M(ctx)`. **Always runs — no kwarg disables it** (cheapest screen, no toggle
   provided).
2. **Pre-winner envelope certificate** (fast_range_screen.jl:906-916) — only if
   `rsc.envelope !== nothing`; `RangedScreenContext.envelope` is built once via
   `build_ranged_screen_context(ctx)` (c10_d20_production_driver.jl:667) and is `nothing` whenever
   the ctx is `EnvelopeUnsupportedContext` — **explicitly disabled by default under
   `destination_sample=:exclude_row`** (the production default — see the header note at
   c10_d20_production_driver.jl:606-608: "The pre-winner envelope screen... remains disabled under
   :exclude_row... zero organic hit rate historically"). No separate runtime kwarg; disabling is a
   ctx-construction-time property.
3. **Witness screen** (fast_range_screen.jl:919-935) — gated by `use_witness::Bool = false`
   (fast_range_screen.jl signature default — **disabled by default**); the production driver passes
   `use_witness = ctx.witness !== nothing` explicitly (c10_d20_production_driver.jl:425), so it is
   effectively on whenever `ctx.witness` was built (`build_extreme_draw_witness`, done at ctx
   construction).
4. **Fused zero-winner + winning-range scan** (fast_range_screen.jl:938-965) —
   `screen_hard_winners_ranged` (envelope available) or `screen_hard_winners` (envelope
   unavailable/legacy) — always runs, no disable kwarg (this IS the winner computation the inner
   solve needs anyway; screening is fused into it, not a separate pass).
5. **General range-screen safety net** (fast_range_screen.jl:988-1000) — gated by
   `use_general_range_safety_net::Bool = true` (fast_range_screen.jl signature default — **on by
   default**, production driver never overrides it, so it always runs); reuses the ALREADY-BUILT
   `cf::CompressedFactual` from step 4, `range_screen_standalone(cf; safety_mult=50.0)`.
6. Only if all five pass: the real inner KNITRO solve (§4).

**To disable screens for a profiled-parameterization A/B test**: pass
`use_witness=false` (already the function default), `use_general_range_safety_net=false`
explicitly, and ensure `rsc.envelope === nothing` (either by using `destination_sample=:exclude_row`,
already the production default, or by constructing an unsupported-envelope ctx) — the pairwise and
fused-winner-scan screens (1 and 4) cannot be disabled via any existing kwarg; they would need a new
opt-out added to `evaluate_fullA_screened_ranged`'s signature if a true "screens fully off" A/B arm
is needed.

Per-driver screen-rejection bookkeeping: `ScreenCounters` struct (c10_d20_production_driver.jl:338),
populated inside `screened_eval` (c10_d20_production_driver.jl:442-464) by pattern-matching
`screen_meta.screen_status` against the six status symbols the functions above return
(`:pairwise_certified_infeasible`, `:witness_certified_infeasible`, `:winner_scan_infeasible`,
`:EXACT_INFEASIBLE_PREWINNER_ENVELOPE`, `:EXACT_INFEASIBLE_WINNING_RANGE`,
`:EXACT_INFEASIBLE_MOMENT_RANGE`).

---

## 10. Outer KNITRO configuration (`:unrestricted` family)

Outer solver setup, `run_profile_checkpointed` (c10_d20_production_driver.jl:753-766):

```julia
kc = KNITRO.KN_new()
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))   # default hessopt_tag="sr1"
KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)      # default 900.0s
KNITRO.KN_set_param_by_name(kc, "maxit", maxit_override === nothing ? 1_000_000 : maxit_override)
if pin_outer_algorithm
    set_production_outer_algorithm!(kc)   # opt-in only, see below
else
    KNITRO.KN_set_param_by_name(kc, "algorithm", 3)   # Active-Set SLQP, HARDCODED default
end
xIndices = KNITRO.KN_add_vars(kc, n)
z_halfwidth = 30.0
KNITRO.KN_set_var_lobnds_all(kc, zfree_start .- z_halfwidth)
KNITRO.KN_set_var_upbnds_all(kc, zfree_start .+ z_halfwidth)
KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)
```
`csw_outer_wallclock_sr1.opt` sets `hessopt 3` (dense quasi-Newton **SR1**, confirmed at
`csw_outer_wallclock_sr1.opt:309`) and `algorithm auto` (the `.opt` file itself, overridden by the
explicit `KN_set_param_by_name(kc, "algorithm", 3)` call above unless `pin_outer_algorithm=true`).
There is **no "Direct" outer algorithm in current production** — `algorithm=3` is Active-Set SLQP,
not Interior/Direct (algorithm codes per the `.opt` file comment: `auto=0, direct=1, cg=2,
active=3, sqp=4, multi=5`). "Direct+SR1" only appears as the historically-observed AUTO resolution
(`knitro_outer_algorithm.jl:16-21`: `auto` resolves to Active-Set/CG for unrestricted's own
unconstrained profile formulation but Interior-Point/Barrier-Direct for CM's constrained
formulation — a finding from the 2026-07-25 audit, not something this driver currently runs, since
it hardcodes `algorithm=3` rather than leaving it `auto`).

Opt-in alternative (`pin_outer_algorithm=true`, default `false`, never applied unless explicitly
requested — `knitro_outer_algorithm.jl`):
```julia
const PRODUCTION_OUTER_ALGORITHM = 2   # Interior/CG
const PRODUCTION_OUTER_HESSOPT = 6     # L-BFGS
function set_production_outer_algorithm!(kc)   # knitro_outer_algorithm.jl:41
    KNITRO.KN_set_param_by_name(kc, "algorithm", PRODUCTION_OUTER_ALGORITHM)
    KNITRO.KN_set_param_by_name(kc, "hessopt", PRODUCTION_OUTER_HESSOPT)
end
```
This pairing (Interior/CG + L-BFGS) beat the `auto`(→Direct)+SR1 default by +9.8% relative kappa in
a controlled 2026-07-23 experiment (memory `outer-strategy-delta2-experiment-2026-07-23`), but is
NOT the production default (`pin_outer_algorithm` defaults false everywhere) — a single controlled
result was judged insufficient to flip every campaign's default (see
`knitro_outer_algorithm.jl` module docstring, lines 1-30).

Callback registration (c10_d20_production_driver.jl:953-955):
```julia
cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
KNITRO.KN_set_cb_grad(kc, cb, cb_G!)
KNITRO.KN_set_newpt_callback(kc, cb_newpt!)
```
`cb_F!` (c10_d20_production_driver.jl:805): decodes `w`→`xf` (§1), calls `screened_eval` warm, retries
cold on infeasible, sets `evalResult.obj[1] = Δ = r.Delta_dual`, updates the incumbent (§5 gate),
checkpoints on new-best or wall-interval. `cb_G!` (c10_d20_production_driver.jl:872): reuses
`last_F_state[]`'s `BaseDualState` when the just-requested point exactly matches `cb_F!`'s last call
(avoiding a redundant inner solve), else recomputes via `screened_eval`, then dispatches to one of
the six gradient backends (§6) based on `resolved_backend`, writes
`evalResult.objGrad .= gfull[2:end]` (dropping index 1, the gp-vs-gp identity — `gfull[1]` is the gp
component, not part of the A-block-only `zfree` gradient this stage's outer variables cover).
`cb_newpt!` (c10_d20_production_driver.jl:929): checkpoints every accepted outer iterate, reusing
`last_F_state[]` when the accepted point matches the last `cb_F!` call (no extra inner solve).

---

## 11. Checkpoint/result export

Schema: `D20CheckpointV4` (c10_d20_production_driver.jl:187-220), `CHECKPOINT_SCHEMA = 4`
(c10_d20_production_driver.jl:223). Fields include `g`, `zfree`, `logA_full` (the FULL D×Ddest
log-A matrix, i.e. already gravity-pivot-EXPANDED — not the reduced `zfree`, so a resumed/replayed
profiled-scales run could reconstruct its own `r_free`/`w_profiled` from this same `logA_full` via
`reduce_to_w_profiled`, §2), `dual_warm_start` (`ctx.obj.x`), `bandwidth_cache`, `best_feasible`,
`n_eval`/`knitro_iter`/`wall_elapsed`, `checkpoint_reason::Symbol`, `screen_counts::NamedTuple`,
four `verify_*` cold-recompute fields (`Delta_dual`/`gravity_value`/`max_abs_moment_kkt_resid`/
`moment_resid_norm`), `draw_design`/`draw_checksum_uniform`/`draw_checksum_transformed`,
`knitro_version`, `destination_sample`/`row_idx`/`D_dest`. Does NOT capture KNITRO's internal
SR1/BFGS/L-BFGS quasi-Newton Hessian approximation state (documented limitation,
c10_d20_production_driver.jl:41-50 — the KNITRO.jl API has no accessor for it).

Write path: `save_checkpoint(path, ckpt)` (c10_d20_production_driver.jl:257) — atomic
serialize-to-`.tmp`-then-`mv`. Read path: `load_checkpoint(path)` (c10_d20_production_driver.jl:263)
— hard schema check (`ckpt.schema == CHECKPOINT_SCHEMA`), no migration path from older schemas
("start a fresh run"). `guard_checkpoint_path` (c10_d20_production_driver.jl:297) refuses to
overwrite an existing checkpoint file built under a different `draw_design`/checksum.

Trigger sites inside `run_profile_checkpointed`, all via the local `do_checkpoint(reason, w, r)`
closure (c10_d20_production_driver.jl:788): `:new_best` (cb_F!, on `is_new_best`),
`:wall_interval` (cb_F!, every `checkpoint_interval_s`, default 90s), `:iteration` (cb_newpt!, every
accepted outer iterate), `:stage_complete`/`:stage_complete_unverified` (end of run, gated by
`is_verified_success(r_final)`, AUD-10 — an unverified terminal point is flagged, never silently
labeled `:stage_complete`). `do_checkpoint` always writes `"$(label)_latest.jls"`, and additionally
a permanently-retained `"$(label)_$(reason)_neval$(n_eval[]).jls"` for the four "important" reasons
(`:new_best`, `:stage_complete`, `:stage_complete_unverified`) — c10_d20_production_driver.jl:800.

A new parameterization would extend `D20CheckpointV4` (or version it — schema bump, `CHECKPOINT_SCHEMA
= 5`) with whatever additional profiled-coordinate state (`AnchorSpec`, `gauge`, `pe_on_retained`,
`w_profiled`) it needs to resume identically; `logA_full` alone is enough to REPRODUCE any profiled
run's state from a full-formulation checkpoint (§2's `reduce_to_w_profiled` is exactly this
reduction), so cross-A/B resume/comparison should already be possible without a schema change if the
new driver reads an old full-formulation checkpoint's `logA_full` as its own seed.

---

## Raw `rg` search results not otherwise covered above

`rg -n "evaluate_fullA|outer.*gradient|Cplus|C\\+|c_plus"` — dominated by diagnostic/benchmark
scripts calling `evaluate_fullA`/`evaluate_fullA_fast`/`evaluate_fullA_screened*` (all covered
above); the only NEW production fact this surfaced was the `:cplus`→`:shared` default-flip
provenance already folded into §6.

`rg -n "finite.*difference|fd_gradient|gradient_task"` — no function literally named `fd_gradient`
or `gradient_task` exists anywhere in this directory; every hit is either the production A-block FD
mechanism itself (§6/§7) or a diagnostic/gate script's own ground-truth FD check (`c8_nestedw_gradcheck.jl`,
`h_sweep.jl`, `directional_sign_audit.jl`, `phaseA_*_revalidation.jl`, `cm_originzc_production.jl:298-302`
which documents the SAME "reoptimized central finite difference" ground-truth concept for a different
family). **No `gradient_task` concept exists in this codebase** — the task prompt's search term did
not match anything real; do not assume one needs to be added.

`rg -n "pivot_expand|pivot_reduce|other_idx|free_idx"` — fully covered in §1/§2; the only additional
fact is that `other_idx`/`free_idx` are used under TWO different names for closely related but
distinct concepts: `gravity_elimination.jl`'s `other_idx` (the D·Ddest-1 non-pivot A-cell linear
indices) vs. `ctx.m`'s own `free_idx`/`fixed_idx` (which outer-vector-vs-full-θ slots are free —
`cc_algo`'s `ModelLayout`-style convention, not redefined by any file read for this doc). Do not
conflate the two when wiring a new coordinate mode.

`rg -n "Direct|SR1|outer.*KNITRO"` — fully covered in §10; no other production fact surfaced (all
other "direct"/"directly" hits are the English word, not the KNITRO algorithm).

`rg -n "changed_cell|winner.*gradient|winner.*switch"` — fully covered in §6;
`winner_certificate.jl:556 coord_winner_update!(winner_out, ref, ctx, x_free', changed_cells)` is the
exact incremental-update function, `changed_cells::Vector{Tuple{Int,Int}}` (winner_certificate.jl:559).

`rg -n "infeasibility_screen|range_screen|pairwise"` — fully covered in §9; additionally confirms
`infeasibility_screen.jl` is included transitively by `draw_design.jl`
(c10_d20_production_driver.jl:83 comment) and directly by `context_real_d20.jl:23` (source of
`precompute_pairwise_M`/`build_extreme_draw_witness`, the functions that populate `ctx.pairwise`/
`ctx.witness` once at context-construction time, reused unchanged by every `screened_eval` call
rather than rebuilt per-callback).
