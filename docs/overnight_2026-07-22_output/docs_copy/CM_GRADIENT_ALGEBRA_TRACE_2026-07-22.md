# Trusted CM gradient/value call-path trace and algebraic decomposition — 2026-07-22

Traced by direct code read at base commit `22683016b5a927d4952e52049145ad8d1f5a2b87`
(`cm-production-ready-2026-07-22-r3`), before any code in this branch was written. No claim below is
assumed; every function/line cited was opened and read.

## 1. Call path, launcher to KNITRO Jacobian

```
scripts/cm_production_supervisor.sh
  -> full_aod_diag/d4_exact/cm_production_stage_runner.jl   (per (chain,delta) stage process)
       -> cm_checkpoint.jl :: run_cm_upper_checkpointed(...)     [driver, line 143]
            pcx = build_cm_production_context(ctx, CS; L, contrasts, probs)   [cm_production_bundle.jl:54]
                -> aug = build_cm_augmented_obj(ctx, CS; L, contrasts, probs)   (CM-augmented obj_cm,
                   cumulative-basis thresholds aug.z, CM column layout)
                -> ctx_cm = merge(ctx, (obj = obj_cm,))
                -> bins   = cm_bin_indices_for(ctx, aug)      = compute_bin_indices(ctx.U, aug.z)
                -> cctx   = build_cm_bin_ctx(ctx, aug)        (Architecture-C bin/scratch context)
            cb_F!(...)  -- KNITRO objective+constraint callback
                -> (_, base, verify) = cm_production_value_verified(xf, pcx)      [cm_production_bundle.jl:231]
                     -> base, verify = archC_verified_state(xf, pcx.ctx_cm, pcx.cctx)  [:159]
                          -> inner_loop_internal_archgeneric(obj, θ_full0; hess_cb_builder=archC_hess_cb_builder(cctx))
                             (Architecture-C structured Hessian for the INNER dual solve only --
                              orthogonal to which OUTER gradient backend is used)
                          -> explicit recompute: obj(inner_x, constr=cbuf) at the converged (ζ*,λ*)
                             -> Delta_dual = cbuf[1]/1e10   [the CANONICAL value, F1-fixed]
                Δ = verify.Delta_dual                                              [cm_checkpoint.jl:293]
                evalResult.c[1] = Δ    (constraint value KNITRO sees)
            cb_G!(...)  -- KNITRO gradient callback
                -> gfull, meta = cm_production_gradient(xf, pcx, ctx, pe; base=..., ...)   [cm_production_bundle.jl:202]
                     base  = base ?? archC_base_state(xf, pcx.ctx_cm, pcx.cctx)     [:123] (same inner solve as cb_F!'s;
                              reused from cb_F!'s last_F_state[] when the point matches exactly, else recomputed)
                     cache = build_lfix_base_cache_cm(xf, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)  [lfix_cm_aware.jl:118]
                     return composite_gradient_at_fast(xf, pcx.ctx_cm, pe; base=base, cache=cache, ...)  [composite_gradient_fast.jl:111]
                evalResult.jac .= gfull    (Jacobian row KNITRO receives; jacIndexCons all point at
                                            the single divergence constraint, jacIndexVars = xIndices)
```

## 2. The canonical value: `Delta_dual`

`Delta_dual = -(mean(Psi(q*)) + zeta*) == cbuf[1]/1e10` (`cm_production_bundle.jl:87-108`,
`archC_verified_state` recompute, `cm_production_bundle.jl:170-176`). Computed by an explicit
`obj(inner_x, constr=...)` call at the ALREADY-CONVERGED `(zeta*, lambda*)`, never trusted from
KNITRO's last FG callback state directly (`archC_base_state` does trust `obj.arg1` for the CHEAP
`base`-only path every existing caller uses; `archC_verified_state` does an independent recompute
specifically because "an independent verification check should not rely on the same assumption it
exists to catch a violation of" — direct quote, `cm_production_bundle.jl:145-147`). This is the value
`cb_F!` reports to KNITRO as the constraint (`Δ`), and the value every gradient-check test below must
match, NOT `-zeta*` alone (F1: `-zeta*` omits `mean(Psi(q*))`, nonzero whenever any recovered weight
`m*` exceeds `e` in the divergence's quadratic branch).

## 3. The trusted gradient: algebraic decomposition

The outer gradient is a **fixed-dual finite-difference secant** of an auxiliary scalar `L_fix`
(`lfix_from_q`, `three_way_derivatives.jl`/`lfix_incremental.jl`), NOT a re-optimized total
derivative of `Delta_dual` itself — this is a pre-existing, already-accepted approximation (AUD-05)
common to every backend (Reference, A+, B, C, C+, KB+) on the UNRESTRICTED path; the CM extension
inherits it unchanged. `L_fix` is defined per-draw via a base-point dual scalar

```
q_s(θ) = -ζ* - λ_G*'G_s(θ) - λ_C*'C_s
```

with `(ζ*, λ_G*, λ_C*)` the CONVERGED CM-augmented dual solution held FIXED across every outer
coordinate probe (the "fixed-dual" approximation), `G_s(θ)` the θ-DEPENDENT economic (winner/CES)
moment columns, and `C_s` the θ-INDEPENDENT common-marginals block (a function of the fixed U-draws
and fixed quantile cutpoints only — confirmed directly from `wrap_moments_with_cm`,
`lfix_cm_aware.jl:5-16`: the CM block is spliced into `G` verbatim from a precomputed matrix, never
touched by `core_moments!`'s θ-dependent computation).

Because `λ_C*'C_s` does not depend on `θ`, it is a CONSTANT across every coordinate probe at a fixed
base dual point — the entire CM-specific cost is a ONE-TIME O(W·D) computation
(`cm_fixed_contribution`, `lfix_cm_aware.jl:83-95`, via the cumulative-suffix-sum lookup identity in
`cm_lookup_kernels.jl`'s `CMLookupState`), folded into the cache's `q0` once:

```
q0_CM[s] = q0_economic[s] - (λ_C*'C_s)          (with_q0, lfix_cm_aware.jl:47-50)
```

`build_lfix_base_cache_cm` (`lfix_cm_aware.jl:118-123`) is literally:

```
cache0      = build_lfix_base_cache(x_free0, ctx_cm, base)     # UNCHANGED Reference economic-block builder
cm_contrib0 = cm_fixed_contribution(base, ctx, aug, bins)      # ONE-TIME O(W*D), theta-independent
return with_q0(cache0, cache0.q0 .- cm_contrib0)
```

`build_lfix_base_cache` itself only ever indexes `base.λstar[1:D^2]` (the economic core columns) and
the counterfactual-column tail check `oci-1 >= D^2+1` (always true once CM columns are appended) — so
calling it UNCHANGED against `ctx_cm` silently and correctly ignores the CM tail of `λstar`, which is
exactly the piece `cm_fixed_contribution` adds back in. This is the precise sense in which "the
CM-specific work is additive, one-shot, and separable from the economic block": every downstream
per-coordinate consumer (`lfix_incremental_at`, `dest_contrib_*`, `gamma_component_analytic`,
`select_bandwidth`) reads `q0` ONLY as "the current base-point dual scalar" and never re-derives it
from `contrib0`/`cf_contrib0` alone, so swapping in a CM-augmented `q0` is sufficient — no other code
needs to know CM exists.

`composite_gradient_at_fast_cm` (`lfix_cm_aware.jl:142-150`) is a thin wrapper: build (or accept) the
CM-augmented cache, then delegate 100% of per-coordinate work (bandwidth selection, incremental FD,
threading, gamma-component) to the UNCHANGED `composite_gradient_at_fast` via its pre-existing `cache=`
kwarg (the same mechanism `base=` already uses).

## 4. What is identical to the unrestricted economic block, what is CM-specific

| Piece | Identical to unrestricted Reference/C+ | CM-specific |
|---|---|---|
| Winner/runner-up/third-place ranking, `contrib0` (economic θ-dependent block) | YES — `build_lfix_base_cache`/`build_lfix_base_cache_C` called UNCHANGED against `ctx_cm` | — |
| Per-coordinate incremental FD (`lfix_incremental_at*`, top-3 tiers, bandwidth selection) | YES — operates on `q0`/`contrib0` generically, never touches the CM tail of `λstar` | — |
| Gamma-component (`gamma_component_analytic`) | YES — reads `cache.λ_cf`/`cache.cf_contrib0`, CM-agnostic | — |
| `q0`'s CM correction (`λ_C*'C_s`, one-shot, O(W·D)) | — | YES — `cm_fixed_contribution` |
| Inner dual solve producing `(ζ*, λ*)` (Architecture-C Hessian) | shared with `archC_base_state`/`archC_verified_state` (same function used regardless of OUTER gradient backend) | orthogonal axis, not part of this decomposition |

## 5. Decomposition validity for a C+ (factorized) backend

`build_lfix_base_cache_C`/`build_lfix_base_cache_C!` (`lfix_factorized.jl`, `lfix_factorized_workspace.jl`)
build a *different* struct (`LFixBaseCacheC`, O(W·D) `WinnerRefCache`-backed, no dense W×D×D price/pTσ
tensor) but via THE SAME indexing discipline: `λstar[1:D^2]`
(`lfix_factorized.jl:98-105`, `CONST_d` loop) and the identical counterfactual tail check
(`lfix_factorized.jl:120-121`, `d1_cf = D^2+1`). `LFixBaseCacheC` also carries its own `q0::Vector{Float64}`
field (`lfix_factorized.jl:72`) built by the identical formula
`q0[s] = -ζ* - sum(contrib0[s,:]) - cf_contrib0[s]` (`lfix_factorized.jl:130`,
`lfix_factorized_workspace.jl:182-185`). Consequently the SAME derivation in §3 applies verbatim with
`build_lfix_base_cache` replaced by `build_lfix_base_cache_C`/`build_lfix_base_cache_C!`: calling it
UNCHANGED against `ctx_cm` produces a `q0` that omits `λ_C*'C_s` for exactly the same structural
reason (it never reads past `λstar[D^2+1]` except the single scalar `λ_cf`), and the SAME
`cm_fixed_contribution` (unmodified, backend-agnostic — it only reads `base.λstar` and `aug`/`bins`,
never the caller's choice of price-cache backend) can be folded into that `q0` via an analogous
field-generic `with_q0` for `LFixBaseCacheC`.

**Conclusion**: the CM-aware C+ backend is `build_lfix_base_cache_C[!]` (UNCHANGED) + the SAME
`cm_fixed_contribution` (UNCHANGED, shared with the Reference CM path) folded into `q0` via a new
`with_q0`-twin for `LFixBaseCacheC`, then delegated to a C+-specific per-coordinate driver (the
existing `composite_gradient_at_Cplus` builds its OWN cache internally with no `cache=` override
point, so a new thin driver, structurally identical to `composite_gradient_at_Cplus` but accepting a
pre-built `LFixBaseCacheC`, is required — see `lfix_cm_cplus.jl`). This was DERIVED from the existing
code above, not assumed; the implementation in `lfix_cm_cplus.jl` is line-for-line traceable back to
this section.

## 6. Sign convention / Jacobian anchor (for §2.4 directional gates)

`evalResult.jac .= gfull` directly (`cm_checkpoint.jl:328`, no negation), and `gfull[k] =
a_block_fd_component(...) = (Lp - Lm)/(2h)` with `Lp = lfix_incremental_at(...)` evaluated at
`w0[k]+h` (`composite_gradient_fast.jl`/`lfix_incremental.jl`). `L_fix`'s own sign matches
`Delta_dual` (`lfix_from_q(q, ζ*) = -(mean(Psi(q))+ζ*)`, same formula as §2) — so a central-FD secant
of the independently-reoptimized `Delta_dual` (`cm_production_value_verified(...).verify.Delta_dual`,
NOT `-zeta*`) at `w0[k]±h` is the correctly-signed, directly-comparable ground truth for §2.4's
directional gates, anchored to the SAME `evalResult.jac` sign KNITRO consumes. This guards against the
previously-discovered diagnostic sign reversal (memory: full-A winner-boundary derivative bug family)
by construction, not by convention alone.
