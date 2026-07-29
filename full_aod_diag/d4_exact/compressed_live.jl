# ============================================================================
# Continuation 8, workstream 2: LIVE wiring of the compressed winner-form
# representation (compressed_moments.jl / compressed_cc_inner.jl, both
# validated to machine precision as standalone bundles by continuation 7)
# into the actual inner-dual KNITRO solve used by evaluate_fullA_fast
# (oracle_fast.jl).
#
# MODE-FLAG API: `evaluate_fullA_fast(x_free, ctx; moment_representation=:dense
# or :compressed, ...)`. Default `:dense` -- zero behavior change for any
# existing caller. This file is included AFTER oracle_fast.jl (it reuses
# oracle_fast.jl's `InnerCallCounters`/`_INNER_CALL_COUNTERS`, and
# oracle_fast.jl's `evaluate_fullA_fast` forward-references
# `evaluate_fullA_fast_compressed`, resolved at CALL time, not include time --
# so this file must be `include`d before `:compressed` mode is ever invoked,
# but oracle_fast.jl itself needs no reordering).
#
# WHAT IS ACTUALLY COMPRESSED: only the inner-dual FG callback (objective +
# gradient w.r.t. (zeta,lambda)), called MANY times per inner KNITRO solve --
# this is where `compressed_cc_value_grad` (O(W*D)) replaces `Q`'s dense
# BLAS.gemv! (O(W*D^2)) on every call, the dominant realized saving (see
# report for the measured n_fg_calls multiplier). The Hessian callback and
# all POST-inner-solve bookkeeping (primal-weight recovery, moment residual,
# KKT residual, gravity_raw) are served from a LAZILY-MATERIALIZED dense G
# (built once per outer point from the already-computed compressed winner/
# value arrays, not from scratch) -- see the "HESSIAN-CALLBACK ADAPTER
# DECISION" note in compressed_cc_inner.jl for why, and
# docs/compressed_live_integration_report.md for the exactness/tolerance
# implications and the measured speedup breakdown.
#
# FALLBACK: `build_compressed_factual` throws `TiedWinnerError` (reused as-is
# from lfix_incremental.jl, NOT reimplemented -- ties are resolved IDENTICALLY
# to the dense path because both use hFunction!/MinInd!'s exact `<=`-min
# convention) on any exact price tie. `evaluate_fullA_fast_compressed` catches
# it, increments `COMPRESSED_FALLBACK_COUNT`, logs via `@warn`, and re-dispatches
# to the ORDINARY dense `evaluate_fullA_fast` for that one point -- never
# silent, never a different tie resolution.
# ============================================================================

# port/shared-winner-pair-core-hessian-production-2026-07-25: the shared exact
# core-Hessian backend (replaces this file's own lazy-dense-materialize +
# CS.hessian! path below -- see core_exact_hessian.jl header for the full
# rationale). Self-include-guarded, this codebase's own convention.
isdefined(Main, :fill_core_hessian_upper!) || include(joinpath(@__DIR__, "core_exact_hessian.jl"))

# Shared economic moment-state builder (2026-07-27 task): defensive self-include, this codebase's
# own established convention (matches cm_production_bundle.jl's identical guard) -- needed for
# `build_economic_moment_state!`/`CompressedFactualWorkspace`, used below by
# `inner_loop_internal_compressed` (fixed this task: previously called the always-allocating
# `build_compressed_factual` directly on EVERY inner solve, even when the production driver had
# already attached a `ctx.cf_workspace` -- see ALLOCATING_COMPRESSED_FACTUAL_CALLSITE_AUDIT_2026-07-27.md).
isdefined(Main, :cf_build) || include(joinpath(@__DIR__, "compressed_factual_buffer_reuse.jl"))
isdefined(Main, :verify_inner_solution_operator_unrestricted!) || include(joinpath(@__DIR__, "operator_verification.jl"))   # verification-defaults task (2026-07-27): evaluate_fullA_fast_compressed's :operator verification backend below
isdefined(Main, :HessianWeightCache) || include(joinpath(@__DIR__, "operator_hessian_weights.jl"))   # no-moments/no-composite-G task (2026-07-28): shared cache/prep, see below
isdefined(Main, :OperatorPsiBundle) || include(joinpath(@__DIR__, "operator_psi_bundle.jl"))   # true no-H operator bundle (2026-07-28 continuation): load-bearing for inner_loop_internal_compressed's dispatch below

"Resolved backend/workers/storage for the UNRESTRICTED family's core Hessian -- read by `_callbackEvalH_inner_compressed!` and by `resolve_unrestricted_manifest` so the two can never silently diverge. Production default is the validated destination-pair-owned parallel kernel; set to :dense_reference for anti-regression / emergency-revert comparisons (see task §5). Worker count defaults via `resolve_core_hessian_workers_default()` (2026-07-25 final gate): 20 when >=20 Julia threads are available (measured 13-20% faster than 10, not a tie), else 10, else the bounded available count."
const UNRESTRICTED_CORE_HESSIAN_BACKEND = Ref{Symbol}(:exact_winner_pair_parallel)
const UNRESTRICTED_CORE_HESSIAN_WORKERS = Ref{Int}(resolve_core_hessian_workers_default())
const UNRESTRICTED_CORE_HESSIAN_STORAGE = Ref{Symbol}(:full_stride)

# ---- fallback counter (Ref{Int}, per the task brief's explicit requirement) ----
const COMPRESSED_FALLBACK_COUNT = Ref(0)

"Reset the compressed-mode dense-fallback counter (call at the start of a fresh run/benchmark)."
reset_compressed_fallback_count!() = (COMPRESSED_FALLBACK_COUNT[] = 0)

# ============================================================================
# Gravity-moment column (index d == outer_constr_index == oci for this ctx,
# per context.jl's own `@assert obj.outer_constr_index == obj.d`): the ONE
# column of G beyond the inner-dual block (1:oci-1) that downstream
# post-processing (constr recovery / benchmark_unweighted_moment_mean) still needs.
#
# NOT part of the compressed winner-form representation -- and does not need
# to be: for UoModel==1 (this ctx), `moments/newGravityMoment!.jl`'s own
# UoModel==1 branch writes a SINGLE DRAW-INDEPENDENT SCALAR to every row
# (`@. G[:, end] = sumGrav`, a two-way-demeaned O(D^2) computation, no draws
# loop at all) -- there is nothing to compress; it was never O(W*D^2). This
# calls the EXISTING production `newGravityMoment!` function on a throwaway
# 1-row buffer (not re-derived) to get the identical raw scalar, then applies
# the identical post-processing formula (SW * nrm, gdiv==1 since column index
# > D^2+1, so it is not divided by gammafac; usePMM subtracted if enabled)
# used by EK_moments_gammanorm_directgp! for every other column.
# ============================================================================

"AodPow (D x D_dest), the level->power transform hFunction!/hFunctionCounter!/newGravityMoment! all consume. Same formula as compressed_moments.jl::build_compressed_factual and oracle_fast.jl's own gravity_compute block (duplicated there too, not newly introduced here). RECTANGULAR (exclude-ROW-destination unrestricted-core release, 2026-07-24): D_dest read from ctx (falls back to ctx.D for contexts without the field, matching oracle_fast.jl's own defensive pattern)."
function aod_pow_matrix(θ_full::AbstractVector, ctx)
    γo = ctx.γ; D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D; μ = θ_full[1]
    lambda = reshape(γo.P, (Ddest, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], (D, Ddest))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    return (Aod ./ γo.cHat) .^ (-μ)
end

"""
    compressed_gravity_raw(θ_full, ctx) -> Float64

RAW (pre-post-processing) gravity-moment scalar at θ_full, via the EXISTING
production `newGravityMoment!` (moments/newGravityMoment!.jl), called on a
throwaway 1-row buffer (UoModel==1's branch writes the same scalar into every
row regardless of W, verified from its own `@. G[:, end...] = sumGrav`
broadcast). Errors loudly (not silently) if this ctx's indicators don't match
the assumptions this file was built and tested against (gravMoment==1),
rather than silently mishandling a differently-configured ctx.
"""
function compressed_gravity_raw(θ_full::AbstractVector, ctx)
    ind = ctx.γ.indicators
    ind.gravMoment == 1 || error("compressed_gravity_raw: ctx.γ.indicators.gravMoment != 1 -- compressed_live.jl was built/validated only for this investigation's gravMoment==1 config; extend before reusing elsewhere.")
    AodPow = aod_pow_matrix(θ_full, ctx)
    G1 = zeros(1, 1)
    # exclude-ROW-destination production release (2026-07-24): newGravityMoment!'s signature
    # (moments/newGravityMoment!.jl) gained a Ddest parameter (inserted right after D, before W)
    # as part of the rectangular D_origin/D_destination port -- this call site was missing it
    # entirely (a real, destination_sample-independent regression: broken for :all_legacy too,
    # not just :exclude_row, since compressed_live.jl was never touched by the omit-ROW work and
    # so never got updated to match). ctx.D_dest == ctx.D under :all_legacy, so this fix is a
    # no-op there and only changes behavior (from "crash") under a rectangular ctx.
    # port/shared-winner-pair-core-hessian-production-2026-07-25: `hasproperty` guard added -- this
    # was the ONE remaining unguarded `ctx.D_dest` access in this file (every other access already
    # used this pattern), a real latent gap flagged but deliberately left untouched by the
    # diag/compressed-hessian-operator-audit-2026-07-25 D=4 validation script's own comments; now a
    # genuine blocker for this port's D=4 gates once CM/CM+meanZC/origin-ZC also call this function
    # from their own new compressed-core paths, so fixed here rather than left as a known gap.
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    newGravityMoment!(G1, ctx.τ, ctx.D, Ddest, 1, ones(ctx.D), AodPow, @view(ctx.U[1:1, :]), ind.GravityMomentFirstApproach, ind.UoModel)
    return G1[1, 1]
end

"""
    fill_gravity_column!(obj, grav_raw)

Writes the fully post-processed gravity-moment column (`G[:, obj.d]`, i.e.
`obj.H[:, 2+obj.d]`) from the raw scalar `grav_raw`, applying the SAME
SamplingWeights/NormalizeMoments/usePMM post-processing
`EK_moments_gammanorm_directgp!` applies to every column (gdiv==1 for this
column since it is > D^2+1, i.e. NOT divided by gammafac -- matches
`simple_end = D^2+1` in moments_gammanorm.jl). Uses ONLY `obj.γ` (== ctx.γ by
construction in `d4_exact_setup`), so no `ctx` argument is needed.
"""
function fill_gravity_column!(obj, grav_raw::Float64)
    γo = obj.γ
    d = obj.d
    W = size(obj.U, 1)
    ind = γo.indicators
    nrm_g = (ind.NormalizeMoments == 1 && !(d in γo.moments_without_var)) ? 1.0 / γo.σ_Moments[d] : 1.0
    pmm_g = ind.usePMM == 1 ? γo.PMM[d] : 0.0
    @views @. obj.H[1:W, 2 + d] = γo.SamplingWeights[1:W] * nrm_g * (grav_raw - pmm_g)
    return nothing
end

# ============================================================================
# Compressed inner-solve callback state + KNITRO callbacks.
# ============================================================================

"Per-inner-solve mutable bundle passed as KNITRO's userParams for the compressed callbacks."
mutable struct CompressedCBState
    obj::Any                    # PsiObjectiveBundleImplicit
    cf::CompressedFactual
    grav_raw::Float64
    dense_materialized::Bool    # true once obj.H's G columns have been filled from cf (lazy, once per inner solve; only needed by the :dense_reference core-Hessian backend now)
    core_ws::Union{Nothing,CoreExactHessianWorkspace}   # built lazily from `cf` on first Hessian call this inner solve (port/shared-winner-pair-core-hessian-production-2026-07-25)
    fg_ws::EconomicFGWorkspace   # Addendum Part A remediation (2026-07-26): persistent scratch for
    # _callbackEvalFG_inner_compressed! (compressed_cc_value_grad!) -- eliminates the per-FG-callback
    # allocations compressed_cc_value_grad used to incur. Built once per inner solve (same lifecycle
    # as `cf`, sized from it), not per callback.
    # No-moments/no-composite-G task (2026-07-28): same-point cache for the shared Hessian-weight
    # prep (operator_hessian_weights.jl) -- unrestricted now goes through the EXACT same
    # `operator_prep_for_hessian!`/`dual_index!` mechanism as the 4 restricted families (previously
    # it unconditionally trusted `obj.arg0` with no same-point check at all; this is a strictly
    # stronger guarantee, not a behavior change on any already-passing gate).
    hw_cache::HessianWeightCache
end

"Backward-compatible outer constructor for the 4 existing call sites that predate the shared core-Hessian workspace field (compressed_live.jl/compressed_live_v2.jl/fast_range_screen.jl/infeasibility_screen.jl/compressed_inner_alt_solvers.jl) -- none of them need to change."
CompressedCBState(obj, cf::CompressedFactual, grav_raw::Float64, dense_materialized::Bool) =
    CompressedCBState(obj, cf, grav_raw, dense_materialized, nothing, EconomicFGWorkspace(cf),
                       HessianWeightCache(1 + obj.outer_constr_index))   # x = [ζ; λ], length(λ) == outer_constr_index (same convention _archC_prep_for_hessian!'s H[:,2:1+outer_constr_index] slice uses)

# No-moments/no-composite-G task (2026-07-28): unrestricted's own dispatched accessors for the
# shared operator_hessian_weights.jl mechanism -- `st.fg_ws.q` is unrestricted's equivalent of the
# 4 restricted families' `st.arg0`, and `st.cf` (held directly, not behind a `Ref{Any}`) is its
# equivalent of their `core_cf_ref[]`.
_r_buffer(st::CompressedCBState) = st.fg_ws.q
_cf_identity(st::CompressedCBState) = st.cf

"""
    dual_index!(st::CompressedCBState, x) -> st.fg_ws.q

Computes `st.fg_ws.q = r = -ζ·1 - E·λ` in place via the shared `compressed_dual_index!`
(compressed_cc_inner.jl) -- the SAME primitive `compressed_cc_value_grad!` (the FG callback) calls,
so FG and the Hessian-weight prep (operator_hessian_weights.jl::operator_prep_for_hessian!) run the
identical code path.
"""
function dual_index!(st::CompressedCBState, x::AbstractVector{Float64})
    ζ = x[1]
    λ = @view x[2:end]
    compressed_dual_index!(st.fg_ws, ζ, λ, st.cf)
    return st.fg_ws.q
end

"""
    _callbackEvalFG_inner_compressed!

Compressed replacement for `_callbackEvalFG_inner_profiled!`: objective +
gradient w.r.t. (zeta,lambda) from `compressed_cc_value_grad` (O(W*D)),
never touching `obj.H`'s dense G. Writes `q` (the returned `arg0`) into
`obj.arg0` so a SUBSEQUENT dense Hessian callback (which reads `obj.arg0`,
see `hessian!` in cc_algo/PsiObjectiveBundle.jl) stays in sync even though
this callback bypasses `Q`'s own callable method entirely.
"""
function _callbackEvalFG_inner_compressed!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    ζ = x[1]
    λ = @view x[2:end]
    @prof "inner_dual_fg_callback_compressed" begin
        # Addendum Part A remediation (2026-07-26): compressed_cc_value_grad! writes g_λ directly
        # into evalResult.objGrad's own view (no extra copy) and uses st.fg_ws's persistent scratch
        # for every intermediate -- was compressed_cc_value_grad, which allocated 7-8 fresh arrays
        # (several W-scale) on every one of these calls (many per inner solve). Same math, same
        # Psi!/dPsi! calls, unchanged.
        f, g_ζ = compressed_cc_value_grad!(st.fg_ws, @view(evalResult.objGrad[2:end]), ζ, λ, st.cf;
                                            Psi! = obj.Psi!, dPsi! = obj.dPsi!)
        evalResult.obj[1] = f <= obj.lower_limit ? -KNITRO.KN_INFINITY : f
        evalResult.objGrad[1] = g_ζ
        obj.arg0 .= st.fg_ws.q
        _publish_dual_index_cache!(st, x)   # no-moments/no-composite-G task (2026-07-28): let a same-point Hessian call reuse this r
    end
    _INNER_CALL_COUNTERS[].n_fg_calls += 1
    return 0
end

"""
    _callbackEvalH_inner_compressed!

port/shared-winner-pair-core-hessian-production-2026-07-25: for the
UNRESTRICTED family the entire Hessian IS the common core block H_EE (there
are no CM/ZC restriction columns to carve off), so this callback now calls
the shared exact winner-pair backend DIRECTLY on `evalResult.hess`
(`hessian_core_winner_pair!`/`winner_pair_hessian!` already produce KNITRO's
packed row-major upper triangle natively -- no dense round-trip needed here,
unlike the restricted families whose H_EE is only a SUB-block of a larger
packed Hessian). Built from `st.cf` (already available, no rebuild), lazily,
once per inner solve -- same cadence the old dense-materialize step used.

The dense `obj.H` materialization (`materialize_dense_factual_structured!` +
`fill_gravity_column!`) this replaced is now SKIPPED for the production
default backend (it cost real time -- see
docs/UNRESTRICTED_WINNER_PAIR_VS_DENSE_BLAS_BENCHMARK_2026-07-25.md's
isolated-callback numbers -- and nothing else in this inner solve reads
`obj.H`'s G columns; the FG callback above already gets everything it needs
from `st.cf` directly). It is still run, and `CS.hessian!` still called, when
`UNRESTRICTED_CORE_HESSIAN_BACKEND[] === :dense_reference` (anti-regression /
emergency-revert path, task §5) so that named fallback stays byte-identical
to pre-port production.
"""
function _callbackEvalH_inner_compressed!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    @prof "inner_dual_hessian_callback_compressed" begin
        backend = UNRESTRICTED_CORE_HESSIAN_BACKEND[]
        if backend === :dense_reference
            if !st.dense_materialized
                ncolI = st.cf.oci - 1
                materialize_dense_factual_structured!(@view(obj.H[:, 3:2+ncolI]), st.cf)
                fill_gravity_column!(obj, st.grav_raw)
                st.dense_materialized = true
            end
            CS.hessian!(evalResult.hess, obj)
            record_core_hessian_call!(:dense_reference; fallback_reason = :debug_reference_requested)
        else
            # No-moments/no-composite-G task (2026-07-28): route through the SAME shared
            # operator_prep_for_hessian!/HessianWeightCache mechanism the 4 restricted families use,
            # instead of unconditionally trusting `obj.arg0` with no same-point check at all (the
            # prior behavior here) -- one function, used everywhere it conceptually applies, per this
            # task's explicit design preference. On the common case (KNITRO's Hessian call
            # immediately follows its FG call at the same point) this is a cache hit, zero extra
            # work; on any other point it recomputes `r` via `dual_index!(st, x)` (the same
            # `compressed_dual_index!` primitive the FG callback itself calls), never a dense `obj.H`
            # read either way.
            operator_prep_for_hessian!(st, evalRequest.x)
            if st.core_ws === nothing
                st.core_ws = build_core_exact_hessian_workspace(st.cf)
                record_compressed_core_rebuild!()
            end
            if backend === :exact_winner_pair_parallel
                hessian_core_winner_pair!(evalResult.hess, obj.arg2, obj, st.core_ws.parallel_ws;
                    workers = UNRESTRICTED_CORE_HESSIAN_WORKERS[], storage = UNRESTRICTED_CORE_HESSIAN_STORAGE[])
                record_core_hessian_call!(:exact_winner_pair_parallel)
            elseif backend === :exact_winner_pair_serial
                winner_pair_hessian!(evalResult.hess, obj, serial_ctx(st.core_ws))
                record_core_hessian_call!(:exact_winner_pair_serial)
            else
                error("_callbackEvalH_inner_compressed!: unknown UNRESTRICTED_CORE_HESSIAN_BACKEND[] = :$backend")
            end
        end
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

"""
    inner_loop_KNITRO_compressed(obj, st) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)

Faithful compressed mirror of `inner_loop_KNITRO_profiled` (oracle_fast.jl):
identical variable/bound/init-value setup and option file, registering the
COMPRESSED callbacks instead, with `st::CompressedCBState` as userParams
instead of `obj` directly.
"""
function inner_loop_KNITRO_compressed(obj, st::CompressedCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)

    # Ported from diag/fullA-inner-blas-threading (parallelism_guards.jl); see the same note in
    # oracle_fast.jl::inner_loop_KNITRO_profiled -- this is the compressed variant of that same
    # production inner-solve choke point.
    CS.guard_enter_inner_solve!()
    try
        kc = KNITRO.KN_new()
        KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
        KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
        KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

        cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_compressed!)
        KNITRO.KN_set_cb_user_params(kc, cb, st)
        KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

        if KNITRO.KN_get_int_param(kc, "hessopt") == 1
            KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_compressed!)
        end
        if obj.complement_index != [0 0]
            CS.inner_loop_complementarity_constraints(kc, obj)
        end

        @prof "inner_knitro_dual_solve_compressed" begin
            KNITRO.KN_solve(kc)
        end
        nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
        CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
        KNITRO.KN_free(kc)

        return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
    finally
        CS.guard_exit_inner_solve!()
    end
end

"""
    inner_loop_internal_compressed(obj, θ, ctx) -> (K_hard, x, nStatus, n_fg, n_hess, st)

Compressed mirror of `inner_loop_internal_profiled`. Builds the
`CompressedFactual` (O(W*D), may throw `TiedWinnerError` -- NOT caught here,
propagates to the caller, matching this file's fallback contract), computes
the K (counterfactual-objective) column directly (O(W), no compression
needed -- see note below), then runs the compressed inner KNITRO solve.
Returns the extra `st::CompressedCBState` (unlike the dense mirror) so the
caller can reuse its lazily-materialized dense G for post-processing.

K COLUMN NOTE: `EK_moments_gammanorm_directgp!`'s `K[s] = theta[3+D] * SW[s]`
is already a trivial O(W) per-draw broadcast of a SCALAR counterfactual value
(gamma'_focal direct, this ctx's `counterExplicit==0` branch) -- there is no
O(W*D^2) structure to compress here either; computed directly rather than
introducing a dependency on the dense `moments!` path.
"""
function inner_loop_internal_compressed(obj, θ_full, ctx)
    # NOTE: build_economic_moment_state! is the compressed analog of dense's ONE `obj.moments!`
    # call -- timed under the SAME "inner_moment_build[_compressed]" label so the two are directly
    # comparable in prof_summary() output (an earlier version of this function left this call
    # OUTSIDE any @prof block, silently under-reporting the compressed build cost as ~0 -- fixed
    # after the benchmark caught it, see docs/compressed_live_integration_report.md's speedup-
    # measurement section).
    #
    # FIX (shared economic moment-state builder task, 2026-07-27): this call previously read
    # `build_compressed_factual(θ_full, ctx; check_ties=true)` directly -- the ALWAYS-allocating
    # reference builder -- even though c10_d20_production_driver.jl's `attach_compressed_factual_
    # workspace` call (§3.1, 2026-07-25) already attaches a campaign-lifetime `ctx.cf_workspace` in
    # every real production run. This is THE per-inner-solve moment build for the unrestricted
    # family under `moment_representation=:compressed` (the production driver's default mode) --
    # i.e. a genuine PRODUCTION_HOT_PATH allocation the 2026-07-25 port task's own remediation
    # (which fixed the 4 restricted families' `cf_build` call sites) missed for THIS, the 5th,
    # family. `build_economic_moment_state!` dispatches to the in-place `build_compressed_factual!`
    # whenever ctx.cf_workspace is attached (bit-identical output either way -- same guarantee
    # build_compressed_factual!'s docstring establishes), and falls back to the allocating builder
    # unchanged for any ctx that never attached one (no regression for non-production callers).
    # No-moments/no-composite-G task (2026-07-28): check_ties=false -- see the identical
    # change/rationale in cm_hessian_architectures.jl::wrap_moments_with_cm_archB. The winner
    # (argmin) assignment itself is computed unconditionally regardless of this flag; disabling the
    # check just means a literal machine-precision tie no longer throws/falls back to the entire
    # dense evaluate_fullA_fast for that point.
    cf = @prof "inner_moment_build_compressed" build_economic_moment_state!(θ_full, ctx; check_ties = false)

    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    # True no-H operator bundle (2026-07-28 continuation): unrestricted's own priming convention
    # already differs from the 4 restricted families' (a bare scalar grav_raw threaded through
    # CompressedCBState, not a materialized column) -- prime_operator! doesn't fit this family
    # cleanly (see operator_psi_bundle.jl's own header note), so this dispatches directly rather
    # than calling it, writing into `obj.payoff` instead of `obj.H[:,1]`.
    if obj isa OperatorPsiBundle
        obj.payoff .= θ_full[3 + ctx.D] .* SW
        obj.H_save = obj.payoff[1] * (-1.0)^obj.find_smallest
    else
        obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
        obj.H[:, 2] .= 1.0
        obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest
    end

    grav_raw = compressed_gravity_raw(θ_full, ctx)

    st = CompressedCBState(obj, cf, grav_raw, false)
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_compressed(obj, st)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess, st
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess, st
    end
end

"""
    build_unrestricted_operator_ctx(ctx; moment_representation=MOMENT_REPRESENTATION[]) -> ctx

Wire unrestricted-family production port task (2026-07-29): unrestricted has no separate
`build_*_production_context` wrapper the way the 4 restricted families do -- `ctx.obj` IS the
production bundle directly (`d20_real_setup`/`d20_real_setup_design` always build a dense
`PsiObjectiveBundleImplicit`). This is the "standalone wrapper" `FIVE_FAMILY_NO_H_BUNDLE_GATE_
2026-07-28.md` describes -- it never actually got called from either production driver
(`c10_d20_production_driver.jl`/`c10_d20_production_driver_unified.jl`), so `ctx.obj` stayed the
dense bundle in every real solve despite the doc's own "WIRED_AND_GATED" claim (confirmed false
live 2026-07-29 -- see `campaign_unrestricted_runner.jl`'s pre-fix hardcoded
`bundle_type=OperatorPsiBundle` print, which was never actually true).

`moment_representation=:operator` (production default, mirrors `build_cm_production_context`'s own
default) constructs a companion `OperatorPsiBundle` from `ctx.obj`'s own scalar fields -- identical
construction to what `test_operator_no_H_bundle_equivalence_unrestricted[_d20].jl` already validate
(D=4 and real D=20/W=100,000, exact agreement, both gates unchanged by this addition) -- and
replaces `ctx.obj` with it. `:dense_reference` (explicit opt-in) returns `ctx` unchanged. Idempotent:
a `ctx` whose `obj` is already an `OperatorPsiBundle` (e.g. a second call on a resumed/merged ctx)
is returned as-is rather than re-wrapped.
"""
function build_unrestricted_operator_ctx(ctx; moment_representation::Symbol = MOMENT_REPRESENTATION[])
    moment_representation in (:operator, :dense_reference) ||
        error("build_unrestricted_operator_ctx: moment_representation must be :operator or :dense_reference, got :$moment_representation")
    moment_representation === :dense_reference && return ctx
    obj_d = ctx.obj
    obj_d isa OperatorPsiBundle && return ctx
    obj_o = OperatorPsiBundle(δ = obj_d.δ, find_smallest = obj_d.find_smallest,
        γ = obj_d.γ, l = obj_d.l, outer_constr_index = obj_d.outer_constr_index,
        inequality_index = obj_d.inequality_index, complement_index = obj_d.complement_index,
        U = obj_d.U, N = obj_d.N, lower_limit = obj_d.lower_limit,
        use_cached_x = obj_d.use_cached_x, threshold_state = obj_d.threshold_state,
        inner_loop_opt = obj_d.inner_loop_opt)
    return merge(ctx, (obj = obj_o,))
end

"""
    evaluate_fullA_fast_compressed(x_free, ctx; kwargs...) -> (result, prof_meta)

Compressed-mode implementation dispatched to by
`evaluate_fullA_fast(...; moment_representation=:compressed)`. Same
signature/semantics/return shape as the dense `evaluate_fullA_fast`; the
POST-inner-solve tail (primal-weight recovery, moment residual, KKT residual,
gravity, winner recompute, result NamedTuple assembly) is a DELIBERATE,
documented DUPLICATE of `evaluate_fullA_fast`'s own tail (oracle_fast.jl) --
not factored into a shared helper -- so the dense function's code is
PROVABLY untouched by this file (see the module-level docstring). Any drift
between the two tails would be caught immediately by
`test_compressed_live_integration.jl`'s field-by-field comparison, which is
run after every change to either.
"""
function evaluate_fullA_fast_compressed(x_free::AbstractVector{Float64}, ctx;
        cache = nothing, use_cache::Bool = true,
        mode::Symbol = :hard, warm::Bool = true, tag::String = "",
        verification_backend::Symbol = UNRESTRICTED_VERIFICATION_BACKEND_DEFAULT[],
        # Final-architecture-closure task (2026-07-27), Goal 9: `gravity_raw` (cbuf[2], via a dense
        # `obj(inner_x, constr=...)` call) and `benchmark_unweighted_moment_mean`/`max_abs_moment_resid`
        # (via `CS.select_G_from_H(obj, obj.H)`) are reporting-only outputs -- NOT consumed by
        # `classify_inner_result`/`is_cacheable_result`/`is_verified_success` (see the comment at this
        # function's `gravity_raw` assignment below, unchanged from the prior verification-defaults
        # phase) -- yet were unconditionally materializing dense obj.H/G even under
        # verification_backend=:operator. Default false: these three fields report NaN/empty (same
        # convention the pre-existing `!solved` failure branch above already uses for them), and the
        # dense materialize/select_G_from_H/constr-eval work is skipped entirely. Pass `true` to get
        # the full dense reporting values back (unchanged behavior from before this flag existed) --
        # e.g. for an explicit diagnostic/debug run. Independent of `verification_backend`: an
        # :operator-verified inner solve with dense_reference_diagnostics=true still verifies via the
        # operator (Delta_dual/Delta_primal/etc. untouched), it just ALSO pays for these three extra
        # reporting fields.
        dense_reference_diagnostics::Bool = false)

    mode == :hard || error("evaluate_fullA_fast_compressed: mode=:$mode not implemented (matches oracle.jl)")

    obj = ctx.obj
    key = FullAEvalKey(collect(x_free), obj.δ, obj.find_smallest, obj.inner_loop_opt, mode, context_fingerprint(ctx))

    if cache !== nothing && use_cache
        hit = @prof "cache_lookup_compressed" _cache_lookup(cache, key)
        if hit !== nothing
            return merge(hit, (cache_hit = true, tag = tag)), (n_inner_solves = 0, n_inner_infeasible = 0, n_inner_iters = 0, n_fg_calls = 0, n_hess_calls = 0)
        end
    end

    solves0 = CS.INNER_SOLVE_COUNT[]; infeas0 = CS.INNER_INFEAS_COUNT[]; iters0 = CS.INNER_ITERS_TOTAL[]

    t_total0 = time()
    if !warm
        @prof "warm_start_reset_compressed" begin
            obj.x .= NaN
        end
    end

    θ_full = @prof "reconstruct_full_compressed" CS.reconstruct_full(x_free, ctx.m)

    local K_hard, inner_x, nStatus, n_fg, n_hess, st
    t_inner0 = time()
    # No-moments/no-composite-G task (2026-07-28): the TiedWinnerError-catch-and-redispatch-to-
    # dense fallback this try/catch used to serve is now unreachable -- `inner_loop_internal_
    # compressed` builds its `cf` with `check_ties=false` (above), so a literal price tie no longer
    # throws here at all (the winner/argmin is still assigned deterministically either way).
    K_hard, inner_x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed(obj, θ_full, ctx)
    t_inner = time() - t_inner0

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        D_dest_fail = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, D_dest_fail),
                  K_hard = NaN, Delta_dual = NaN, Delta_primal = NaN, Delta_minus_delta = NaN,
                  gravity_raw = NaN, gravity_value = NaN, gravity_R_sum = NaN, gravity_R_mean = NaN,
                  gravity_R_beta = NaN, benchmark_unweighted_moment_mean = Float64[], max_abs_moment_resid = NaN,
                  zeta = NaN, lambda = Float64[], m_mean = NaN, m_min = NaN, m_max = NaN,
                  weight_norm_resid = NaN, mean_m_resid = NaN, max_abs_moment_kkt_resid = NaN,
                  winner_hash = UInt64(0), inner_status = nStatus, inner_iters = inner_iters,
                  primal_dual_gap = NaN, cache_hit = false, warm_started = warm, tag = tag,
                  elapsed = elapsed, error_reason = "inner solve failed: nStatus=$nStatus")
        cache !== nothing && is_cacheable_result(result) && @prof("cache_materialize_compressed", _cache_store!(cache, key, result))
        prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                     n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                     n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0, n_fg_calls = n_fg, n_hess_calls = n_hess)
        return result, prof_meta
    end

    W = size(obj.U, 1); d = obj.d
    # Final-architecture-closure task (2026-07-27), Goal 9: the dense obj.H materialization +
    # `CS.select_G_from_H`/`obj(inner_x, constr=...)` block below feeds THREE kinds of consumer:
    # (a) `:dense_reference` verification's own Delta_dual/Delta_primal/mean_m_resid/
    #     max_abs_moment_kkt_resid (cbuf[1], obj.arg1, G -- genuinely needs it, that backend IS the
    #     dense path); (b) reporting-only `gravity_raw`/`benchmark_unweighted_moment_mean`/
    #     `max_abs_moment_resid` (cbuf[2], G again -- NOT consumed by
    #     classify_inner_result/is_cacheable_result/is_verified_success, disclosed scope limit, see
    #     docs/FIVE_FAMILY_OPERATOR_VERIFICATION_DEFAULT_RELEASE_2026-07-27.md). Under
    #     verification_backend=:operator this block used to run UNCONDITIONALLY just to serve (b) --
    #     a real dense-G materialization in an otherwise fully operator/compressed production hot
    #     path. `need_dense_block` gates it: on whenever :dense_reference verification needs it
    #     regardless, OR the caller explicitly asked for the extra reporting fields via
    #     `dense_reference_diagnostics=true`.
    need_dense_block = (verification_backend === :dense_reference) || dense_reference_diagnostics
    local K, G, cbuf, gravity_raw
    if need_dense_block
        # ---- ensure obj.H's dense G columns are populated (lazy; may already be done by the
        # Hessian callback -- if KNITRO converged without ever calling it, e.g. a warm-started
        # already-converged point, do it here instead) so the REST of this tail can reuse the
        # SAME post-processing formulas the dense path uses, unchanged. ----
        if !st.dense_materialized
            @prof "materialize_dense_for_postproc" begin
                ncolI = st.cf.oci - 1
                # Continuation 10 Section 9: same structured swap as the Hessian callback above,
                # kept consistent so this (rarely-hit) fallback path can never drift from it.
                materialize_dense_factual_structured!(@view(obj.H[:, 3:2+ncolI]), st.cf)
                fill_gravity_column!(obj, st.grav_raw)
                st.dense_materialized = true
            end
        end

        K, G = @prof "moments_reuse_compressed" begin
            (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))
        end

        ncon = obj.d - obj.outer_constr_index + 2
        cbuf = zeros(ncon)
        @prof "primal_weight_recovery_compressed" begin
            obj(inner_x, constr = @view(cbuf[1:ncon]))
        end
        gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN
    else
        K = Float64[]; G = zeros(0, 0); cbuf = Float64[]
        gravity_raw = NaN   # reporting-only diagnostic, not computed at this default -- see
        # dense_reference_diagnostics kwarg docstring above; NOT read by classify_inner_result/
        # is_cacheable_result/is_verified_success (same convention the pre-existing !solved failure
        # branch above already uses for this field).
    end
    ζstar = inner_x[1]; λstar = inner_x[2:end]

    local m_weights, Delta_dual, Delta_primal, mean_m_resid, max_abs_moment_kkt_resid, weight_norm_resid_val
    if verification_backend === :operator
        cf = st.cf
        cf isa CompressedFactual || error("evaluate_fullA_fast_compressed: verification_backend=:operator requires st.cf to be a CompressedFactual (got $(typeof(cf))) -- prerequisite not met, refusing silent dense fallback")
        ov = @prof "operator_verification_compressed" verify_inner_solution_operator_unrestricted!(ζstar, λstar, cf, obj, W)
        m_weights, verify_op = verify_namedtuple_from_operator(ov, obj, W, nStatus)
        Delta_dual = verify_op.Delta_dual
        Delta_primal = verify_op.Delta_primal
        mean_m_resid = verify_op.mean_m_resid
        max_abs_moment_kkt_resid = verify_op.max_abs_moment_kkt_resid
        weight_norm_resid_val = verify_op.weight_norm_resid
    elseif verification_backend === :dense_reference
        Delta_dual = cbuf[1] / 1e10
        m_weights = copy(obj.arg1)
        p_weights = m_weights ./ sum(m_weights)
        Delta_primal = @prof "primal_divergence_compute_compressed" primal_divergence(m_weights)

        mean_m_resid = abs(sum(m_weights) / W - 1.0)
        nkkt = min(length(λstar), size(G, 2))
        # Continuation 10 Section 9: BLAS-gemv swap (kkt_residual_blas, oracle_fast.jl) --
        # see docs/fullA_D20_blas_audit_report.md, ~2.1-2.2x.
        max_abs_moment_kkt_resid = @prof "kkt_residual_compute_compressed" kkt_residual_blas(G, m_weights, nkkt, W)
        weight_norm_resid_val = abs(sum(p_weights) - 1.0)
        record_dense_reference_verification!()
    else
        error("evaluate_fullA_fast_compressed: unknown verification_backend=:$verification_backend (expected :operator or :dense_reference)")
    end
    D_dest_g = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D*D_dest_g], ctx.D, D_dest_g)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (D_dest_g, ctx.D))'
    gravity_val, logA, R_sum, R_mean, R_beta = @prof "gravity_compute_compressed" begin
        Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
        AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
        gv = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
        lA = -log.(AodPow)
        rs = sum(ctx.q_tilde .* lA)
        (gv, lA, rs, rs / (ctx.D * D_dest_g), rs / sum(ctx.q_tilde .^ 2))
    end

    # Continuation 10 Section 9: BLAS-gemv swap (moment_resid_blas, oracle_fast.jl) --
    # see docs/fullA_D20_blas_audit_report.md, ~2.1x.
    benchmark_unweighted_moment_mean = need_dense_block ?
        (@prof "moment_resid_compute_compressed" moment_resid_blas(G, d, W)) : Float64[]
    max_abs_moment_resid = isempty(benchmark_unweighted_moment_mean) ? NaN : maximum(abs.(benchmark_unweighted_moment_mean))

    winner, price_, gap_ = @prof "winner_compute_compressed" compute_winners_fast(θ_full, ctx)
    winner_hash = hash(winner)

    t_total = time() - t_total0
    elapsed = (total = t_total, inner = t_inner, post = t_total - t_inner)

    result = (x_free = collect(x_free), θ_full = θ_full,
              gamma_focal_prime = θ_full[3+ctx.D], logA = logA,
              K_hard = K_hard, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
              Delta_minus_delta = Delta_dual - obj.δ,
              gravity_raw = gravity_raw, gravity_value = gravity_val,
              gravity_R_sum = R_sum, gravity_R_mean = R_mean, gravity_R_beta = R_beta,
              benchmark_unweighted_moment_mean = benchmark_unweighted_moment_mean, max_abs_moment_resid = max_abs_moment_resid,
              zeta = ζstar, lambda = collect(λstar),
              m_mean = sum(m_weights)/W, m_min = minimum(m_weights), m_max = maximum(m_weights),
              weight_norm_resid = weight_norm_resid_val,
              mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
              winner_hash = winner_hash, inner_status = nStatus, inner_iters = inner_iters,
              primal_dual_gap = abs(Delta_dual - Delta_primal),
              cache_hit = false, warm_started = warm, tag = tag,
              elapsed = elapsed, error_reason = nothing)

    cache !== nothing && is_cacheable_result(result) && @prof("cache_materialize_compressed", _cache_store!(cache, key, result))
    prof_meta = (n_inner_solves = CS.INNER_SOLVE_COUNT[] - solves0,
                 n_inner_infeasible = CS.INNER_INFEAS_COUNT[] - infeas0,
                 n_inner_iters = CS.INNER_ITERS_TOTAL[] - iters0, n_fg_calls = n_fg, n_hess_calls = n_hess)
    return result, prof_meta
end

# ============================================================================
# Base-state interface for downstream consumers (lfix_incremental.jl /
# composite_gradient*.jl, owned by a DIFFERENT parallel workstream this
# session -- not editable here, documented instead).
# ============================================================================

"""
    compressed_base_state(x_free0, ctx) -> BaseDualState

Runs the COMPRESSED inner solve at `x_free0` and returns a `BaseDualState` --
the EXACT, PRE-EXISTING struct type from `three_way_derivatives.jl` (field
names `x_free0`, `θ_full0`, `ζstar`, `λstar`, `m_star`, `inner_status`), NOT a
new/parallel type. `m_star` is obtained from ONE extra call to
`compressed_cc_value_grad` at the converged (ζ*,λ*) (returns `dPsq`, exactly
matching `BaseDualState`'s own field comment "= dPsi(q_s*)") -- no dense G
needed for this step either. See
`docs/compressed_live_integration_report.md`'s base-state interface note for
why this is drop-in compatible with `lfix_incremental.jl::build_lfix_base_cache`,
which takes a `BaseDualState` as-is regardless of how it was produced.

Does NOT catch `TiedWinnerError` -- propagates to the caller (a base-point
tie is rare enough, per lfix_incremental.jl's own docstring, that callers of
THIS function should decide their own fallback, e.g. `solve_base_state` from
three_way_derivatives.jl, which is unaffected by anything in this file).
"""
function compressed_base_state(x_free0::AbstractVector, ctx)
    obj = ctx.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    K_hard, inner_x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed(obj, θ_full0, ctx)
    nStatus in (0, -100, -101, -103) || error("compressed_base_state: inner solve failed, nStatus=$nStatus")
    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    _, _, _, q, m_star = compressed_cc_value_grad(ζstar, λstar, st.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, m_star, nStatus)
end
