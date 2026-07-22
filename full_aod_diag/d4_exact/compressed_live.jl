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

"AodPow (D x D), the level->power transform hFunction!/hFunctionCounter!/newGravityMoment! all consume. Same formula as compressed_moments.jl::build_compressed_factual and oracle_fast.jl's own gravity_compute block (duplicated there too, not newly introduced here)."
function aod_pow_matrix(θ_full::AbstractVector, ctx)
    γo = ctx.γ; D = ctx.D; μ = θ_full[1]
    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
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
    newGravityMoment!(G1, ctx.τ, ctx.D, 1, ones(ctx.D), AodPow, @view(ctx.U[1:1, :]), ind.GravityMomentFirstApproach, ind.UoModel)
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
    dense_materialized::Bool    # true once obj.H's G columns have been filled from cf (lazy, once per inner solve)
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
        f, g_ζ, g_λ, q, _ = compressed_cc_value_grad(ζ, λ, st.cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
        evalResult.obj[1] = f <= obj.lower_limit ? -KNITRO.KN_INFINITY : f
        evalResult.objGrad[1] = g_ζ
        @views evalResult.objGrad[2:end] .= g_λ
        obj.arg0 .= q
    end
    _INNER_CALL_COUNTERS[].n_fg_calls += 1
    return 0
end

"""
    _callbackEvalH_inner_compressed!

Dense Hessian callback (see the "HESSIAN-CALLBACK ADAPTER DECISION" note in
compressed_cc_inner.jl for why this stays dense). Lazily materializes
`obj.H`'s G columns (bilateral+CF block from `cf.winner`/`cf.wval` via
`materialize_dense_factual!`; the one extra gravity column via
`fill_gravity_column!`) EXACTLY ONCE per inner solve (theta -- hence G -- is
fixed for the whole solve), then calls the UNCHANGED production `hessian!`.
"""
function _callbackEvalH_inner_compressed!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    @prof "inner_dual_hessian_callback_compressed" begin
        if !st.dense_materialized
            ncolI = st.cf.oci - 1
            # Continuation 10 Section 9: structured (rank-one + winner-scatter) construction
            # replaces the generic materialize_dense_factual! here -- this IS the actual
            # Hessian-callback moment-materialization step (docs/fullA_D20_structured_moment_report.md,
            # ~4-23x isolated / ~1.32x full-cold-inner-solve, bit-identical).
            materialize_dense_factual_structured!(@view(obj.H[:, 3:2+ncolI]), st.cf)
            fill_gravity_column!(obj, st.grav_raw)
            st.dense_materialized = true
        end
        CS.hessian!(evalResult.hess, obj)
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
    # NOTE: build_compressed_factual is the compressed analog of dense's ONE `obj.moments!` call --
    # timed under the SAME "inner_moment_build[_compressed]" label so the two are directly comparable
    # in prof_summary() output (an earlier version of this function left this call OUTSIDE any @prof
    # block, silently under-reporting the compressed build cost as ~0 -- fixed after the benchmark
    # caught it, see docs/compressed_live_integration_report.md's speedup-measurement section).
    cf = @prof "inner_moment_build_compressed" build_compressed_factual(θ_full, ctx; check_ties = true)   # may throw TiedWinnerError

    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

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
        mode::Symbol = :hard, warm::Bool = true, tag::String = "")

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
    try
        K_hard, inner_x, nStatus, n_fg, n_hess, st = inner_loop_internal_compressed(obj, θ_full, ctx)
    catch e
        if e isa TiedWinnerError
            COMPRESSED_FALLBACK_COUNT[] += 1
            @warn "compressed mode: exact price tie detected, falling back to dense for this point" n_tied_pairs=e.n_tied_pairs examples=e.examples fallback_count=COMPRESSED_FALLBACK_COUNT[] x_free_hash=hash(round.(collect(x_free), digits = 12))
            return evaluate_fullA_fast(x_free, ctx; cache = cache, use_cache = use_cache,
                                        mode = mode, warm = warm, tag = tag, moment_representation = :dense)
        else
            rethrow()
        end
    end
    t_inner = time() - t_inner0

    inner_iters = try
        CS.INNER_ITERS_TOTAL[]
    catch
        missing
    end

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        elapsed = (total = time() - t_total0, inner = t_inner, post = 0.0)
        result = (x_free = collect(x_free), θ_full = θ_full,
                  gamma_focal_prime = θ_full[3+ctx.D], logA = fill(NaN, ctx.D, ctx.D),
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

    W = size(obj.U, 1); d = obj.d
    K, G = @prof "moments_reuse_compressed" begin
        (copy(@view(obj.H[:, 1])), copy(CS.select_G_from_H(obj, obj.H)))
    end

    ncon = obj.d - obj.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = @prof "primal_weight_recovery_compressed" begin
        obj(inner_x, constr = @view(cbuf[1:ncon]))
    end
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj.arg1)
    p_weights = m_weights ./ sum(m_weights)
    Delta_primal = @prof "primal_divergence_compute_compressed" primal_divergence(m_weights)

    mean_m_resid = abs(sum(m_weights) / W - 1.0)
    ζstar = inner_x[1]; λstar = inner_x[2:end]
    nkkt = min(length(λstar), size(G, 2))
    # Continuation 10 Section 9: BLAS-gemv swap (kkt_residual_blas, oracle_fast.jl) --
    # see docs/fullA_D20_blas_audit_report.md, ~2.1-2.2x.
    max_abs_moment_kkt_resid = @prof "kkt_residual_compute_compressed" kkt_residual_blas(G, m_weights, nkkt, W)

    gravity_raw = obj.outer_constr_index <= d ? cbuf[2] : NaN
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+ctx.D^2], ctx.D, ctx.D)
    μ_here = θ_full[1]
    lambda_g = reshape(ctx.γ.P, (ctx.D, ctx.D))'
    gravity_val, logA, R_sum, R_mean, R_beta = @prof "gravity_compute_compressed" begin
        Aod_lvl = Aod_θ .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_here)) .* (lambda_g ./ lambda_g[1,:]')
        AodPow = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_here)
        gv = gravity_value(ctx.τ, AodPow, ctx.q_tilde, ctx.N_obs)
        lA = -log.(AodPow)
        rs = sum(ctx.q_tilde .* lA)
        (gv, lA, rs, rs / ctx.D^2, rs / sum(ctx.q_tilde .^ 2))
    end

    # Continuation 10 Section 9: BLAS-gemv swap (moment_resid_blas, oracle_fast.jl) --
    # see docs/fullA_D20_blas_audit_report.md, ~2.1x.
    benchmark_unweighted_moment_mean = @prof "moment_resid_compute_compressed" moment_resid_blas(G, d, W)
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
              weight_norm_resid = abs(sum(p_weights) - 1.0),
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
