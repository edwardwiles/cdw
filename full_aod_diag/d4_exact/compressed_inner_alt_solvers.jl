# ============================================================================
# Continuation 9, Phase 3C: three ways to solve the inner CC dual problem
# WITHOUT the dense Hessian callback's dependency on a materialized W-by-
# moment G matrix. All three build on the EXISTING, already-validated
# compressed primitives from compressed_cc_inner.jl (compressed_cc_value_grad,
# compressed_cc_hvp -- both built in Continuation 8, verified against the real
# dense bundle via central-FD in test_compressed_cc_inner.jl, but never wired
# into an actual KNITRO solve before this task) and compressed_live.jl's FG
# callback (`_callbackEvalFG_inner_compressed!`, reused unchanged).
#
# BACKGROUND: compressed_live.jl's existing `:compressed` mode already made
# the FG callback dense-free (O(W*D) per call, per docs/fullA_D20_W80k...).
# But its HESSIAN callback (`_callbackEvalH_inner_compressed!`) still lazily
# materializes the FULL dense W x (oci-1) G matrix once per inner solve,
# purely to call the unchanged, existing `hessian!` (cc_algo/PsiObjectiveBundle.jl),
# which does one O(W*ncol^2) BLAS gemm. This file's three variants remove
# THAT remaining dependency:
#
#   Option 3 (:qn)         -- no Hessian callback at all; KNITRO's own
#                              quasi-Newton approximation (BFGS/SR1/L-BFGS,
#                              hessopt 2/3/6) drives the inner solve using
#                              only the (already dense-free) FG callback.
#                              Needs ZERO new code: inner_loop_KNITRO_compressed
#                              (compressed_live.jl) already skips Hessian
#                              registration whenever hessopt != 1 -- only a
#                              new .opt file (ek_inner_{bfgs,sr1,lbfgs}.opt)
#                              is needed; this file's :qn wrapper exists only
#                              for naming symmetry in the benchmark harness.
#   Option 1 (:denseaccum) -- KNITRO still gets a DENSE (ncol+1)x(ncol+1)
#                              Hessian each call (hessopt=exact), but it is
#                              built via (ncol+1) calls to the ALREADY-EXACT
#                              `compressed_cc_hvp` (one per basis direction),
#                              each O(W*D) -- total O(W*D*(ncol+1)) =
#                              O(W*D^3), vs the dense path's O(W*ncol^2) =
#                              O(W*D^4) BLAS gemm. Never touches the W x
#                              (oci-1) G matrix. A genuine D=20x asymptotic
#                              flop reduction (D^4 -> D^3); whether that beats
#                              one highly-optimized BLAS gemm in WALL CLOCK
#                              (many small scalar-loop calls vs one large
#                              vectorized one) is exactly what this task
#                              measures, not assumed.
#   Option 2 (:hvp)        -- genuinely matrix-free: KNITRO's hessopt=product
#                              (5) mode calls this same callback ONLY in
#                              KN_RC_EVALHV mode, requesting ONE Hessian-vector
#                              product per call (the direction KNITRO wants,
#                              read from `evalRequest.vec`) -- answered
#                              directly by `compressed_cc_hvp`, O(W*D) per
#                              call, the dense (ncol+1)x(ncol+1) matrix is
#                              NEVER assembled, not even implicitly. Requires
#                              KNITRO's CG-based interior algorithm (product
#                              Hessians are not compatible with direct/SQP
#                              factorization) -- ek_inner_hvp.opt sets
#                              algorithm=cg accordingly.
#
# Correctness for all three rests on compressed_cc_value_grad/compressed_cc_hvp
# already being validated against the real dense PsiObjectiveBundleImplicit
# bundle (test_compressed_cc_inner.jl, FD tolerance 1e-5) -- not re-derived
# here. This file adds NO new moment-representation math, only new KNITRO
# wiring around already-proven primitives.
# ============================================================================

"O(W) exact q_s = -ζ - λ'G_s at an arbitrary (ζ,λ), from the compressed factual. Needed because KNITRO may call the Hessian/HVP callback at an x DIFFERENT from the most recent FG callback's x (e.g. during a CG sub-iteration within one Newton step) -- q cannot be assumed cached, so it is recomputed here every call (O(W*D), the same cost as one HVP)."
function compressed_q_at(ζ::Real, λ::AbstractVector, cf::CompressedFactual)
    contr = compressed_dual_contraction(λ, cf)
    q = similar(contr)
    @inbounds @. q = -ζ - contr
    return q
end

# ============================================================================
# Option 2 (:hvp) -- genuinely matrix-free Hessian-vector product callback.
# Registered the same way as a dense Hessian callback (KN_set_cb_hess); which
# mode KNITRO actually invokes it in is controlled entirely by the .opt
# file's hessopt (5 = product => every call has evalRequestCode==KN_RC_EVALHV,
# never KN_RC_EVALH) -- confirmed from KNITRO.jl's C_wrapper.jl EvalRequest/
# EvalResult struct definitions (`vec`/`hessVec` fields exist distinctly from
# `hess`) and libknitro.jl's KN_RC_EVALHV=7 vs KN_RC_EVALH=3 request codes.
# ============================================================================
function _callbackEvalHV_inner_compressed!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    v = evalRequest.vec
    ζ = x[1]; λ = @view x[2:end]
    pζ = v[1]; pλ = @view v[2:end]
    @prof "inner_dual_hvp_callback_compressed" begin
        q = compressed_q_at(ζ, λ, st.cf)
        Hpζ, Hpλ = compressed_cc_hvp(q, pζ, pλ, st.cf; ddPsi! = obj.ddPsi!)
        evalResult.hessVec[1] = Hpζ
        @views evalResult.hessVec[2:end] .= Hpλ
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

# ============================================================================
# Option 1 (:denseaccum) -- dense (ncol+1)x(ncol+1) Hessian assembled via
# (ncol+1) compressed_cc_hvp basis-vector calls. Never touches the W x
# (oci-1) G matrix; DOES still hand KNITRO the small dense block (hessopt=
# exact still applies, so KNITRO's direct/SQP algorithms remain usable,
# unlike Option 2).
# ============================================================================
function _callbackEvalH_inner_compressed_denseaccum!(kc, cb, evalRequest, evalResult, userParams)
    st = userParams
    obj = st.obj
    x = evalRequest.x
    ζ = x[1]; λ = @view x[2:end]
    n = length(x)
    @prof "inner_dual_hessian_callback_compressed_denseaccum" begin
        q = compressed_q_at(ζ, λ, st.cf)
        Hd = Matrix{Float64}(undef, n, n)
        e = zeros(n)
        @inbounds for k in 1:n
            e[k] = 1.0
            pζ = e[1]
            Hpζ, Hpλ = compressed_cc_hvp(q, pζ, @view(e[2:end]), st.cf; ddPsi! = obj.ddPsi!)
            Hd[1, k] = Hpζ
            @views Hd[2:end, k] .= Hpλ
            e[k] = 0.0
        end
        # pack upper-triangular row-major, matching cc_algo/PsiObjectiveBundle.jl::hessian!'s
        # OWN packing convention exactly (not re-derived) -- symmetrized defensively (the HVP
        # composition is symmetric analytically; averaging absorbs FP-order-of-operations noise,
        # never masks a real asymmetry since |Hd[i,j]-Hd[j,i]| is checked separately in validation).
        idx = 1
        @inbounds for i in 1:n, j in i:n
            evalResult.hess[idx] = 0.5 * (Hd[i, j] + Hd[j, i])
            idx += 1
        end
    end
    _INNER_CALL_COUNTERS[].n_hess_calls += 1
    return 0
end

# ============================================================================
# Solver-loop wrappers, one per variant. Each mirrors inner_loop_KNITRO_compressed
# (compressed_live.jl) exactly in variable/bound/init-value setup and FG-callback
# registration -- ONLY the Hessian registration differs.
# ============================================================================

"Option 3 (quasi-Newton, no exact Hessian at all): thin wrapper around the EXISTING inner_loop_KNITRO_compressed, unmodified -- its own `if hessopt==1` guard already skips Hessian registration whenever `obj.inner_loop_opt` points to a hessopt!=1 file. Named separately here only for benchmark-harness symmetry with the other two variants."
inner_loop_KNITRO_compressed_qn(obj, st::CompressedCBState) = inner_loop_KNITRO_compressed(obj, st)

function inner_loop_KNITRO_compressed_denseaccum(obj, st::CompressedCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
    KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
    KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_compressed!)
    KNITRO.KN_set_cb_user_params(kc, cb, st)
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)
    hessopt = KNITRO.KN_get_int_param(kc, "hessopt")
    hessopt == 1 || error("inner_loop_KNITRO_compressed_denseaccum: expected hessopt=exact(1), got $hessopt -- check obj.inner_loop_opt points at a hessopt=exact file")
    KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalH_inner_compressed_denseaccum!)
    if obj.complement_index != [0 0]
        CS.inner_loop_complementarity_constraints(kc, obj)
    end
    @prof "inner_knitro_dual_solve_denseaccum" begin
        KNITRO.KN_solve(kc)
    end
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
    KNITRO.KN_free(kc)
    return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
end

function inner_loop_KNITRO_compressed_hvp(obj, st::CompressedCBState)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
    KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
    KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_compressed!)
    KNITRO.KN_set_cb_user_params(kc, cb, st)
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)
    hessopt = KNITRO.KN_get_int_param(kc, "hessopt")
    hessopt == 5 || error("inner_loop_KNITRO_compressed_hvp: expected hessopt=product(5), got $hessopt -- check obj.inner_loop_opt points at ek_inner_hvp.opt")
    KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, _callbackEvalHV_inner_compressed!)
    if obj.complement_index != [0 0]
        CS.inner_loop_complementarity_constraints(kc, obj)
    end
    @prof "inner_knitro_dual_solve_hvp" begin
        KNITRO.KN_solve(kc)
    end
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
    KNITRO.KN_free(kc)
    return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _INNER_CALL_COUNTERS[].n_hess_calls
end

"""
    inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant) -> (K_hard, x, nStatus, n_fg, n_hess, st)

Top-level driver, one per Phase 3C variant (`:qn`, `:denseaccum`, `:hvp`).
Mirrors `inner_loop_internal_compressed` (compressed_live.jl) exactly except
for which solver-loop wrapper it dispatches to. `obj.inner_loop_opt` MUST
already point at the matching .opt file (ek_inner_{bfgs,sr1,lbfgs}.opt for
`:qn`, `ek_inner.opt`/`ek_inner_loose.opt` for `:denseaccum`, ek_inner_hvp.opt
for `:hvp`) -- each wrapper asserts this via `KN_get_int_param(kc,"hessopt")`
rather than silently proceeding with the wrong mode.
"""
function inner_loop_internal_compressed_variant(obj, θ_full, ctx; variant::Symbol)
    # EXPLICIT_REFERENCE / benchmark-only (Phase 3C alt-solver comparison harness, never wired into
    # evaluate_fullA_fast's moment_representation dispatch) -- updated to the canonical
    # `build_economic_moment_state!` anyway (2026-07-27 task) for consistency; unchanged allocating
    # fallback for any ctx without an attached cf_workspace.
    cf = @prof "inner_moment_build_compressed" build_economic_moment_state!(θ_full, ctx; check_ties = true)

    W = size(obj.U, 1)
    SW = ctx.γ.SamplingWeights[1:W]
    obj.H[:, 1] .= θ_full[3 + ctx.D] .* SW
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    grav_raw = compressed_gravity_raw(θ_full, ctx)

    st = CompressedCBState(obj, cf, grav_raw, false)
    nStatus, objSol, x, lambda_, n_fg, n_hess = if variant == :qn
        inner_loop_KNITRO_compressed_qn(obj, st)
    elseif variant == :denseaccum
        inner_loop_KNITRO_compressed_denseaccum(obj, st)
    elseif variant == :hvp
        inner_loop_KNITRO_compressed_hvp(obj, st)
    else
        error("inner_loop_internal_compressed_variant: unknown variant=$variant (must be :qn, :denseaccum, or :hvp)")
    end

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
