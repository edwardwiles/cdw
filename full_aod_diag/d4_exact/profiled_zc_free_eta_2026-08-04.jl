# ============================================================================
# fix/profiled-functional-readiness-closeout-2026-08-03, task §8: genuine free eta_nu/nu for
# origin-ZC and CM+ZC. ADDITIVE ONLY -- does not modify
# profiled_zc_lane_point_evaluators_2026-08-02.jl. That file's 3-positional-arg
# `evaluate_profiled_originzc_point(w_profiled, fctx, pes)` /
# `evaluate_profiled_cmzc_point(w_profiled, fctx, pes)` (fixed nu via `pes.nu_full`) are left
# UNCHANGED and stay the ones every existing inner unit test calls, per this task's own §8.1
# instruction ("preserve fixed-nu wrappers only for existing inner unit tests").
#
# This file adds NEW 4-positional-arg methods of the SAME function names, with `eta_nu` as an
# explicit argument (log-nu units, matching FULL's own eta=log(nu) convention -- see
# cm_originzc_config.jl/cm_originzc_moments.jl, reused unchanged below). This is legal, unambiguous
# Julia multiple dispatch (distinct arity from the existing 3-positional-arg methods, not an
# override) -- both call surfaces coexist.
#
# Why this refactor was smaller than the prior session's own flag suggested: the prior session
# (profiled-outer-production-readiness-2026-08-03) found `evaluate_profiled_originzc_point`/
# `evaluate_profiled_cmzc_point` take nu only via a closed-over FIXED `pes.nu_full` and flagged
# changing that as a structural blocker needing user sign-off. Confirmed live this session: the
# DEEPER kernels `reduced_originzc_base_state`/`reduced_meanzc_base_state`
# (profiled_reduced_originzc_lookup_kernels_2026-08-02.jl / profiled_reduced_meanzc_lookup_kernels_2026-08-02.jl)
# ALREADY take `νfull`/`νvec` as a plain explicit positional argument -- the fixed-nu constraint
# was only ever in this thin wrapper layer, not the inner solve machinery. No inner-kernel changes
# were needed, only this wrapper.
#
# nu is a PLAIN explicit function argument on every call here, never closed over any mutable
# state -- there is no generation-ID/staleness risk in the point-eval call itself (each call gets
# its own eta_nu vector). `nu_generation_id` below is the seed for cache-key/dual-bank/checkpoint
# generation tagging (task §8.4) -- NOT wired into any cache/dual-bank/checkpoint consumer this
# session (real remaining integration work, tracked in MASTER.md), added now so this layer has a
# stable notion of "which nu generation" from day one rather than retrofitting it later.
#
# The analytic eta gradient reuses `d_delta_dual_d_eta_origin_vec` (cm_originzc_moments.jl)
# UNCHANGED -- that function's own docstring states it reduces, under `SharedByPowerLayout`, to
# `d_delta_dual_d_eta_nu_vec`'s formula, so ONE function serves both origin-by-power (origin_ZC)
# and shared-by-power (CM+ZC, in its "one safe config") layouts; not re-derived here.
# ============================================================================

isdefined(Main, :evaluate_profiled_originzc_point) ||
    error("profiled_zc_free_eta_2026-08-04.jl requires profiled_zc_lane_point_evaluators_2026-08-02.jl to be included first.")
isdefined(Main, :d_delta_dual_d_eta_origin_vec) ||
    error("profiled_zc_free_eta_2026-08-04.jl requires cm_originzc_moments.jl to be included first.")
isdefined(Main, :n_eta) ||
    error("profiled_zc_free_eta_2026-08-04.jl requires cm_originzc_target_layout.jl to be included first.")

# ---------------------------------------------------------------------------
# Combined outer-coordinate layout: [gp; profiled retained-A; eta_nu] (task §8.3).
# ---------------------------------------------------------------------------

"""
    ZCFreeNuOuterLayout(n_econ_outer, n_eta_dim)

Combined outer-coordinate layout for a free-nu ZC family: `[gp; profiled retained-A; eta_nu]`.
`economic_outer_range` = `1:n_econ_outer` (gp + retained-A, byte-identical in meaning to every
fixed-nu REDUCED family's own `w_profiled`/`outer_dim_profiled(pe)`); `eta_nu_outer_range` =
`n_econ_outer+1 : n_econ_outer+n_eta_dim`, appended at the tail.

Deliberately a DIFFERENT typed range from `CMZCFamilyCtx.n_lambda_meanpair` (the INNER Z mean/pair
DUAL-coordinate width) -- that struct's own docstring already warns about this exact name
collision (`n_eta(zc_layout)` vs `n_lambda_meanpair`, a real bug caught live 2026-08-02). Two
separate typed ranges here so a caller can never index one with the other's width.
"""
struct ZCFreeNuOuterLayout
    economic_outer_range::UnitRange{Int}
    eta_nu_outer_range::UnitRange{Int}
end
function ZCFreeNuOuterLayout(n_econ_outer::Int, n_eta_dim::Int)
    n_econ_outer >= 1 || error("ZCFreeNuOuterLayout: n_econ_outer must be >= 1, got $n_econ_outer")
    n_eta_dim >= 1 || error("ZCFreeNuOuterLayout: n_eta_dim must be >= 1, got $n_eta_dim")
    return ZCFreeNuOuterLayout(1:n_econ_outer, (n_econ_outer + 1):(n_econ_outer + n_eta_dim))
end

"split_free_nu_outer(w_full, layout) -> (w_econ, eta_nu) -- splits the combined outer vector."
split_free_nu_outer(w_full::AbstractVector{Float64}, layout::ZCFreeNuOuterLayout) =
    (w_full[layout.economic_outer_range], w_full[layout.eta_nu_outer_range])

"pack_free_nu_outer(w_econ, eta_nu) -> Vector{Float64} -- inverse of split_free_nu_outer."
pack_free_nu_outer(w_econ::AbstractVector{Float64}, eta_nu::AbstractVector{Float64}) = vcat(w_econ, eta_nu)

"""
    nu_generation_id(eta_nu) -> UInt64

Cheap content hash identifying which nu generation an eta_nu vector belongs to -- the primitive
task §8.4's cache-key/dual-bank-compatibility/checkpoint generation tagging would build on. Rounds
to 12 decimal digits first so bit-noise from KNITRO's own internal representation does not spuriously
mint a new generation for what is numerically the same eta_nu.
"""
nu_generation_id(eta_nu::AbstractVector{Float64}) = hash(round.(eta_nu; digits = 12))

# ---------------------------------------------------------------------------
# origin_ZC: free eta_nu point evaluator + gradient
# ---------------------------------------------------------------------------

"""
    evaluate_profiled_originzc_point(w_econ, eta_nu, fctx::OriginZCFamilyCtx, pes::OriginZCPointEvalState;
        maxit_override=nothing) -> NamedTuple

Free-nu origin-ZC point evaluator: `eta_nu` (length `n_eta(fctx.zc_layout)`, log-nu units) is a
PLAIN explicit argument, decoded here as `nu_full = exp.(eta_nu)` -- `pes.nu_full` is IGNORED by
this method (kept only so `OriginZCPointEvalState` does not need a second, nu-less struct; `pes`
still supplies the mutable per-solve `octx` workspace, which is legitimately reused/mutated across
calls, unlike nu). SAME output shape as the fixed-nu 3-arg method, PLUS `eta_nu` itself and
`st.zc_layout`/`st.n_econ` (needed by `reduced_originzc_outer_gradient_with_eta` below, which the
fixed-nu method's own `st_for_gradient` does not carry).
"""
function evaluate_profiled_originzc_point(w_econ::AbstractVector{Float64}, eta_nu::AbstractVector{Float64},
        fctx::OriginZCFamilyCtx, pes::OriginZCPointEvalState; maxit_override::Union{Nothing,Int} = nothing)
    length(eta_nu) == n_eta(fctx.zc_layout) ||
        error("evaluate_profiled_originzc_point: length(eta_nu)=$(length(eta_nu)) != n_eta(fctx.zc_layout)=$(n_eta(fctx.zc_layout))")
    nu_full = exp.(eta_nu)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_econ), ctx, fctx.pe)
    r = reduced_originzc_base_state(decoded.xf, ctx, fctx.layout, pes.octx, nu_full)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    cf_solved = r.st.core_cf_ref[]
    n_econ = fctx.layout.total_reduced_economic_moments
    op = r.st.op
    β_econ = @view r.λstar[1:n_econ]
    λ_mean = @view r.λstar[n_econ+1:n_econ+n_mean(op)]
    λ_pair = @view r.λstar[n_econ+n_mean(op)+1:n_econ+n_mean(op)+n_pair(op)]
    ov = verify_inner_solution_reduced_originzc!(r.ζstar, β_econ, λ_mean, λ_pair,
        cf_solved, ctx, θ_full, fctx.layout, op, r.st.zc_layout, nu_full, r.obj, cf_solved.W)
    m_weights, verify = verify_namedtuple_from_operator(ov, r.obj, cf_solved.W, r.inner_status)

    # mirrors the fixed-nu method's own documented cf-vs-getproperty bugfix (2026-08-02) -- same
    # reason, same fix, ADDITIONALLY carrying zc_layout/n_econ for the eta gradient below.
    st_for_gradient = (cf = r.st.core_cf_ref[], layout = fctx.layout, zc_layout = r.st.zc_layout, n_econ = n_econ)

    result = merge(verify, (zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess))
    return (result = result, obj = r.obj, st = st_for_gradient, m_weights = m_weights, theta_full = θ_full,
            decoded = decoded, nu_full = nu_full, eta_nu = collect(Float64, eta_nu),
            nu_generation = nu_generation_id(eta_nu))
end

"""
    reduced_originzc_outer_gradient_with_eta(w_econ, eta_nu, ctx, fctx::OriginZCFamilyCtx, ev) -> (g_ext, meta)

`vcat(g_econ, d_eta)`, mirroring FULL's `cm_originzc_production_gradient` combined-gradient
convention exactly. `g_econ` = `shared_family_outer_gradient` (the SAME one-method A/gp engine
every family uses, un-rederived); `d_eta` = `d_delta_dual_d_eta_origin_vec` (FULL's own analytic
envelope-theorem eta sensitivity, un-rederived) evaluated at `ev`'s solved dual, using ONLY fields
`evaluate_profiled_originzc_point`'s eta-explicit method above populates (`ev.st.zc_layout`,
`ev.st.n_econ`, `ev.nu_full`, `ev.result.beta`, `ev.result.m_mean`).
"""
function reduced_originzc_outer_gradient_with_eta(w_econ::AbstractVector{Float64}, eta_nu::AbstractVector{Float64},
        ctx, fctx::OriginZCFamilyCtx, ev; threaded::Bool = false)
    g_econ, meta = shared_family_outer_gradient(w_econ, ctx, fctx, ev; threaded = threaded)
    # aug.ncore_econ in d_delta_dual_d_eta_origin_vec's own convention (cm_originzc_moments.jl:257,
    # `ncore_econ = obj0.d`) is `n_econ_duals + 1`, NOT n_econ_duals itself -- confirmed live
    # 2026-08-04 via that file's own comment ("economic (ncore_econ-1) | mean_1(D) ...", line 26)
    # after a D4 FD gate caught a systematic one-coordinate shift in the eta block (FD[k] matched
    # analytic[k+1] exactly, the fingerprint of exactly this off-by-one). `ev.st.n_econ` is the
    # actual economic-dual COUNT (matches `fctx.layout.total_reduced_economic_moments`, the
    # convention `β_econ = r.λstar[1:n_econ]` above already uses) -- +1 here bridges to the
    # OTHER convention this borrowed FULL formula expects.
    aug_like = (layout = ev.st.zc_layout, ncore_econ = ev.st.n_econ + 1, Zraw_all = (Matrix{Float64}(undef, 0, ctx.D),))
    d_eta = d_delta_dual_d_eta_origin_vec(ev.result.beta, aug_like, ev.nu_full; mean_m = ev.result.m_mean)
    return vcat(g_econ, d_eta), meta
end

# ---------------------------------------------------------------------------
# CM+ZC: free eta_nu point evaluator + gradient
# ---------------------------------------------------------------------------

"""
    evaluate_profiled_cmzc_point(w_econ, eta_nu, fctx::CMZCFamilyCtx, pes::CMZCPointEvalState;
        maxit_override=nothing) -> NamedTuple

Free-nu CM+ZC point evaluator, same convention as origin-ZC's above. `st = r.st` is passed through
UNCHANGED (unlike origin-ZC, `ReducedCMMeanZCOperatorState` already carries `zc_layout`/`n_econ`
as direct fields -- no remapping needed for the gradient function below to reach them).
"""
function evaluate_profiled_cmzc_point(w_econ::AbstractVector{Float64}, eta_nu::AbstractVector{Float64},
        fctx::CMZCFamilyCtx, pes::CMZCPointEvalState; maxit_override::Union{Nothing,Int} = nothing)
    length(eta_nu) == n_eta(fctx.zc_layout) ||
        error("evaluate_profiled_cmzc_point: length(eta_nu)=$(length(eta_nu)) != n_eta(fctx.zc_layout)=$(n_eta(fctx.zc_layout))")
    nu_full = exp.(eta_nu)
    ctx = fctx.ctx
    decoded = decode_outer_profiled(collect(Float64, w_econ), ctx, fctx.pe)
    r = reduced_meanzc_base_state(decoded.xf, nu_full, ctx, fctx.layout, pes.cctx)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    cctx = pes.cctx
    bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
    ov = verify_inner_solution_reduced_cmzc!(r.ζstar, r.λstar, r.st.cf, ctx, θ_full, fctx.layout,
        r.st.zc_op, r.st.zc_layout, nu_full, cctx.L, length(cctx.origins), cctx.origins,
        cctx.refIndex1, bins_u, cctx.R, r.obj, r.st.cf.W)
    m_weights, verify = verify_namedtuple_from_operator(ov, r.obj, r.st.cf.W, r.inner_status)

    result = merge(verify, (zeta = r.ζstar, beta = r.λstar, n_fg_calls = r.n_fg, n_hess_calls = r.n_hess))
    return (result = result, obj = r.obj, st = r.st, m_weights = m_weights, theta_full = θ_full,
            decoded = decoded, nu_full = nu_full, eta_nu = collect(Float64, eta_nu),
            nu_generation = nu_generation_id(eta_nu))
end

"""
    reduced_cmzc_outer_gradient_with_eta(w_econ, eta_nu, ctx, fctx::CMZCFamilyCtx, ev) -> (g_ext, meta)

Same convention as `reduced_originzc_outer_gradient_with_eta`.
"""
function reduced_cmzc_outer_gradient_with_eta(w_econ::AbstractVector{Float64}, eta_nu::AbstractVector{Float64},
        ctx, fctx::CMZCFamilyCtx, ev; threaded::Bool = false)
    g_econ, meta = shared_family_outer_gradient(w_econ, ctx, fctx, ev; threaded = threaded)
    # aug.ncore_econ in d_delta_dual_d_eta_origin_vec's own convention (cm_originzc_moments.jl:257,
    # `ncore_econ = obj0.d`) is `n_econ_duals + 1`, NOT n_econ_duals itself -- confirmed live
    # 2026-08-04 via that file's own comment ("economic (ncore_econ-1) | mean_1(D) ...", line 26)
    # after a D4 FD gate caught a systematic one-coordinate shift in the eta block (FD[k] matched
    # analytic[k+1] exactly, the fingerprint of exactly this off-by-one). `ev.st.n_econ` is the
    # actual economic-dual COUNT (matches `fctx.layout.total_reduced_economic_moments`, the
    # convention `β_econ = r.λstar[1:n_econ]` above already uses) -- +1 here bridges to the
    # OTHER convention this borrowed FULL formula expects.
    aug_like = (layout = ev.st.zc_layout, ncore_econ = ev.st.n_econ + 1, Zraw_all = (Matrix{Float64}(undef, 0, ctx.D),))
    d_eta = d_delta_dual_d_eta_origin_vec(ev.result.beta, aug_like, ev.nu_full; mean_m = ev.result.m_mean)
    return vcat(g_econ, d_eta), meta
end
