isdefined(Main, :EconomicAGradientWorkspace) || include(joinpath(@__DIR__, "shared_a_gradient.jl"))   # shared-FG-verification-and-A-gradient release (2026-07-27): common-Frechet's DEFAULT (g,A_od)-block gradient backend, see cm_frechet_production_gradient below
isdefined(Main, :verify_inner_solution_operator_cm_frechet!) || include(joinpath(@__DIR__, "operator_verification.jl"))   # verification-defaults task (2026-07-27): archC_frechet_verified_state's :operator backend below

# ================================================================================================
# Fixed Fréchet as flexible CM plus a common-level anchor -- Part IV (outer gradient / C+ path).
#
# See docs/COMMON_FRECHET_GRADIENT_INTERPRETATION_2026-07-25.md. Summary: both the CM block AND
# the level block are THETA-INDEPENDENT (built once from the fixed baseline draws U, dM/dA=0 at
# fixed theta) -- exactly the property `lfix_cm_aware.jl`/`lfix_cm_cplus.jl` already exploit for
# CM's own restriction: `cm_fixed_contribution` computes lambda_C*'C_s ONCE (not re-evaluated per
# outer-coordinate probe) and folds it into the cached base dual scalar q0 via `with_q0`/`with_q0_C`
# BEFORE the coordinate loop runs -- the coordinate loop itself (`composite_gradient_at_Cplus_from_cache`)
# needs ZERO changes, since it never touches the CM/level tail of lambda* at all once q0 already
# reflects it.
#
# `cm_fixed_contribution` (lfix_cm_aware.jl:83) hardcodes `aug.ncm` as the FULL CM-only tail
# length and reshapes it `(nO, L)` -- for a :common_frechet aug, `aug.ncm = D*L != nO*L`, so it
# cannot be called unmodified (would throw a clear DimensionMismatch, not silently misinterpret
# data). This file adds the level-aware sibling, reusing `apply_contrast`/`suffix_sums`/
# `cumulative_forward_contribution!` (cm_lookup_kernels.jl) UNCHANGED for the CM part, adding one
# new small forward-contribution helper for the level part's different (all-D-origin SUM, not
# reference-differenced) structure -- and, because the level feature carries a NONZERO target
# (same root cause as Part III's Hessian bug), an extra constant term
# `sum(lambda_level .* level_targets)` that has no CM analog (CM's own target is always zero).
# ================================================================================================

"""
    frechet_level_forward_sum!(out, bins, D, λmat_ext)

`out[s] = sum_{o=1}^D λmat_ext[bins[s,o]]`, `λmat_ext` a length-`(L+1)` suffix-sum-extended vector
(column `L+1` == 0, the dropped/beyond-last-threshold bin -- same convention as
`interval_forward_contribution!`). Unlike that function, this one SUMS over all `D` origins rather
than differencing against a reference -- the level feature's own structure (task math doc §2:
`u=ones(D)/sqrt(D)`, symmetric across all origins). O(W*D), no loop over `L`.
"""
function frechet_level_forward_sum!(out::AbstractVector{Float64}, bins::AbstractMatrix{<:Unsigned}, D::Int,
                                     λmat_ext::AbstractVector{Float64})
    W = length(out)
    @inbounds for s in 1:W
        acc = 0.0
        for o in 1:D
            acc += λmat_ext[Int(bins[s, o])]
        end
        out[s] = acc
    end
    return out
end

"""
    frechet_cm_level_fixed_contribution(base::BaseDualState, ctx, aug, bins) -> Vector{Float64}

Level-aware analog of `lfix_cm_aware.jl::cm_fixed_contribution`: computes the FULL fixed
(theta-independent) contribution `lambda_tail*'M_s` per draw `s`, where the tail
`lambda_tail = base.λstar[aug.ncore : aug.ncore-1+aug.ncm]` (length `aug.ncm = D*L`) splits into
`lambda_cm` (first `ncm_cm=(D-1)*L`) and `lambda_level` (last `ncm_level=L`). Harmonization
(2026-07-29): the CM part now calls the SAME shared `cm_fixed_value_contribution`
(lfix_cm_aware.jl) `cm_fixed_contribution` itself calls, rather than an inlined verbatim copy of
that computation -- `cm_fixed_contribution` couldn't be called directly because it hardcoded
`aug.ncm` as its own tail-slicing bound (this file's `aug.ncm = ncm_cm+ncm_level != ncm_cm`);
slicing `λ_cm` here first and passing it to the shared value kernel removes that obstacle. The
level part is genuinely new (see `frechet_level_forward_sum!` above), including the constant
target-correction term `sum(lambda_level .* aug.level_targets)` -- structurally the gradient-side
analog of Part III's Hessian target-correction terms (an additive constant in the level feature
contributes an additive-constant term to `lambda_level'*level_s`, unlike CM's own always-zero-target
columns).
"""
function frechet_cm_level_fixed_contribution(base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned})
    ncore = aug.ncore; ncm_cm = aug.ncm_cm; ncm_level = aug.ncm_level; L = aug.L
    nO = length(aug.origins)
    D = ctx.D
    @assert length(base.λstar) >= ncore - 1 + ncm_cm + ncm_level "base.λstar too short for aug's (ncore,ncm_cm,ncm_level) -- was base solved against aug.obj_cm?"
    # 2026-08-06 (levelpow kernel task): `nf==2` means BOTH ncm_cm and ncm_level are TWO-FAMILY-
    # widened (ncm_cm=2*nO*L, ncm_level=2*L) -- this function used to hardcode the single-family
    # widths (`reshape(λ_cm,nO,L)`/`reshape(λ_level,1,L)` on the FULL, now-doubled slices), which
    # threw exactly the DimensionMismatch a real public-driver outer-gradient call hit live
    # (caught by this session's own W20k common-Fréchet two-family smoke, not by the D4/D20
    # calibration-only FG gates above, which never exercise this C+ envelope-theorem path). Fixed
    # by splitting each tail into its cdf/pow halves BEFORE slicing into the single-family kernel,
    # mirroring lfix_cm_aware.jl::cm_fixed_value_contribution_two_family's own accepted pattern for
    # the CM block's own eq.36 term (a direct, unoptimized matvec against aug.CM's already-built
    # pow sub-block, done ONCE per outer point -- cheap, not a per-Newton-iteration cost). Unlike
    # that CM-block precedent, no separate target-correction term is needed for the levelpow half:
    # `precalc_frechet_levelpow_dense` (cm_frechet_level.jl) already subtracts levelpow_targets
    # INTO aug.CM's own levelpow columns at construction time, so a plain matvec against them
    # already yields the fully target-corrected contribution.
    nf = hasproperty(aug, :n_families) ? aug.n_families : 1
    R = aug.contrasts == :orthonormal ? orthonormal_contrast_matrix(D) : nothing

    λ_cm_full = base.λstar[ncore : ncore - 1 + ncm_cm]
    ncm_cdf = nf == 2 ? aug.ncm_cdf : ncm_cm
    λ_cm_cdf = @view λ_cm_full[1:ncm_cdf]
    cm_out = cm_fixed_value_contribution(λ_cm_cdf, nO, L, aug.refIndex1, aug.origins, bins, R)
    if nf == 2
        ncm_pow = aug.ncm_pow
        λ_cm_pow = @view λ_cm_full[ncm_cdf+1:ncm_cdf+ncm_pow]
        CM_pow = @view aug.CM[:, ncm_cdf+1:ncm_cdf+ncm_pow]
        cm_out = cm_out .+ CM_pow * λ_cm_pow
    end

    λ_level_full = base.λstar[ncore + ncm_cm : ncore - 1 + ncm_cm + ncm_level]
    ncm_level_cdf = nf == 2 ? aug.ncm_level_cdf : ncm_level   # == L either way
    λ_level_cdf = @view λ_level_full[1:ncm_level_cdf]
    P_level_mat = suffix_sums(reshape(λ_level_cdf, 1, L))   # 1 x (L+1), column L+1 == 0
    invsqrtD = 1.0 / sqrt(D)
    level_out = Vector{Float64}(undef, size(bins, 1))
    frechet_level_forward_sum!(level_out, bins, D, vec(P_level_mat))
    level_out .*= invsqrtD
    level_out .-= sum(λ_level_cdf .* aug.level_targets[1:ncm_level_cdf])   # constant target-correction term (see docstring)
    if nf == 2
        ncm_level_pow = aug.ncm_level_pow
        λ_level_pow = @view λ_level_full[ncm_level_cdf+1:ncm_level_cdf+ncm_level_pow]
        levelpow_col_off = ncm_cm + ncm_level_cdf   # aug.CM layout: [CM_cdf|CM_pow|level_cdf|level_pow]
        CM_levelpow = @view aug.CM[:, levelpow_col_off+1:levelpow_col_off+ncm_level_pow]
        level_out = level_out .+ CM_levelpow * λ_level_pow
    end

    return cm_out .+ level_out
end

"""
    build_lfix_base_cache_cm_frechet_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCacheC

`:common_frechet` analog of `lfix_cm_cplus.jl::build_lfix_base_cache_cm_C!`. Identical structure --
calls `build_lfix_base_cache_C!` UNCHANGED (same reasoning: it only ever indexes
`base.λstar[1:D^2]`, silently and correctly ignoring the whole CM+level tail), then folds in
`frechet_cm_level_fixed_contribution` instead of plain `cm_fixed_contribution`.
"""
function build_lfix_base_cache_cm_frechet_C!(ws::LFixFactorizedWorkspace, x_free0::AbstractVector, ctx_cm,
                                              base::BaseDualState, ctx, aug, bins::AbstractMatrix{<:Unsigned};
                                              validate_dense::Bool = false)
    cache0 = build_lfix_base_cache_C!(ws, x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = frechet_cm_level_fixed_contribution(base, ctx, aug, bins)
    return with_q0_C(cache0, cache0.q0 .- contrib0)
end

"""
    archC_frechet_base_state(x_free0, ctx_cm, cctx, level_targets) -> BaseDualState

`:common_frechet` analog of `cm_production_bundle.jl::archC_base_state`, using
`archC_frechet_hess_cb_builder(cctx, level_targets)` (Part III) instead of `archC_hess_cb_builder(cctx)`.
"""
function archC_frechet_base_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx, level_targets::Vector{Float64})
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    # HISTORY: this function briefly set cctx.skip_cm_fill_ref[]=true whenever inner_fg_backend=
    # :cm_frechet_lookup (Phase 5.2, 2026-07-26), on the claim that the Hessian callback below never
    # reads obj.H's CM/level columns. That claim was found FALSE at the time (commit 5fd6347,
    # 2026-07-27 10:52: reproducible nStatus=-400 beyond the calibration point) and the skip was
    # removed -- CM/level dense columns filled unconditionally, exactly as the pre-Phase-5.2 code did.
    #
    # RE-INVESTIGATED 2026-07-27/28 (final-architecture-closure task, Goal 10): hypothesized the
    # winner-bin H_E,level Hessian path (winner_pair_cross_hessian_colsum!/_esum!, added in commits
    # e3bce93/d458702, which POSTDATE the nStatus=-400 bugfix commit 5fd6347 by ~7.5 hours, confirmed
    # via `git merge-base --is-ancestor 5fd6347 d458702`) might have made the skip safe again, since
    # neither `hessian_cm_frechet_structured!` nor `CMFrechetLookupState` were found (by static code
    # read) to depend on obj.H's CM/level columns anymore. D=4 multi-point re-test (calibration + 3
    # perturbed/hard points) supported this: all feasible, values agreed to 8-11 significant figures,
    # winner_bin genuinely engaged throughout. **A real D=20/W=80,000 re-test (destination_sample=
    # :exclude_row) then DISPROVED the hypothesis**: calibration agreed closely (nStatus -103 vs 0,
    # |Δzeta*|=4.4e-11, a benign KN_RC_FEAS_FTOL-vs-KN_RC_OPTIMAL label difference -- see
    # docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md), but BOTH tested non-calibration points
    # reproduced the EXACT original nStatus=-400 failure with the skip enabled, while the SAME points
    # solved cleanly (nStatus=0) with the fill left in place -- i.e. this codebase's own
    # Hessian/gradient path for common-Fréchet genuinely still depends on this dense fill somewhere
    # not caught by the static trace, exactly as the original 2026-07-26 finding said, and exactly
    # reproducing that finding's own "missed at the calibration point, caught beyond it" pattern. The
    # skip is REMOVED again here -- CM/level dense columns are filled UNCONDITIONALLY for common-
    # Fréchet, matching the pre-2026-07-26-Phase-5.2 and pre-this-reinvestigation behavior exactly.
    # Do not re-attempt this skip without first root-causing (not just re-testing) exactly which read
    # inside the actual Hessian/gradient callback chain depends on these columns -- a passing D=4-only
    # gate is NOT sufficient evidence, per this exact history repeating itself twice now.
    use_lookup = cctx.inner_fg_backend == :cm_frechet_lookup
    skip_fill_safe_frechet = false   # ALWAYS false for common-Fréchet -- see HISTORY comment above
    K, x, nStatus, n_fg, n_hess = use_lookup ?
        inner_loop_internal_cmfrechetlookup_production(obj, θ_full0, cctx, level_targets;
            hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, level_targets),
            skip_fill = skip_fill_safe_frechet) :
        inner_loop_internal_archgeneric(obj, θ_full0;
            hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, level_targets))
    nStatus in (0, -100, -101, -103) || throw(CMExpectedSolveFailure("archC_frechet_base_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    ζstar = x[1]; λstar = collect(x[2:end])
    return BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, copy(obj.arg1), nStatus)
end

"""
    composite_gradient_at_Cplus_frechet(x_free0, ctx_cm, pe, ctx, aug, bins, pool, ws, cctx; base=nothing, cache=nothing, kwargs...) -> (g, meta)

`:common_frechet` analog of `lfix_cm_cplus.jl::composite_gradient_at_Cplus_cm`. `base` defaults to
`archC_frechet_base_state` (requires `aug.level_targets`); everything downstream
(`composite_gradient_at_Cplus_from_cache`, the actual coordinate loop) is called UNCHANGED -- the
only new work needed for the outer-gradient path is computing the correct fixed q0 contribution
(`build_lfix_base_cache_cm_frechet_C!`), exactly per this file's header rationale.
"""
function composite_gradient_at_Cplus_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned},
        pool::GradWorkspacePool, ws::LFixFactorizedWorkspace, cctx;
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCacheC} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? archC_frechet_base_state(x_free0, ctx_cm, cctx, aug.level_targets) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet_C!(ws, x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_Cplus_from_cache(x_free0, ctx_cm, pe, pool, cache; base = base, kwargs...)
end

# ================================================================================================
# Reference (non-factorized) envelope backend -- the "reference envelope gradient" task §10/§11 asks
# the C+ path above to be validated against. Structural twin of `lfix_cm_aware.jl`'s
# `build_lfix_base_cache_cm`/`composite_gradient_at_fast_cm`, using `frechet_cm_level_fixed_contribution`
# in place of `cm_fixed_contribution`. `solve_base_state` (three_way_derivatives.jl) and
# `build_lfix_base_cache`/`composite_gradient_at_fast` are all ALREADY GENERIC on `ctx.obj` (no
# restriction-family-specific code in any of them) -- called completely unchanged, exactly the same
# "reuse, don't duplicate" pattern as the C+ side above.
# ================================================================================================

"""
    build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense=false) -> LFixBaseCache

`:common_frechet` analog of `lfix_cm_aware.jl::build_lfix_base_cache_cm`, for the Reference
(non-factorized) backend. `ctx_cm.obj` here is expected to be the DENSE Architecture-A Fréchet obj
(`build_cm_frechet_level_augmented_obj`'s `obj_cm`), matching how `build_lfix_base_cache_cm` itself
is normally paired with the dense reference obj rather than an Architecture-B/C one.
"""
function build_lfix_base_cache_cm_frechet(x_free0::AbstractVector, ctx_cm, base::BaseDualState,
                                           ctx, aug, bins::AbstractMatrix{<:Unsigned}; validate_dense::Bool = false)
    cache0 = build_lfix_base_cache(x_free0, ctx_cm, base; validate_dense = validate_dense)
    contrib0 = frechet_cm_level_fixed_contribution(base, ctx, aug, bins)
    return with_q0(cache0, cache0.q0 .- contrib0)
end

"""
    composite_gradient_at_fast_frechet(x_free0, ctx_cm, pe, ctx, aug, bins; base=nothing, cache=nothing, kwargs...) -> (g, meta)

`:common_frechet` analog of `lfix_cm_aware.jl::composite_gradient_at_fast_cm`. `base` defaults to
`solve_base_state(x_free0, ctx_cm)` (the plain dense inner solve, unchanged, generic on `ctx_cm.obj`).
"""
function composite_gradient_at_fast_frechet(x_free0::AbstractVector, ctx_cm, pe, ctx, aug, bins::AbstractMatrix{<:Unsigned};
        base::Union{Nothing,BaseDualState} = nothing, cache::Union{Nothing,LFixBaseCache} = nothing,
        validate_dense::Bool = false, kwargs...)
    base = base === nothing ? solve_base_state(x_free0, ctx_cm) : base
    if cache === nothing
        cache = build_lfix_base_cache_cm_frechet(x_free0, ctx_cm, base, ctx, aug, bins; validate_dense = validate_dense)
    end
    return composite_gradient_at_fast(x_free0, ctx_cm, pe; base = base, cache = cache, kwargs...)
end

# ================================================================================================
# Public-driver production bundle (Part V) -- level-aware siblings of cm_production_bundle.jl's
# archC_verified_state(_screened)/cm_production_value_verified_screened/cm_production_gradient(_cplus),
# with the SAME (x_free0, pcx, ...) call signature `run_cm_upper_checkpointed` (cm_checkpoint.jl)
# already uses for plain flexible CM, so that file's cb_F!/cb_G!/checkpoint-save call sites need
# only a marginal_restriction-keyed dispatch, not a rewrite. `cm_screen_precheck!` (cm_screen_bridge.jl)
# is reused UNCHANGED -- it operates on ctx_cm.pairwise/.m, entirely unrelated to which restriction
# family is active.
# ================================================================================================

"""
    archC_frechet_verified_state(x_free0, ctx_cm, cctx, level_targets) -> (base, verify)

Level-aware analog of `cm_production_bundle.jl::archC_verified_state`, using
`archC_frechet_hess_cb_builder(cctx, level_targets)` instead of `archC_hess_cb_builder(cctx)`.
Every other line (KKT residual, Delta_dual/Delta_primal, weight-norm checks) is IDENTICAL and
copied verbatim -- none of it is restriction-family-specific.
"""
function archC_frechet_verified_state(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx, level_targets::Vector{Float64};
        dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0,
        verification_backend::Symbol = CM_FRECHET_VERIFICATION_BACKEND_DEFAULT[])
    obj = ctx_cm.obj
    θ_full0 = CS.reconstruct_full(x_free0, ctx_cm.m)
    warm_label = :unset
    if dual_bank !== nothing
        x0, warm_label, _ = select_warm_start_restricted(dual_bank, obj, collect(x_free0))
        obj.x = x0
        warm_label == :neutral ? (RESTRICTED_DUAL_BANK_COUNTERS[].cold_inner_solves += 1) :
                                  (RESTRICTED_DUAL_BANK_COUNTERS[].warm_inner_solves += 1)
    end
    # RE-INVESTIGATED then REVERTED 2026-07-27/28 (final-architecture-closure task, Goal 10): the
    # POST-solve `:operator` verification branch below (`verify_inner_solution_operator_cm_frechet!`)
    # genuinely never reads `obj.H` (confirmed by reading it in full, operator_verification.jl) -- that
    # narrower claim is correct and is NOT what this comment is walking back. But the `skip_fill`
    # argument passed into `inner_loop_internal_cmfrechetlookup_production` below controls the SAME
    # underlying dense CM/level column fill that feeds the live INNER SOLVE's own Hessian callback
    # (`archC_frechet_hess_cb_builder`, identical mechanism to `archC_frechet_base_state`'s own call) --
    # a real D=20/W=80,000 re-test (see archC_frechet_base_state's own HISTORY comment and
    # docs/GOAL10_SKIP_CM_FILL_REF_REMOVAL_2026-07-27.md) found that skipping this fill reproduces the
    # original nStatus=-400 failure at non-calibration points. Kept at `false` unconditionally here for
    # the same reason -- do not re-enable without root-causing the actual dependency first.
    use_lookup_verify_frechet = cctx.inner_fg_backend == :cm_frechet_lookup
    skip_fill_verify_frechet = false   # ALWAYS false for common-Fréchet -- see HISTORY comment above and archC_frechet_base_state's own
    K, inner_x, nStatus, n_fg, n_hess = cctx.inner_fg_backend == :cm_frechet_lookup ?
        inner_loop_internal_cmfrechetlookup_production(obj, θ_full0, cctx, level_targets;
            hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, level_targets),
            skip_fill = skip_fill_verify_frechet) :
        inner_loop_internal_archgeneric(obj, θ_full0;
            hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(cctx, level_targets))
    if nStatus ∉ (0, -100, -101, -103)
        dual_bank !== nothing && warm_label != :neutral && (RESTRICTED_DUAL_BANK_COUNTERS[].warm_start_failures += 1)
        throw(CMExpectedSolveFailure("archC_frechet_verified_state: inner solve failed, nStatus=$nStatus (x_free0=$x_free0)"))
    end

    ζstar = inner_x[1]; λstar = collect(inner_x[2:end])
    W = size(obj.U, 1)

    local m_weights, verify
    if verification_backend === :operator
        # Verification-defaults task (2026-07-27): operator-based post-solve verification, G=[E|C|Level]
        # via the shared economic/CM-grid/level operators -- no dense obj.H read.
        cf = cctx.core_cf_ref[]
        cf isa CompressedFactual || error("archC_frechet_verified_state: verification_backend=:operator requires cctx.core_cf_ref[] to be a CompressedFactual (got $(typeof(cf))) -- prerequisite not met, refusing silent dense fallback")
        bins_u = cctx.Bidx isa Matrix{UInt32} ? cctx.Bidx : Matrix{UInt32}(cctx.Bidx)
        ov = verify_inner_solution_operator_cm_frechet!(ζstar, λstar, cf, cctx.L, cctx.nO, cctx.origins,
            cctx.refIndex1, bins_u, cctx.R, level_targets, obj, W;
            Pow = cctx.n_families == 2 ? cctx.Pow : nothing)
        m_weights, verify = verify_namedtuple_from_operator(ov, obj, W, nStatus)
    elseif verification_backend === :dense_reference
        G = CS.select_G_from_H(obj, obj.H)

        ncon = obj.d - obj.outer_constr_index + 2
        cbuf = zeros(ncon)
        obj(inner_x, constr = @view(cbuf[1:ncon]))
        Delta_dual = cbuf[1] / 1e10
        m_weights = copy(obj.arg1)
        # Allocation fix (shared outer-A-gradient task, 2026-07-27, task §10): non-allocating
        # weight_norm_resid -- see cm_production_bundle.jl's identical fix for the full rationale and
        # the bit-identity verification.
        s_m_weights = sum(m_weights)
        Delta_primal = primal_divergence(m_weights)

        mean_m_resid = abs(sum(m_weights) / W - 1.0)
        nkkt = min(length(λstar), size(G, 2))
        max_abs_moment_kkt_resid = kkt_residual_blas(G, m_weights, nkkt, W)

        verify = (inner_status = nStatus, Delta_dual = Delta_dual, Delta_primal = Delta_primal,
                  primal_dual_gap = abs(Delta_dual - Delta_primal),
                  weight_norm_resid = abs(sum(x -> x / s_m_weights, m_weights) - 1.0),
                  mean_m_resid = mean_m_resid, max_abs_moment_kkt_resid = max_abs_moment_kkt_resid,
                  m_mean = sum(m_weights) / W, m_min = minimum(m_weights), m_max = maximum(m_weights))
        record_dense_reference_verification!()
    else
        error("archC_frechet_verified_state: unknown verification_backend=:$verification_backend (expected :operator or :dense_reference)")
    end

    base = BaseDualState(collect(x_free0), θ_full0, ζstar, λstar, m_weights, nStatus)
    dual_bank !== nothing && record_success_restricted!(dual_bank, eval_id, collect(x_free0), inner_x)
    return base, verify
end

"""
    archC_frechet_verified_state_screened(x_free0, ctx_cm, cctx, level_targets; counters=nothing, use_witness=false) -> (base, verify)

Level-aware analog of `cm_screen_bridge.jl::archC_verified_state_screened`. `cm_screen_precheck!`
reused UNCHANGED.
"""
function archC_frechet_verified_state_screened(x_free0::AbstractVector, ctx_cm, cctx::CMBinHessCtx, level_targets::Vector{Float64};
                                                counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false,
                                                dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    cm_screen_precheck!(x_free0, ctx_cm; counters = counters, use_witness = use_witness)
    return archC_frechet_verified_state(x_free0, ctx_cm, cctx, level_targets; dual_bank = dual_bank, eval_id = eval_id)
end

"""
    cm_frechet_production_value_verified_screened(x_free0, pcx; counters=nothing, use_witness=false) -> (K, base, verify)

Level-aware analog of `cm_screen_bridge.jl::cm_production_value_verified_screened`, for a
`pcx = build_cm_frechet_production_context(...)` (`marginal_restriction=:common_frechet`, requires
`cm_hessian_backend=:structured` so `pcx.cctx !== nothing`).
"""
function cm_frechet_production_value_verified_screened(x_free0::AbstractVector, pcx;
                                                         counters::Union{Nothing,CMScreenCounters} = nothing, use_witness::Bool = false,
                                                         dual_bank::Union{Nothing,RestrictedDualBank} = nothing, eval_id::Int = 0)
    pcx.cctx === nothing && error("cm_frechet_production_value_verified_screened: pcx.cctx is nothing -- " *
        "requires cm_hessian_backend=:structured (the public driver's screened/verified path is Architecture-C-only).")
    base, verify = archC_frechet_verified_state_screened(x_free0, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets;
        counters = counters, use_witness = use_witness, dual_bank = dual_bank, eval_id = eval_id)
    K = pcx.ctx_cm.obj.H_save
    return K, base, verify
end

"""
    cm_frechet_production_gradient(x_free0, pcx, ctx, pe; gradient_backend=:shared_inplace_pooled,
                                    econ_ws=nothing, base=nothing, kwargs...) -> (g, meta)

Level-aware analog of `cm_production_bundle.jl::cm_production_gradient` (Reference/non-C+ backend),
for a `pcx = build_cm_frechet_production_context(...)`.

`gradient_backend` (shared-FG-verification-and-A-gradient release, 2026-07-27): mirrors
`cm_production_gradient`/`cm_originzc_production_gradient`/`cm_meanzc_production_gradient`'s own
kwarg exactly -- common-Frechet was the last of the CM-family restricted wrappers still hardcoded
to the allocating reference gradient.
  - `:shared_inplace_pooled` (DEFAULT): the shared `economic_A_gradient!` entry point
    (shared_a_gradient.jl). `build_lfix_base_cache_cm_frechet`'s own CM/level-folded `q0` is an
    `LFixBaseCache` (NOT `LFixBaseCacheC` -- that's the separate `:cplus`-only factorized type used
    by `cm_frechet_production_gradient_cplus` below), so it is passed through unchanged via
    `economic_A_gradient!`'s `cache=` kwarg -- same contract as the plain-CM wiring.
  - `:legacy_unbuffered`: the ORIGINAL, fully-allocating `composite_gradient_at_fast` -- kept ONLY
    as an explicit reference/debug backend.
"""
function cm_frechet_production_gradient(x_free0::AbstractVector, pcx, ctx, pe;
        base::Union{Nothing,BaseDualState} = nothing,
        gradient_backend::Symbol = :shared_inplace_pooled,
        econ_ws::Union{Nothing,EconomicAGradientWorkspace} = nothing, kwargs...)
    base = base === nothing ? archC_frechet_base_state(x_free0, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets) : base
    cache = build_lfix_base_cache_cm_frechet(x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    if gradient_backend === :shared_inplace_pooled
        D = pcx.ctx_cm.D; Ddest = hasproperty(pcx.ctx_cm, :D_dest) ? pcx.ctx_cm.D_dest : pcx.ctx_cm.D
        ws = econ_ws === nothing ? get_or_build_econ_a_grad_ws(cache.W) : econ_ws
        g_econ = zeros(D * Ddest)
        meta = economic_A_gradient!(g_econ, base, pcx.ctx_cm, pe, ws; cache = cache, kwargs...)
        return g_econ, meta
    elseif gradient_backend === :legacy_unbuffered
        return composite_gradient_at_fast(x_free0, pcx.ctx_cm, pe; base = base, cache = cache, kwargs...)
    else
        error("cm_frechet_production_gradient: gradient_backend must be :shared_inplace_pooled|:legacy_unbuffered, got $gradient_backend")
    end
end

"""
    cm_frechet_production_gradient_cplus(x_free0, pcx, ctx, pe, pool, ws; base=nothing, verify=nothing, kwargs...) -> (g, meta)

`:cplus`-backend level-aware analog of `cm_production_bundle.jl::cm_production_gradient_cplus`, the
production `cb_G!` entry point when `cm_gradient_backend=:cplus` (unchanged default) AND
`marginal_restriction=:common_frechet`. `pcx` is the SAME `build_cm_frechet_production_context(...)`
return value both gradient backends share.

2026-08-06 outer-production-closeout task, BUG FIX: on an UNMATCHED gradient call (`base===nothing`
-- a real, documented calling mode, not just an internal fallback: KNITRO does not guarantee `cb_G!`
is always called immediately after `cb_F!` at the identical point, and this function's own `base=`
kwarg exists specifically to let a caller skip re-solving when it IS matched), this used to fall
back to the BARE `archC_frechet_base_state` -- unlike `cm_meanzc_production_gradient_cplus`'s own
identical fallback, which correctly uses the VERIFIED `archC_meanzc_verified_state`. Under the
production-default operator FG backend (`CM_FRECHET_INNER_FG_BACKEND_DEFAULT[]=:cm_frechet_lookup`),
`CMFrechetLookupState`'s own FG functor writes its result into its OWN private `st.arg1` scratch,
NEVER into `obj.arg1` (confirmed by reading `cm_frechet_lookup_kernels.jl`/`cm_lookup_kernels.jl`/
`cm_meanzc_lookup_kernels.jl`'s functor bodies -- true of all three operator states, not
Fréchet-specific) -- so `archC_frechet_base_state`'s own `BaseDualState(..., copy(obj.arg1), ...)`
silently returns an ALL-ZERO `m_star` on this path. `gamma_component_analytic` (`lfix_factorized.jl`)
computes `d(Delta)/d(gp)` as proportional to `mean(m_star .* SW)` -- an all-zero `m_star` therefore
makes the analytic gp-gradient IDENTICALLY (bit-exact) zero, confirmed live: a central-FD check at
a real D20/W=20,000 point gave `d(Delta)/d(gp)=0.462` while the (unmatched-path) analytic gradient
returned exactly `0.0`. `archC_frechet_verified_state` independently computes `m_weights` via
operator-based verification (`verify_inner_solution_operator_cm_frechet!`), never reading
`obj.arg1` at all -- the same mechanism that makes CM+ZC's own gradient correct. Fixed by matching
`cm_meanzc_production_gradient_cplus`'s exact pattern.
"""
function cm_frechet_production_gradient_cplus(x_free0::AbstractVector, pcx, ctx, pe, pool::GradWorkspacePool,
        ws::LFixFactorizedWorkspace; base::Union{Nothing,BaseDualState} = nothing, verify = nothing, kwargs...)
    if base === nothing || verify === nothing
        base, verify = archC_frechet_verified_state(x_free0, pcx.ctx_cm, pcx.cctx, pcx.aug.level_targets)
    end
    cache = build_lfix_base_cache_cm_frechet_C!(ws, x_free0, pcx.ctx_cm, base, ctx, pcx.aug, pcx.bins)
    return composite_gradient_at_Cplus_from_cache(x_free0, pcx.ctx_cm, pe, pool, cache; base = base, kwargs...)
end
