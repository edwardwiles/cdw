# ============================================================================
# Claude Code task 2026-08-01, §7/§9: the profiled unrestricted OUTER
# evaluator + analytic ("C+") outer gradient, spliced as ONE parameterization
# branch (economic_parameterization = :profiled_destination_scales) rather
# than a parallel campaign pipeline -- reuses, unmodified:
#   - decode_outer_profiled/reduce_to_w_profiled (outer_coordinate_layout_profiled_2026-07-31.jl)
#   - build_profiled_operator_bundle/inner_loop_KNITRO_profiled (profiled_operator_bundle_2026-08-01.jl)
#   - reduced_homogeneous_dual_contraction/reduced_homogeneous_transpose_contraction!
#     (reduced_homogeneous_contraction_2026-08-01.jl)
#   - verify_inner_solution_reduced_profiled!/verify_namedtuple_from_operator
#     (reduced_operator_verification_2026-08-01.jl / operator_verification.jl)
#   - CS.reconstruct_full (cc_algo), the SAME xf->theta_full reconstruction the
#     production full-formulation path uses.
# The analytic gradient below implements PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md
# sections 3-4 exactly; every intermediate quantity it needs (kappa/Cbar, B/Tslot)
# is produced by calling the EXISTING reduced contraction kernels at the solved
# LFD -- no new O(W*D) accumulation kernel. ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :decode_outer_profiled) || error("profiled_outer_evaluator_2026-08-01.jl requires outer_coordinate_layout_profiled_2026-07-31.jl to be included first.")
isdefined(Main, :build_profiled_operator_bundle) || error("profiled_outer_evaluator_2026-08-01.jl requires profiled_operator_bundle_2026-08-01.jl to be included first.")
isdefined(Main, :verify_inner_solution_reduced_profiled!) || error("profiled_outer_evaluator_2026-08-01.jl requires reduced_operator_verification_2026-08-01.jl to be included first.")

"""
    build_profiled_ab_spec_pe(ctx; global_overrides=Dict{Int,Int}()) -> (spec, gauge, pe)

Builds the (AnchorSpec, gauge, PivotGravityElimOnRetained) triple ONCE per ctx,
using the GENUINE calibration z-matrix (ctx.θ0_up's own A_od block) for the
anchor gauge -- never the gravity-elimination pivot's zfree=0 reference point
(this repo's standing CLAUDE.md warning: A_od==1 is not calibration). Matches
the exact pattern already gated in
test_unrestricted_knitro_d20_omitrow_smallw_2026-08-01.jl.
"""
function build_profiled_ab_spec_pe(ctx; global_overrides::Dict{Int,Int} = Dict{Int,Int}())
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    spec = build_anchor_spec_from_ctx(ctx; global_overrides = global_overrides)
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gauge = build_anchor_gauge(z_calib, spec)
    pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
    return spec, gauge, pe
end

"""
    reduce_calibration_to_w_profiled(ctx, pe) -> Vector{Float64}

The calibration point in profiled coordinates -- `reduce_to_w_profiled` applied
to ctx's own θ0_up, i.e. the "economically equivalent" profiled start (task §14:
"The profiled start must be produced by reducing the exact full calibration point.").
"""
function reduce_calibration_to_w_profiled(ctx, pe::PivotGravityElimOnRetained)
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    θ0 = ctx.θ0_up
    z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
    gp0 = θ0[3+D]
    return reduce_to_w_profiled(gp0, z_calib, pe)
end

"""
    evaluate_profiled_point(w_profiled, ctx, pe; ref_obj=ctx.obj, maxit_override=nothing) -> NamedTuple

Full profiled per-point evaluator: decode -> reconstruct theta_full (via the
SAME CS.reconstruct_full the production full-formulation path uses, so `xf`
means the identical thing in both formulations) -> build reduced operator
bundle -> KNITRO inner solve -> independent operator verification. Returns a
NamedTuple with the SAME FIELD NAMES `classify_inner_result`/`is_verified_success`
(oracle.jl) require (`inner_status`, `Delta_dual`, `primal_dual_gap`,
`mean_m_resid`, `max_abs_moment_kkt_resid`, `m_min`, ...), PLUS the extra
state (`obj`, `st`, `m_weights`, `spec`, `pe`, `theta_full`) the gradient
function below needs -- so a caller gets one evaluator call per point, not two.
"""
function evaluate_profiled_point(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained; ref_obj = ctx.obj, maxit_override::Union{Nothing,Int} = nothing)
    decoded = decode_outer_profiled(collect(Float64, w_profiled), ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)

    obj_p, st_p = build_profiled_operator_bundle(ctx, θ_full, spec; ref_obj = ref_obj)
    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_profiled(obj_p, st_p; maxit_override = maxit_override)

    zeta = x[1]; beta = x[2:end]
    ov = verify_inner_solution_reduced_profiled!(zeta, beta, st_p.cf, ctx, θ_full, st_p.layout, obj_p, st_p.cf.W)
    m_weights, verify = verify_namedtuple_from_operator(ov, obj_p, st_p.cf.W, nStatus)

    gravity_value = gravity_from_logz(decoded.z_full, ctx)

    result = merge(verify, (gravity_value = gravity_value, n_fg_calls = n_fg, n_hess_calls = n_hess,
        x_sol = x, zeta = zeta, beta = beta, winner_checksum = sum(Float64.(st_p.cf.winner))))
    return (result = result, obj = obj_p, st = st_p, m_weights = m_weights, theta_full = θ_full, decoded = decoded)
end

"""
    profiled_outer_gradient(w_profiled, ctx, spec, pe, ev) -> Vector{Float64}

The analytic ("C+") profiled outer gradient, `PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md`
sections 3-4. `ev` is the NamedTuple `evaluate_profiled_point` returns for THIS
`w_profiled` (must be the same point -- reuses its solved theta_full/obj/st/m_weights,
exactly mirroring how production's cb_G! reuses cb_F!'s `last_F_state`/`base`
rather than resolving). Returns a vector of length `outer_dim_profiled(pe)`
(361 at real D20): `grad[1] = dK*/dgp`, `grad[2:end] = dK*/d(r_free)`.
"""
function profiled_outer_gradient(w_profiled::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, ev)
    st = ev.st; obj = ev.obj; θ_full = ev.theta_full; m_weights = ev.m_weights
    cf = st.cf; layout = st.layout
    D = cf.D; Ddest = cf.D_dest; W = cf.W
    μ = θ_full[1]; σ = θ_full[2]
    e_exponent = μ * (σ - 1)
    M = obj.M

    # ---- kappa[o,slot]/Cbar[slot] (§3a) -- EXACT re-derivation of
    # reduced_homogeneous_dual_contraction's own local kappa/Cbar (that function does not
    # expose them, so recomputing here is cheap -- O(length(retained_full_factual_j)) -- and
    # additive-only; no change to that file needed).
    β = ev.result.beta
    κ = zeros(D, Ddest)
    Cbar = zeros(Ddest)
    @inbounds for k in eachindex(layout.retained_full_factual_j)
        o = layout.retained_origin[k]; slot = layout.retained_slot[k]
        j_full = layout.retained_full_factual_j[k]
        kk = β[k] * cf.nrm[j_full] * cf.gdiv[j_full]
        κ[o, slot] = kk
        Cbar[slot] += kk * cf.Pmat[o, slot]
    end

    has_france = layout.france_ratio_reduced_j > 0
    κ_cf = 0.0; gpσ = 0.0; bi_slot = 0
    if has_france
        bi = ctx.bi; bi_slot = dest_slot(ctx, bi)
        gp = w_profiled[1]
        gpσ = gp^σ
        j_cf_full = cf.cf_col
        κ_cf = β[layout.france_ratio_reduced_j] * cf.nrm[j_cf_full] * cf.gdiv[j_cf_full]
    end

    # ---- B[r,d]/Tslot[d] at the SOLVED LFD (§3a/§3c) -- reuse the EXISTING transpose
    # contraction kernel unchanged, called with weights=m_weights instead of dPsi(q) at an
    # arbitrary point; B/Tslot are caller-owned scratch that the kernel leaves populated after
    # return (it never clears them post-loop), so they are read directly here.
    B = zeros(D, Ddest); Tslot = zeros(Ddest)
    v_dummy = zeros(layout.total_reduced_economic_moments)
    reduced_homogeneous_transpose_contraction!(v_dummy, m_weights, cf, ctx, θ_full, layout, B, Tslot)

    # ---- g_full[pos], pos=1..n_retained(spec), SAME order as pe.cr/other_pos/pivot_pos
    # (retained_linear_indices(spec), i=o+(d-1)*D convention) -- §3a/§3b.
    ridx = retained_linear_indices(spec)
    n_ret = length(ridx)
    g_full = zeros(n_ret)
    @inbounds for pos in 1:n_ret
        i = ridx[pos]
        d = div(i - 1, D) + 1
        r = i - (d - 1) * D
        Cbar_eff = Cbar[d] + (has_france && d == bi_slot ? κ_cf * gpσ : 0.0)
        g_full[pos] = -e_exponent * (κ[r, d] - Cbar_eff) * B[r, d] / M
    end

    # ---- gp component (§3c)
    grad1 = has_france ? κ_cf * σ * gp_pow_sigma_minus1(w_profiled[1], σ) * Tslot[bi_slot] / M : 0.0

    # ---- gravity-pivot chain rule (§4)
    other_pos = pe.other_pos; pivot_pos = pe.pivot_pos; cr = pe.cr
    n_free = length(other_pos)
    g_free = Vector{Float64}(undef, n_free)
    @inbounds for k in 1:n_free
        g_free[k] = g_full[other_pos[k]] - g_full[pivot_pos] * cr[other_pos[k]] / cr[pivot_pos]
    end

    return vcat(grad1, g_free)
end

"gp^(σ-1), split out only so a σ==1 degenerate edge case is a single obvious place to special-case if ever needed (not expected to occur -- sigma is always >1 in this model's calibration)."
gp_pow_sigma_minus1(gp::Float64, σ::Float64) = gp^(σ - 1)
