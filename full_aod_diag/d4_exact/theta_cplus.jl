# ============================================================================
# Fast, exact fixed-dual theta secant (task: "replace brute-force flexible-theta secants with a
# C+-style fixed-dual theta derivative", 2026-07-26). See docs/THETA_CPLUS_MATHEMATICAL_
# DERIVATION_2026-07-26.md for the full math this is built on.
#
# Replaces the OLD theta_fixed_dual_delta_pivot_A path (flexible_theta_aspace_production.jl),
# which computed each probe via the GENERIC dense obj.moments!/CS.reconstruct_full/obj(x) pipeline
# (~1.4s / ~743MB per probe at real D=20/W=80,000, per
# docs/FLEXIBLE_THETA_DERIVATIVE_PERFORMANCE_ANALYSIS_2026-07-26.md), with a call into the
# ALREADY-EXISTING, ALREADY-VALIDATED compressed winner-form pipeline
# (compressed_factual_buffer_reuse.jl::build_compressed_factual! + compressed_cc_inner.jl::
# compressed_cc_value_grad) that every other cheap-scoring path in this codebase (DualBank's
# cheap_score, the Hessian-callback adapter) already uses instead of the dense reconstruction.
#
# ADDENDUM COMPLIANCE (mid-task user instruction): this remains an EXACT fixed-dual central
# secant. build_compressed_factual! performs a full, exact re-scan of every origin at every
# (draw,destination) at whatever theta it is given -- no runner-up/top-3/largest-U shortcut, no
# stability-radius-gated partial rescan, no optional analytic stable-winner backend. The speedup
# comes entirely from routing through the O(W*D^2)-compute/O(W*Ddest)-memory compressed pipeline
# instead of the O(W*382)-with-dense-allocation generic one, plus eliminating redundant work
# around it (no outer-vector copy, no repeated pivot-cache rebuild, no third unused reconstruction
# -- verified dead: see PsiObjectiveBundleImplicitMethodB_fullA.jl::inner_loop_internal, which
# unconditionally rebuilds ctx.obj.H fresh at the START of every subsequent inner solve, and
# hessian! (the only reader of obj.H) only fires inside that inner-solve context, never at the
# outer-gradient-callback level -- cb_G!'s post-secant obj.H write was already fully superseded
# before anything could read it).
# ============================================================================

isdefined(Main, :build_compressed_factual!) || error("theta_cplus.jl requires compressed_factual_buffer_reuse.jl to already be included.")
isdefined(Main, :compressed_cc_value_grad) || error("theta_cplus.jl requires compressed_cc_inner.jl to already be included.")
isdefined(Main, :pivot_expand_cheap) || error("theta_cplus.jl requires gravity_elimination.jl to already be included.")

"""
    ThetaCPlusWorkspace

Campaign-lifetime scratch for the fast theta secant. Two independent `CompressedFactualWorkspace`
buffers (plus/minus) so both perturbed winner states can be compared (diagnostic
`n_winner_changes`) without either overwriting the other before the comparison -- the only
W-scale state this workspace owns; both buffers are built ONCE (`build_theta_cplus_workspace`)
and reused in place every call (`build_compressed_factual!`'s own aliasing discipline).

Diagnostic counters (task §12): `generic_moments_calls` must read 0 after any real campaign that
exercises this path -- kept here rather than as a global so it is unambiguously scoped to one
workspace/ctx.
"""
mutable struct ThetaCPlusWorkspace
    cf_ws_plus::CompressedFactualWorkspace
    cf_ws_minus::CompressedFactualWorkspace
    h_theta::Float64
    n_calls::Int
    n_winner_changes_total::Int
    generic_moments_calls::Int
    check_ties::Bool
end

"One-time construction, matching `build_compressed_factual_workspace`'s own (D,Ddest,W) shape."
function build_theta_cplus_workspace(D::Int, Ddest::Int, W::Int; h_theta::Float64 = 1e-3, check_ties::Bool = true)
    return ThetaCPlusWorkspace(
        build_compressed_factual_workspace(D, Ddest, W),
        build_compressed_factual_workspace(D, Ddest, W),
        h_theta, 0, 0, 0, check_ties)
end

"""
    decode_theta_probe(eta_theta, gp, a_nonpivot, ctx, xy, pgc) -> xf

Cheap decode for a SINGLE perturbed `eta_theta`, holding `gp`/`a_nonpivot` fixed -- the theta-
secant analogue of `decode_outer_unified`, but taking scalars/a view instead of a full outer
vector `w`, and reusing the ALREADY-BUILT `pgc` directly (no internal `build_pivot_elimination_
cheap` rebuild, unlike the old `decode_and_expand_flexible_A`). Allocates only O(D*Ddest)-scale
vectors (z_nonpivot, Aod_levels, xf), never anything W-scale, and never copies the outer vector
`w` itself -- `a_nonpivot` is expected to be a view.
"""
function decode_theta_probe(eta_theta::Float64, gp::Float64, a_nonpivot::AbstractVector{Float64},
        ctx, xy::APivotXY, pgc::PivotGravityElimCache)
    tol = 1e-9 * max(1.0, abs(ctx.θ_lo[1]), abs(ctx.θ_hi[1]))
    if !(ctx.θ_lo[1] - tol <= eta_theta <= ctx.θ_hi[1] + tol)
        reject_point(eta_theta, "decode_theta_probe: eta_theta=$eta_theta outside configured box [$(ctx.θ_lo[1]), $(ctx.θ_hi[1])]")
    end
    theta = exp(eta_theta)
    mu = 1.0 / theta
    logX_nonpivot = vec(xy.logX)[pgc.other_idx]
    logY_nonpivot = vec(xy.logY)[pgc.other_idx]
    z_nonpivot = @. -theta * (a_nonpivot + logX_nonpivot) - logY_nonpivot
    Aod_levels = vec(exp.(pivot_expand_cheap(collect(z_nonpivot), pgc, mu)))
    return vcat(mu, gp, Aod_levels)
end

"""
    theta_cplus_probe_value(eta_theta, gp, a_nonpivot, ctx, xy, pgc, base, cf_ws; check_ties) -> Float64

`-f` (Delta_dual sign convention, matching the old `theta_fixed_dual_delta_pivot_A`) at a single
perturbed `eta_theta`, dual held FIXED at `base.ζstar`/`base.λstar`. Routes through the compressed
winner-form pipeline: `build_compressed_factual!` does the exact full re-scan (every origin, every
draw/destination) at the perturbed theta, in place into `cf_ws`'s buffers; `compressed_cc_value_
grad` evaluates the SAME fixed-dual objective formula the dense `obj(x)` functor computes, from
the compressed representation. `ctx.obj.moments!`/`CS.reconstruct_full`-on-the-dense-path is never
called (this is what `generic_moments_calls` tracks -- always 0 through this function).
"""
function theta_cplus_probe_value(eta_theta::Float64, gp::Float64, a_nonpivot::AbstractVector{Float64},
        ctx, xy::APivotXY, pgc::PivotGravityElimCache, base::BaseDualState,
        cf_ws::CompressedFactualWorkspace; check_ties::Bool = true)
    xf = decode_theta_probe(eta_theta, gp, a_nonpivot, ctx, xy, pgc)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    cf = build_compressed_factual!(cf_ws, θ_full, ctx; check_ties = check_ties)
    f, _, _, _, _ = compressed_cc_value_grad(base.ζstar, base.λstar, cf; Psi! = ctx.obj.Psi!, dPsi! = ctx.obj.dPsi!)
    return -f, cf
end

"""
    theta_cplus_secant(w, ctx, xy, pgc, base, ws::ThetaCPlusWorkspace) -> NamedTuple

Drop-in replacement for cb_G!'s
`(D_plus,D_minus)=theta_fixed_dual_delta_pivot_A(w_plus/minus,...); grad_eta_theta=(D_plus-D_minus)/(2h)`
block. No `copy(w)`/`w_plus`/`w_minus` -- `gp`/`a_nonpivot` are read via a view into `w`, only
`eta_theta` (a scalar) differs between the two probes.
"""
function theta_cplus_secant(w::AbstractVector{Float64}, ctx, xy::APivotXY, pgc::PivotGravityElimCache,
        base::BaseDualState, ws::ThetaCPlusWorkspace)
    eta0 = w[1]; gp = w[2]; a_nonpivot = @view w[3:end]
    D_plus, cf_plus = theta_cplus_probe_value(eta0 + ws.h_theta, gp, a_nonpivot, ctx, xy, pgc, base, ws.cf_ws_plus; check_ties = ws.check_ties)
    D_minus, cf_minus = theta_cplus_probe_value(eta0 - ws.h_theta, gp, a_nonpivot, ctx, xy, pgc, base, ws.cf_ws_minus; check_ties = ws.check_ties)
    grad_eta_theta = (D_plus - D_minus) / (2 * ws.h_theta)
    nchanges = 0
    @inbounds for i in eachindex(cf_plus.winner)
        cf_plus.winner[i] != cf_minus.winner[i] && (nchanges += 1)
    end
    ws.n_calls += 1
    ws.n_winner_changes_total += nchanges
    return (D_plus = D_plus, D_minus = D_minus, grad_eta_theta = grad_eta_theta, n_winner_changes = nchanges)
end
