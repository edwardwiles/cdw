# ============================================================================
# Claude Code task 2026-08-01, §9 (v2, per live user correction): the
# profiled "C+" outer gradient, kept SURGICAL -- reuses the existing,
# already-gated kernels UNCHANGED (`build_compressed_factual`,
# `reduced_homogeneous_dual_contraction`, `CS.Psi!`, `CS.reconstruct_full`,
# `decode_outer_profiled`) for a fixed-dual (zeta*,beta* held at the just-
# solved optimum) central finite difference over every profiled outer
# coordinate. No new winner-selection/incremental-update logic of any kind --
# every probe rebuilds the compressed factual from scratch via the SAME
# function the inner solve itself already uses, so there is no separate
# formula to get wrong.
#
# This deliberately replaces an earlier draft of this file that reimplemented
# production's O(1) incremental-winner-update cache (LFixBaseCache-style) for
# the reduced moment basis -- that draft failed its own D4 gate
# (cos_sim~0.03 vs ground-truth re-solved FD) and, per live user feedback,
# was far more custom code than the task warranted ("the exact same gradient
# METHOD... just with the marginal adaptation for the changes in
# parameterization and the inner loop moments" -- fixed-dual central FD is
# the method; incremental O(1) winner caching is a PERFORMANCE optimization
# on top of it, not the method itself, and is what this file skips).
#
# Cost: O(W*D*Ddest) per probe (one `build_compressed_factual` rebuild),
# same order as production's own `:block_local` tier (`dest_contrib_block_local`,
# composite_gradient.jl) -- correct but not exploiting incremental sparsity.
# Left as an explicit, documented performance gap (task's own decision-rule
# framing already anticipates "is the rewrite worth it" -- an honest
# wall-clock characteristic of this diagnostic, not a defect to hide).
# ADDITIVE ONLY.
# ============================================================================

isdefined(Main, :evaluate_profiled_point) || error("profiled_outer_gradient_fd_2026-08-01.jl requires profiled_outer_evaluator_2026-08-01.jl to be included first.")

"""
    profiled_lfix_at(w, ctx, pe, layout, β, ζ, obj) -> Float64

Fixed-dual L_fix (== Delta_dual's own sign convention, `-(mean(Psi(q))+zeta)`)
at outer point `w` (profiled coordinates), with the dual `(zeta,β)` held FIXED
at the base point's solved values. Full rebuild via the unchanged production
`build_compressed_factual` + this session's already-gated
`reduced_homogeneous_dual_contraction` -- zero new economic-moment logic.
"""
function profiled_lfix_at(w::AbstractVector{Float64}, ctx, pe::PivotGravityElimOnRetained,
        layout::ProfiledEconomicMomentLayout, β::AbstractVector{Float64}, ζ::Float64, obj)
    decoded = decode_outer_profiled(w, ctx, pe)
    θ_full = CS.reconstruct_full(decoded.xf, ctx.m)
    cf = build_compressed_factual(θ_full, ctx; check_ties = false)
    t = reduced_homogeneous_dual_contraction(β, cf, ctx, θ_full, layout)
    q = -ζ .- t
    Psi_q = similar(q)
    obj.Psi!(Psi_q, q)
    return -(sum(Psi_q) / obj.M + ζ)
end

"""
    profiled_composite_gradient_at(w0, ctx, spec, pe, ev; h=0.01) -> (g, meta)

Central finite difference of `profiled_lfix_at` over every coordinate of the
profiled outer vector (`gp` included -- gp is smooth, no winner-switching
issue, so FD is exact for it up to ordinary truncation error; not special-cased
to a separate analytic formula, keeping this file to one code path). `ev` is
`evaluate_profiled_point`'s return at `w0` (reused dual solution, not
re-solved -- mirrors production's cb_G! reusing cb_F!'s last state).
"""
function profiled_composite_gradient_at(w0::AbstractVector{Float64}, ctx, spec::AnchorSpec,
        pe::PivotGravityElimOnRetained, ev; h::Float64 = 0.01)
    layout = ev.st.layout
    β = ev.result.beta; ζ = ev.result.zeta; obj = ev.obj
    n_total = outer_dim_profiled(pe)
    g = Vector{Float64}(undef, n_total)
    @inbounds for k in 1:n_total
        wp = copy(w0); wp[k] += h
        wm = copy(w0); wm[k] -= h
        Lp = profiled_lfix_at(wp, ctx, pe, layout, β, ζ, obj)
        Lm = profiled_lfix_at(wm, ctx, pe, layout, β, ζ, obj)
        g[k] = (Lp - Lm) / (2h)
    end
    return g, (w0 = collect(Float64, w0), h = h)
end
