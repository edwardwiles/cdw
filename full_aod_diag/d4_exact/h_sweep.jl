# ============================================================================
# Task §13: finite-difference step (h) experiments. For each method
# (frozen_adjoint_Q, fixed_dual_L, optimized_Delta) and each h in a
# configurable grid, records left/right/central slopes plus winner-switch
# counts/mass -- so h=0.1 (the sequential-method benchmark) can be evaluated
# as a MEASUREMENT, not assumed near-optimal, per the task's explicit
# instruction (sec 13: "Treat h=0.1 as a benchmark... not as an assumed
# optimum").
# ============================================================================

const DEFAULT_H_GRID = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]

"""
    h_sweep_one_direction(x_free0, v_free, ctx, base; h_grid=DEFAULT_H_GRID, common_random=true) -> Vector{NamedTuple}

Sweeps h for ONE direction v_free (common random numbers automatically
guaranteed here since ctx.U is fixed/shared across all evaluations -- task
§13's explicit requirement). Records, per h: left/right/central slopes for
all three methods, switch counts/mass (via `compute_winners` at +-h), and
elapsed time.
"""
function h_sweep_one_direction(x_free0::AbstractVector, v_free::AbstractVector, ctx, base;
        h_grid::Vector{Float64} = DEFAULT_H_GRID)
    xfun = t -> x_free0 .* exp.(t .* v_free)   # exact multiplicative/exponential family (matches winner_switching.jl)
    θ0 = CS.reconstruct_full(x_free0, ctx.m)
    winner0, _, _ = compute_winners(θ0, ctx)
    rows = NamedTuple[]
    for h in h_grid
        t0 = time()
        θp = CS.reconstruct_full(xfun(h), ctx.m); θm = CS.reconstruct_full(xfun(-h), ctx.m)
        winner_p, _, _ = compute_winners(θp, ctx); winner_m, _, _ = compute_winners(θm, ctx)
        switches_p = switch_stats(winner0, winner_p); switches_m = switch_stats(winner0, winner_m)

        Qp = frozen_adjoint_Q(xfun(h), ctx, base); Qm = frozen_adjoint_Q(xfun(-h), ctx, base); Q0 = frozen_adjoint_Q(x_free0, ctx, base)
        Lp = fixed_dual_L(xfun(h), ctx, base);     Lm = fixed_dual_L(xfun(-h), ctx, base);     L0 = fixed_dual_L(x_free0, ctx, base)
        Dp = optimized_Delta(xfun(h), ctx);        Dm = optimized_Delta(xfun(-h), ctx);        D0 = optimized_Delta(x_free0, ctx)

        elapsed = time() - t0
        push!(rows, (h = h,
            Q_left = (Q0-Qm)/h, Q_right = (Qp-Q0)/h, Q_central = (Qp-Qm)/(2h),
            L_left = (L0-Lm)/h, L_right = (Lp-L0)/h, L_central = (Lp-Lm)/(2h),
            D_left = (D0-Dm)/h, D_right = (Dp-D0)/h, D_central = (Dp-Dm)/(2h),
            n_switches_plus = switches_p.n_switches, n_switches_minus = switches_m.n_switches,
            switch_mass_plus = switches_p.base_mass, switch_mass_minus = switches_m.base_mass,
            elapsed = elapsed))
    end
    return rows
end

"""
    adaptive_h_candidate(x_free0, v_free, ctx; min_switch_mass=1/size(ctx.U,1), h_max=0.2) -> Float64

Task §13's adaptive-h rule (step 1-2 of the 5-point spec): use the exact tie
thresholds along the direction to find the smallest h crossing a configurable
minimum effective switching mass (default: at least ONE draw switches, i.e.
mass >= 1/W). Steps 3-5 (bandwidth-neighbor stability, optimized-value
directional agreement, bound/infeasibility rejection) are NOT implemented
here -- this is the threshold-crossing candidate only, documented as a
partial implementation, not the full adaptive rule.
"""
function adaptive_h_candidate(x_free0::AbstractVector, v_free::AbstractVector, ctx;
        min_switch_mass::Float64 = 1.0 / size(ctx.U, 1), h_max::Float64 = 0.2)
    θ0 = CS.reconstruct_full(x_free0, ctx.m)
    v_mat = v_free_to_Amat(v_free, ctx)
    thresh, winner0 = exact_tie_thresholds(θ0, v_mat, ctx)
    W, D = size(winner0)
    finite_abs = sort(abs.(thresh[isfinite.(thresh) .& (abs.(thresh) .> 1e-10)]))
    min_switches_needed = max(1, round(Int, min_switch_mass * W * D))
    idx = min(min_switches_needed, length(finite_abs))
    return idx == 0 ? h_max : min(finite_abs[idx], h_max)
end
