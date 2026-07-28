# 2026-07-28 outer-search gradient/redundancy/sensitivity continuation, Phases 6-8.
#
# Phase 6: does a tiny epsilon step actually improve the objective / move DeltaStar as
# predicted? Tested at D4 calibration + D4 near-boundary + real-D20 calibration + real-D20
# interior + real-D20 near-boundary points, for three direction types: pure-g (= the full
# reported objective direction here, since the objective is exactly linear in theta[1] with
# constant gradient +-e1 -- confirmed structurally, not re-derived, from
# finite_delta_outer.jl's own Section 3.1), negative-DeltaStar-gradient, and the Phase 5
# projected tangent direction (d = -c + q*dot(q,c)/dot(q,q)).
#
# Phase 7: A/f redundancy -- singular values of the local moment-response Jacobian in raw
# log-A/log-f coordinates, D4 (an explicit W*num_moments x n matrix, built column-by-column
# via the ALREADY-validated Method D directional-derivative formula
# (melitz_moment_directional_derivative, finite_delta_outer.jl) -- no new derivative code).
#
# Phase 8: intensive vs extensive decomposition of a moment-space step. Intensive
# (coefficient-change-holding-participation-fixed) approximated via the SAME Method D linear
# directional derivative (exact to first order, by construction, since it IS the
# fixed-active-set derivative); extensive isolated two ways -- the residual between the true
# nonlinear moment change and the linear (intensive) prediction, AND the direct
# participation-flip count (count_switches, gradient_lab.jl, already-validated).
#
# Usage: julia --project=. -t 20 scripts/melitz_phase6_7_8_gradient_redundancy_sensitivity_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const OUTDIR = joinpath(REPO, "docs", "key_results")
mkpath(OUTDIR)
const CAP = 10.0
const EPSILONS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

function write_csv(path, rows)
    isempty(rows) && return
    open(path, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
end

# ============================================================================
# Phase 6: epsilon-step diagnostics.
# ============================================================================

"Compute grad Delta (raw, d(1e10*Delta)/dtheta) at (theta,x) via the sorted parallel direct backend (falls back serial at small D)."
function grad_delta_at(theta, ctx, obj, x; h=1e-4)
    n = length(theta)
    g = zeros(n)
    gfn = ctx.D >= 10 && Threads.nthreads() > 1 ? make_melitz_gradient_delta_direct_sorted_parallel(h) :
                                                    make_melitz_gradient_delta_direct_sorted_serial(h)
    gfn(g, theta, ctx, obj, x)
    return g
end

function phase6_point(label, ctx, obj, theta_pt; bank=MelitzDualBank(8))
    n = length(theta_pt)
    r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
    if !(r0 isa FiniteSolved)
        println("  SKIP $label: base point is not FiniteSolved (", typeof(r0), ")")
        return NamedTuple[]
    end
    Delta0 = r0.Delta
    qraw = grad_delta_at(theta_pt, ctx, obj, r0.x)   # d(1e10*Delta)/dtheta
    q = qraw ./ 1e10                                  # d(Delta)/dtheta
    qnorm = norm(q)
    rows = NamedTuple[]
    for direction_sym in (:upper, :lower)
        sgn = direction_sym == :upper ? 1.0 : -1.0     # signed_objective(theta) = sgn*theta[1]
        c = zeros(n); c[1] = sgn                       # grad(signed_objective) -- exact, constant

        pure_g = zeros(n); pure_g[1] = -sgn            # improving direction for the objective: -grad(objective)
        neg_delta_grad = qnorm > 0 ? (-q ./ qnorm) : zeros(n)
        # Phase 5 projected direction: d = -c + q*(dot(q,c)/dot(q,q)) (minimization convention);
        # normalize for a fair epsilon comparison across direction types.
        proj = qnorm > 0 ? (-c .+ q .* (dot(q, c) / dot(q, q))) : (-c)
        proj_norm = norm(proj) > 0 ? proj ./ norm(proj) : proj

        for (dirname, dvec) in (("pure_g", pure_g), ("neg_DeltaStar_grad", neg_delta_grad), ("projected_tangent", proj_norm))
            norm(dvec) == 0 && continue
            for eps in EPSILONS
                theta_new = theta_pt .+ eps .* dvec
                r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank)
                obj_old = sgn * theta_pt[1]
                obj_new = sgn * theta_new[1]
                predicted_obj_change = eps * dot(dvec, c)
                realized_obj_change = obj_new - obj_old
                predicted_delta_change = eps * dot(dvec, q)
                finite_new = r_new isa FiniteSolved
                realized_delta_change = finite_new ? (r_new.Delta - Delta0) : NaN
                push!(rows, (point=label, direction=string(direction_sym), step_type=dirname, epsilon=eps,
                    obj_improves=(realized_obj_change < 0), predicted_obj_change=predicted_obj_change,
                    realized_obj_change=realized_obj_change,
                    delta_finite_after_step=finite_new, predicted_delta_change=predicted_delta_change,
                    realized_delta_change=realized_delta_change, new_status=string(typeof(r_new)), Delta0=Delta0))
            end
        end
    end
    return rows
end

# ============================================================================
# Phase 7: A/f redundancy Jacobian (D4).
# ============================================================================

function phase7_redundancy(ctx, obj, theta_pt)
    n = length(theta_pt)
    D = ctx.D
    W = size(obj.U, 1)
    num_moments = ctx.moment_layout.num_moments
    println("  building full moment-response Jacobian: W*num_moments=", W * num_moments, " x n=", n)
    J = zeros(W * num_moments, n)
    ei = zeros(n)
    for k in 1:n
        ei[k] = 1.0
        dG = melitz_moment_directional_derivative(theta_pt, ei, ctx, obj)   # W x num_moments
        J[:, k] .= vec(dG)
        ei[k] = 0.0
    end
    sv = svdvals(J)
    cond_num = sv[1] / sv[end]
    n_near_null = count(s -> s < 1e-6 * sv[1], sv)
    return (n=n, W=W, num_moments=num_moments, sv_max=sv[1], sv_min=sv[end], cond_number=cond_num,
            n_near_null_directions=n_near_null, singular_values=sv)
end

# ============================================================================
# Phase 8: intensive/extensive decomposition.
# ============================================================================

function phase8_decomposition(ctx, obj, theta_pt, label; step_norm=1e-3, seed_rng=1234)
    n = length(theta_pt)
    D = ctx.D
    nA = D^2 - 1
    snap = base_active_mask(theta_pt, ctx, obj)

    directions = Dict{String,Vector{Float64}}()
    g_dir = zeros(n); g_dir[1] = 1.0
    directions["pure_g"] = g_dir
    A_dir = zeros(n); A_dir[2:1+nA] .= randn(MersenneTwister(seed_rng), nA); directions["A_only"] = A_dir ./ norm(A_dir)
    f_dir = zeros(n); f_dir[2+nA:end] .= randn(MersenneTwister(seed_rng + 1), n - 1 - nA); directions["f_only"] = f_dir ./ norm(f_dir)
    mixed = zeros(n); mixed[2:end] .= randn(MersenneTwister(seed_rng + 2), n - 1); directions["mixed_Af"] = mixed ./ norm(mixed)

    rows = NamedTuple[]
    for (dname, dvec) in directions
        step = step_norm .* dvec
        theta_new = theta_pt .+ step
        G_old = fixed_active_set_moments(theta_pt, ctx, obj)
        G_new_true = zeros(size(G_old))
        profit_scratch = zeros(size(G_old, 1))
        fixed_active_set_moments!(G_new_true, profit_scratch, theta_new, ctx, obj)
        true_change = G_new_true .- G_old
        total_norm = norm(true_change)

        dG_linear = melitz_moment_directional_derivative(theta_pt, dvec, ctx, obj) .* step_norm
        intensive_norm = norm(dG_linear)
        residual = true_change .- dG_linear
        extensive_residual_norm = norm(residual)

        n_switches, per_cell, n_autarky = count_switches(snap, theta_new, ctx, obj)

        push!(rows, (point=label, direction=dname, step_norm=step_norm, total_moment_change_norm=total_norm,
            intensive_linear_norm=intensive_norm, extensive_residual_norm=extensive_residual_norm,
            frac_extensive=(total_norm > 0 ? extensive_residual_norm / total_norm : NaN),
            n_switches=n_switches, n_autarky_switches=n_autarky))
    end
    return rows
end

function main()
    BLAS.set_num_threads(1)
    println("="^100); println("D4 fixture"); println("="^100)
    d4 = build_d4_fixture()
    ctx4, obj4, theta0_4 = d4.ctx, d4.obj, d4.theta0

    profile4_path = joinpath(OUTDIR, "melitz_phase2_gamma_profile_d4_2026-07-28.csv")
    theta_near_boundary_4 = copy(theta0_4)
    if isfile(profile4_path)
        prof4 = load_gamma_profile_csv(profile4_path)
        finite4 = filter(r -> r.classification == "FiniteSolved", prof4)
        row = finite4[argmax([abs(r.g - theta0_4[1]) for r in finite4])]
        theta_near_boundary_4[1] = row.g
        println("  D4 near-boundary point from existing profile CSV: g=", row.g, " DeltaStar=", row.DeltaStar)
    else
        theta_near_boundary_4[1] -= 0.03
        println("  D4 profile CSV not found -- using a direct -0.03 g perturbation as the near-boundary point")
    end

    println("\n" * "="^100); println("PHASE 6 (D4)"); println("="^100)
    rows6 = NamedTuple[]
    append!(rows6, phase6_point("D4_calibration", ctx4, obj4, theta0_4))
    append!(rows6, phase6_point("D4_near_boundary", ctx4, obj4, theta_near_boundary_4))
    write_csv(joinpath(OUTDIR, "melitz_phase6_epsilon_step_d4_2026-07-28.csv"), rows6)
    println("  D4 epsilon-step rows: ", length(rows6))
    flush(stdout)

    println("\n" * "="^100); println("PHASE 7 (D4)"); println("="^100)
    p7 = phase7_redundancy(ctx4, obj4, theta0_4)
    @printf("  n=%d W=%d num_moments=%d sv_max=%.4e sv_min=%.4e cond=%.4e n_near_null=%d\n",
        p7.n, p7.W, p7.num_moments, p7.sv_max, p7.sv_min, p7.cond_number, p7.n_near_null_directions)
    write_csv(joinpath(OUTDIR, "melitz_phase7_af_redundancy_d4_2026-07-28.csv"),
        [(index=i, singular_value=s) for (i, s) in enumerate(p7.singular_values)])
    write_csv(joinpath(OUTDIR, "melitz_phase7_af_redundancy_d4_summary_2026-07-28.csv"),
        [(n=p7.n, W=p7.W, num_moments=p7.num_moments, sv_max=p7.sv_max, sv_min=p7.sv_min,
          cond_number=p7.cond_number, n_near_null_directions=p7.n_near_null_directions)])
    flush(stdout)

    println("\n" * "="^100); println("PHASE 8 (D4)"); println("="^100)
    rows8 = NamedTuple[]
    append!(rows8, phase8_decomposition(ctx4, obj4, theta0_4, "D4_calibration"; step_norm=1e-3))
    append!(rows8, phase8_decomposition(ctx4, obj4, theta_near_boundary_4, "D4_near_boundary"; step_norm=1e-3))
    append!(rows8, phase8_decomposition(ctx4, obj4, theta0_4, "D4_calibration_bigger_step"; step_norm=1e-2))
    write_csv(joinpath(OUTDIR, "melitz_phase8_intensive_extensive_d4_2026-07-28.csv"), rows8)
    for r in rows8
        @printf("  [%s/%s] step=%.0e total=%.4e intensive=%.4e extensive_resid=%.4e frac_ext=%.3f n_switches=%d\n",
            r.point, r.direction, r.step_norm, r.total_moment_change_norm, r.intensive_linear_norm,
            r.extensive_residual_norm, r.frac_extensive, r.n_switches)
    end
    flush(stdout)

    println("\n" * "="^100); println("PHASE 6 (real D20, lighter pass)"); println("="^100)
    d20 = build_realD20_fixture()
    ctx20, obj20, theta0_20 = d20.ctx, d20.obj, d20.theta0
    profile20 = load_gamma_profile_csv(joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"))
    finite20 = filter(r -> r.classification == "FiniteSolved", profile20)
    gs = [r.g for r in finite20]; ds = [r.DeltaStar for r in finite20]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    k = findfirst(i -> ds[i] <= 0.5 <= ds[i+1] || ds[i] >= 0.5 >= ds[i+1], 1:length(ds)-1)
    t = (log(0.5) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g_interior = gs[k] + t * (gs[k+1] - gs[k])
    theta_interior_20 = copy(theta0_20); theta_interior_20[1] = g_interior
    r_boundary = filter(r -> r.gamma_fraction == 0.65, profile20)
    theta_boundary_20 = copy(theta0_20)
    isempty(r_boundary) || (theta_boundary_20[1] = r_boundary[1].g)

    rows6_20 = NamedTuple[]
    append!(rows6_20, phase6_point("D20_calibration", ctx20, obj20, theta0_20))
    append!(rows6_20, phase6_point("D20_interior", ctx20, obj20, theta_interior_20))
    isempty(r_boundary) || append!(rows6_20, phase6_point("D20_near_boundary", ctx20, obj20, theta_boundary_20))
    write_csv(joinpath(OUTDIR, "melitz_phase6_epsilon_step_d20_2026-07-28.csv"), rows6_20)
    println("  D20 epsilon-step rows: ", length(rows6_20))

    println("\nDONE. CSVs written to ", OUTDIR)
end

main()
