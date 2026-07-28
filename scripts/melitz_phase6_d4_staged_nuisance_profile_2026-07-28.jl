# 2026-07-28 step-control/robustness session, Phase 6: strengthen the D4 nuisance profile.
#
# At every fixed g on the SAME predetermined Phase 2 fraction grid (subset, dense enough over
# the finite region): (1) A-only nuisance minimization (f held fixed), (2) f-only (A held
# fixed), (3) full A/f seeded from the BETTER of the two restricted solutions at that SAME g.
# Reported incumbent at each g = min(fixed-A/f, A-only, f-only, full), so a failed/unconverged
# full solve can never overwrite a better restricted point (governing prompt's own explicit
# requirement) -- fixed-A/f is always a feasible point of every more-flexible search (setting
# the extra coordinates to their calibrated value), so the reported curve is monotone
# non-increasing in flexibility by construction, never merely by luck.
#
# Continuation: A-only and f-only each warm-start (theta AND inner dual, from a genuinely
# verified/finite point only) from the PRECEDING grid point's own same-restriction solution;
# the full stage warm-starts its theta from whichever of THIS g's own A-only/f-only result has
# the lower Delta (the "better restricted nuisance solution", per the governing prompt), and
# its dual from that same chosen point.
#
# Usage: julia --project=. scripts/melitz_phase6_d4_staged_nuisance_profile_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, LinearAlgebra
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
using KNITRO

const OUTDIR = joinpath(dirname(@__DIR__), "docs", "key_results")
mkpath(OUTDIR)
const REPO = dirname(@__DIR__)
const CAP = 10.0
const FRACTIONS = [0.00, 0.10, 0.20, 0.35, 0.50, 0.65]

kappa_of_g(g, wratio, sigma) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa, wratio, sigma) = log((kappa / wratio)^(sigma - 1))

block_free_mask(n, D, free_g, free_A, free_f) = (nA = D^2 - 1; m = falses(n); m[1] = free_g; m[2:1+nA] .= free_A; m[2+nA:end] .= free_f; m)

ok_status(n) = n in (0, -101, -102, -103)

function solve_stage(label, ctx, obj, theta_start, free_mask, inner_opt, inner_cfg, warm_x)
    local res
    wall = @elapsed begin
        res = solve_melitz_nuisance_min_delta(ctx, obj, theta_start; free_mask=free_mask,
            radius=0.3, gradient_backend=:B_direct_argument_parallel, h=1e-4,
            inner_loop_opt=inner_opt, warm_start_x=warm_x, inner_solve_config=inner_cfg,
            forbid_dense_fallback=true)
    end
    ok = ok_status(res.nStatus) && res.r_final.verified
    @printf("    [%-10s] nStatus=%4d wall=%5.2fs n_fc=%3d n_ga=%3d Delta=%s ok=%s\n",
        label, res.nStatus, wall, res.n_fc_calls, res.n_ga_calls,
        res.r_final.verified ? @sprintf("%.4e", res.r_final.Delta) : "UNVERIFIED", ok)
    flush(stdout)
    return res, ok, wall
end

function main()
    BLAS.set_num_threads(1)
    data = generate_fake_melitz_data(; D=4, sigma=2.5, theta_star=6.8, target_country=1, seed=29, W=20_000)
    obj, theta0 = build_melitz_psi_bundle(data; forbid_dense_fallback=true, backend=:matrix_free)
    ctx = obj.γ
    n = length(theta0)
    D = ctx.D
    r0 = evaluate_melitz_delta(theta0, ctx, obj; cold=true, store_G=false)
    @assert r0.verified

    j = ctx.target_country
    state0 = melitz_outer_state(theta0, ctx)
    lambda_jj = state0.equilibrium.trade_flow[j, j] / state0.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]
    sigma = ctx.sigma
    kappa_pareto = kappa_of_g(theta0[1], wratio, sigma)
    kappa_min = lambda_jj^(1 / (sigma - 1))

    inner_opt = joinpath(REPO, "melitz_inner_loop_options_capped_2026-07-24.opt")
    inner_cfg = MelitzInnerSolveConfig(:diagnostic; delta_evaluation_cap=CAP)
    mask_A = block_free_mask(n, D, false, true, false)
    mask_f = block_free_mask(n, D, false, false, true)
    mask_full = block_free_mask(n, D, false, true, true)

    rows = NamedTuple[]
    warm_x_A = nothing; warm_theta_A = copy(theta0)
    warm_x_f = nothing; warm_theta_f = copy(theta0)

    for frac in FRACTIONS
        kappa_f = kappa_pareto + frac * (kappa_min - kappa_pareto)
        g_f = frac == 0.0 ? theta0[1] : g_of_kappa(kappa_f, wratio, sigma)
        theta_fixed = copy(theta0); theta_fixed[1] = g_f
        r_fixed = evaluate_melitz_delta(theta_fixed, ctx, obj; cold=true, store_G=false)
        println("="^90); @printf("g=%.6f (frac=%.2f)  fixed-A/f Delta=%s\n", g_f, frac,
            r_fixed.verified ? @sprintf("%.4e", r_fixed.Delta) : "UNVERIFIED nStatus=$(r_fixed.nStatus)")

        theta_start_A = copy(warm_theta_A); theta_start_A[1] = g_f
        res_A, ok_A, wall_A = solve_stage("A-only", ctx, obj, theta_start_A, mask_A, inner_opt, inner_cfg, warm_x_A)
        if ok_A
            warm_x_A = res_A.r_final.dual_x
            warm_theta_A = copy(res_A.theta_final)
        end

        theta_start_f = copy(warm_theta_f); theta_start_f[1] = g_f
        res_f, ok_f, wall_f = solve_stage("f-only", ctx, obj, theta_start_f, mask_f, inner_opt, inner_cfg, warm_x_f)
        if ok_f
            warm_x_f = res_f.r_final.dual_x
            warm_theta_f = copy(res_f.theta_final)
        end

        # Full A/f seeded from the BETTER of the two restricted solutions at THIS g.
        candidates = NamedTuple[]
        ok_A && push!(candidates, (theta=res_A.theta_final, dual=res_A.r_final.dual_x, Delta=res_A.r_final.Delta))
        ok_f && push!(candidates, (theta=res_f.theta_final, dual=res_f.r_final.dual_x, Delta=res_f.r_final.Delta))
        local res_full, ok_full, wall_full
        if isempty(candidates)
            theta_start_full = copy(theta_fixed)
            res_full, ok_full, wall_full = solve_stage("full", ctx, obj, theta_start_full, mask_full, inner_opt, inner_cfg, nothing)
        else
            best = candidates[argmin([c.Delta for c in candidates])]
            res_full, ok_full, wall_full = solve_stage("full", ctx, obj, copy(best.theta), mask_full, inner_opt, inner_cfg, best.dual)
        end

        deltas = Dict{String,Float64}()
        r_fixed.verified && (deltas["fixed"] = r_fixed.Delta)
        ok_A && (deltas["A_only"] = res_A.r_final.Delta)
        ok_f && (deltas["f_only"] = res_f.r_final.Delta)
        ok_full && (deltas["full"] = res_full.r_final.Delta)
        best_label = isempty(deltas) ? "none" : first(sort(collect(keys(deltas)), by=k -> deltas[k]))
        best_delta = isempty(deltas) ? NaN : deltas[best_label]

        push!(rows, (fixture="D4_seed29_W20000", gamma_fraction=frac, g=g_f,
            Delta_fixed=get(deltas, "fixed", NaN),
            Delta_A_only=get(deltas, "A_only", NaN), Delta_A_only_ok=ok_A,
            Delta_f_only=get(deltas, "f_only", NaN), Delta_f_only_ok=ok_f,
            Delta_full=get(deltas, "full", NaN), Delta_full_ok=ok_full,
            Delta_reported=best_delta, best_source=best_label,
            wall_A=wall_A, wall_f=wall_f, wall_full=wall_full,
            n_fc_A=res_A.n_fc_calls, n_fc_f=res_f.n_fc_calls, n_fc_full=res_full.n_fc_calls))
        @printf("  BEST: %s -> Delta=%.4e  (fixed=%.4e)\n", best_label, best_delta, get(deltas, "fixed", NaN))
    end

    # Structural guarantee check: the reported curve must never exceed the fixed-A/f value.
    for r in rows
        if !isnan(r.Delta_fixed) && !isnan(r.Delta_reported)
            @assert r.Delta_reported <= r.Delta_fixed + 1e-9 "governing-prompt violation at frac=$(r.gamma_fraction): flexible reported ($(r.Delta_reported)) > fixed-A/f ($(r.Delta_fixed))"
        end
    end

    outfile = joinpath(OUTDIR, "melitz_phase6_d4_staged_nuisance_profile_2026-07-28.csv")
    open(outfile, "w") do io
        cols = keys(rows[1])
        println(io, join(cols, ","))
        for r in rows
            println(io, join([r[c] for c in cols], ","))
        end
    end
    println("\nDONE (structural guarantee verified: reported <= fixed-A/f at every grid point). CSV written to ", outfile)
    return rows
end

main()
