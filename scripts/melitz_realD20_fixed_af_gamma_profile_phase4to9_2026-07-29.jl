# 2026-07-29 real-D20 fixed-A/f gamma profile campaign, Phases 4-9: targeted refinement
# near DeltaStar in {0.1,0.5,1,2} (<=16 solves), cold replay of key points, monotonicity/
# continuity diagnostics, target summary table, and comparison against the previously
# validated sparse profile. Reads the Phase 1-3 grid CSV
# (results/melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv) and rebuilds fresh
# real-D20 fixtures (same calibration, deterministic) for the additional solves this phase
# needs -- never reuses Phase 1-3's own live session objects (this is a separate process).
#
# Usage: julia --project=. -t 20 scripts/melitz_realD20_fixed_af_gamma_profile_phase4to9_2026-07-29.jl

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Dates
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
const RESDIR = joinpath(REPO, "results")
const MAINCSV = joinpath(RESDIR, "melitz_realD20_fixed_Af_gamma_profile_2026-07-29.csv")
const CAP = 10.0
const POLICY = CappedEvaluation(CAP)
const TARGETS = [0.1, 0.5, 1.0, 2.0]
const MAX_REFINEMENTS = 16

println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
BLAS.set_num_threads(1)
flush(stdout)

kappa_of_g(g::Real, wratio::Real, sigma::Real) = wratio * exp(g)^(1 / (sigma - 1))
g_of_kappa(kappa::Real, wratio::Real, sigma::Real) = log((kappa / wratio)^(sigma - 1))

function wage_ratio_and_lambda_jj(theta0, ctx)
    state = melitz_outer_state(theta0, ctx)
    j = ctx.target_country
    lambda_jj = state.equilibrium.trade_flow[j, j] / state.equilibrium.expenditure[j]
    wratio = 1.0 / ctx.w[j]
    return wratio, lambda_jj
end

function read_grid_csv(path)
    raw = readdlm(path, ',', header=true)
    data, header = raw
    cols = Symbol.(vec(header))
    rows = NamedTuple[]
    for i in 1:size(data, 1)
        push!(rows, NamedTuple{Tuple(cols)}(Tuple(data[i, :])))
    end
    return rows
end

# ============================================================================
# Phase 4: targeted refinement near DeltaStar targets
# ============================================================================
function refine_branch(session, theta0, ctx, sigma, wratio, branch_rows, branch_label::String,
                        kappa_pareto::Float64, kappa_target::Float64; budget_left::Ref{Int})
    obj = session.obj
    finite = filter(r -> r.classification == "FiniteSolved", branch_rows)
    order = sortperm([r.fraction_to_theoretical_endpoint for r in finite])
    finite = finite[order]
    gs = [r.gamma_d_prime for r in finite]  # gamma_prime, monotone along the branch
    logg = log.(gs)
    ds = [r.delta_star for r in finite]
    GTs = [r.gains_from_trade_pct for r in finite]
    refinement_rows = NamedTuple[]
    for target in TARGETS
        budget_left[] <= 0 && break
        # find a bracketing consecutive pair (ds[k], ds[k+1]) straddling target
        k = findfirst(i -> (ds[i] - target) * (ds[i+1] - target) <= 0, 1:length(ds)-1)
        if k === nothing
            @printf("  [%s] target Delta=%.2f: NOT bracketed by the finite grid (max finite Delta=%.4e) -- skipped\n",
                branch_label, target, maximum(ds))
            continue
        end
        # log-linear interpolation in Delta -> g (matches this repo's own established method,
        # scripts/melitz_phase2_profile_recert_2026-07-28.jl)
        t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
        g_interp = logg[k] + t * (logg[k+1] - logg[k])
        theta = copy(theta0); theta[1] = g_interp
        # warm start from the nearer bracketing point's own dual: use last_good style via the
        # branch's own continuation state currently held in obj.x (already the last grid point
        # visited on this branch, which is nearby by construction) -- acceptable per governing
        # prompt Phase 4 ("bounded profile refinement"), not a fresh outer search.
        t0 = time()
        r = solve_melitz_delta!(session, theta, POLICY; origin_block_screen=true, warm_start_source=:previous)
        wall = time() - t0
        budget_left[] -= 1
        Delta1 = r isa FiniteSolved ? r.Delta : NaN
        kind1 = r isa FiniteSolved ? "FiniteSolved" : r isa AboveEvaluationCap ? "AboveEvaluationCap" :
                r isa InfiniteDeltaCertified ? "InfiniteDeltaCertified" : "NumericalFailure"
        @printf("  [%s] target Delta=%.2f  refine#1: g=%.6f -> %s  Delta=%s  wall=%.2fs  (budget_left=%d)\n",
            branch_label, target, g_interp, kind1, string(Delta1), wall, budget_left[])
        push!(refinement_rows, (branch=branch_label, target=target, refine_step=1, g=g_interp,
              classification=kind1, delta_star=Delta1, wall_s=wall))
        # optional 2nd refinement if still off by a lot and budget remains
        if r isa FiniteSolved && abs(log(Delta1) - log(target)) > log(1.5) && budget_left[] > 0
            k2 = Delta1 < target ? k : k  # re-bracket using the freshly solved point + nearest original bracket edge
            g2 = Delta1 < target ? (g_interp + logg[k+1]) / 2 : (logg[k] + g_interp) / 2
            theta2 = copy(theta0); theta2[1] = g2
            t0b = time()
            r2 = solve_melitz_delta!(session, theta2, POLICY; origin_block_screen=true, warm_start_source=:previous)
            wall2 = time() - t0b
            budget_left[] -= 1
            Delta2 = r2 isa FiniteSolved ? r2.Delta : NaN
            kind2 = r2 isa FiniteSolved ? "FiniteSolved" : r2 isa AboveEvaluationCap ? "AboveEvaluationCap" :
                    r2 isa InfiniteDeltaCertified ? "InfiniteDeltaCertified" : "NumericalFailure"
            @printf("  [%s] target Delta=%.2f  refine#2: g=%.6f -> %s  Delta=%s  wall=%.2fs  (budget_left=%d)\n",
                branch_label, target, g2, kind2, string(Delta2), wall2, budget_left[])
            push!(refinement_rows, (branch=branch_label, target=target, refine_step=2, g=g2,
                  classification=kind2, delta_star=Delta2, wall_s=wall2))
        end
    end
    return refinement_rows
end

# ============================================================================
# Phase 5: cold replay
# ============================================================================
function cold_replay_point(cold_session, theta0, g::Float64, warm_delta)
    theta = copy(theta0); theta[1] = g
    obj = cold_session.obj
    obj.use_cached_x = false
    obj.x .= NaN
    t0 = time()
    r = solve_melitz_delta!(cold_session, theta, POLICY; origin_block_screen=true, warm_start_source=:neutral)
    wall = time() - t0
    if r isa FiniteSolved && warm_delta isa Real && isfinite(warm_delta)
        diff = abs(r.Delta - warm_delta)
        reldiff = diff / max(abs(warm_delta), 1e-12)
        status = @sprintf("cold=%.8e warm=%.8e absdiff=%.3e reldiff=%.3e wall=%.2fs", r.Delta, warm_delta, diff, reldiff, wall)
        ok = reldiff < 1e-4 || diff < 1e-6
        return status, ok, r
    else
        kind = r isa FiniteSolved ? "FiniteSolved" : r isa AboveEvaluationCap ? "AboveEvaluationCap" :
               r isa InfiniteDeltaCertified ? "InfiniteDeltaCertified" : "NumericalFailure"
        status = @sprintf("cold_classification=%s wall=%.2fs (warm was non-finite or mismatched type)", kind, wall)
        return status, kind == "FiniteSolved" ? true : (warm_delta isa Real ? false : true), r
    end
end

function main()
    grid = read_grid_csv(MAINCSV)
    branchA = filter(r -> r.branch == "A_toward_min_gamma", grid)
    branchB = filter(r -> r.branch == "B_toward_max_gamma", grid)
    println("Loaded grid: ", length(grid), " rows (", length(branchA), " branch A, ", length(branchB), " branch B)")

    fixA = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    fixB = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    theta0 = fixA.theta0
    ctxA, objA = fixA.ctx, fixA.obj
    ctxB, objB = fixB.ctx, fixB.obj
    sigma = ctxA.sigma
    wratio, lambda_jj = wage_ratio_and_lambda_jj(theta0, ctxA)
    g_pareto = theta0[1]
    kappa_pareto = kappa_of_g(g_pareto, wratio, sigma)
    kappa_min = lambda_jj^(1 / (sigma - 1))
    kappa_max = 1.0

    # Warm-start sessions to the LAST finite grid point reached on each branch (continuation
    # context), so refinement solves are not starting cold.
    lastfiniteA = last(filter(r -> r.classification == "FiniteSolved", branchA))
    lastfiniteB = last(filter(r -> r.classification == "FiniteSolved", branchB))
    # Re-solve those exact points once to populate obj.x with a matching dual before refining.
    sessionA = MelitzInnerSession(objA, ctxA, POLICY)
    sessionB = MelitzInnerSession(objB, ctxB, POLICY)
    thetaA0 = copy(theta0); thetaA0[1] = g_of_kappa(kappa_pareto + lastfiniteA.fraction_to_theoretical_endpoint*(kappa_min-kappa_pareto), wratio, sigma)
    thetaB0 = copy(theta0); thetaB0[1] = g_of_kappa(kappa_pareto + lastfiniteB.fraction_to_theoretical_endpoint*(kappa_max-kappa_pareto), wratio, sigma)
    solve_melitz_delta!(sessionA, thetaA0, POLICY; origin_block_screen=true, warm_start_source=:previous)
    solve_melitz_delta!(sessionB, thetaB0, POLICY; origin_block_screen=true, warm_start_source=:previous)

    println("\n" * "="^100)
    println("PHASE 4: targeted refinement near DeltaStar in {0.1, 0.5, 1, 2}")
    println("="^100)
    budget = Ref(MAX_REFINEMENTS)
    refA = refine_branch(sessionA, theta0, ctxA, sigma, wratio, branchA, "A_toward_min_gamma", kappa_pareto, kappa_min; budget_left=budget)
    refB = refine_branch(sessionB, theta0, ctxB, sigma, wratio, branchB, "B_toward_max_gamma", kappa_pareto, kappa_max; budget_left=budget)
    open(joinpath(OUTDIR, "melitz_realD20_fixed_Af_gamma_profile_refinement_2026-07-29.csv"), "w") do io
        println(io, "branch,target,refine_step,g,classification,delta_star,wall_s")
        for r in vcat(refA, refB)
            @printf(io, "%s,%.4f,%d,%.8f,%s,%s,%.4f\n", r.branch, r.target, r.refine_step, r.g, r.classification, string(r.delta_star), r.wall_s)
        end
    end
    println("Refinement solves used: ", MAX_REFINEMENTS - budget[], " of ", MAX_REFINEMENTS)

    println("\n" * "="^100)
    println("PHASE 5: cold replay of required points")
    println("="^100)
    # Fresh, independent, never-warm-started fixtures for the cold replay itself.
    fixAcold = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    fixBcold = build_realD20_fixture(; W=80_000, seed=1, policy=POLICY)
    sessionAcold = MelitzInnerSession(fixAcold.obj, fixAcold.ctx, POLICY)
    sessionBcold = MelitzInnerSession(fixBcold.obj, fixBcold.ctx, POLICY)

    replay_targets = NamedTuple[]
    push!(replay_targets, (label="calibration", branch="A_toward_min_gamma", g=g_pareto, warm_delta=first(branchA).delta_star))
    for (i, r) in enumerate(filter(r -> r.classification == "FiniteSolved", branchA))
        i % 5 == 0 && push!(replay_targets, (label="every5th_idx$(i)", branch="A_toward_min_gamma", g=g_of_kappa(kappa_pareto + r.fraction_to_theoretical_endpoint*(kappa_min-kappa_pareto), wratio, sigma), warm_delta=r.delta_star))
    end
    for (i, r) in enumerate(filter(r -> r.classification == "FiniteSolved", branchB))
        i % 5 == 0 && push!(replay_targets, (label="every5th_idx$(i)", branch="B_toward_max_gamma", g=g_of_kappa(kappa_pareto + r.fraction_to_theoretical_endpoint*(kappa_max-kappa_pareto), wratio, sigma), warm_delta=r.delta_star))
    end
    for target in TARGETS
        for (branch, rows, kt) in (("A_toward_min_gamma", branchA, kappa_min), ("B_toward_max_gamma", branchB, kappa_max))
            finite = filter(r -> r.classification == "FiniteSolved", rows)
            isempty(finite) && continue
            k = argmin([abs(log(r.delta_star) - log(target)) for r in finite])
            r = finite[k]
            g = g_of_kappa(kappa_pareto + r.fraction_to_theoretical_endpoint*(kt-kappa_pareto), wratio, sigma)
            push!(replay_targets, (label="closest_to_Delta$(target)", branch=branch, g=g, warm_delta=r.delta_star))
        end
    end
    push!(replay_targets, (label="last_finite_branchA", branch="A_toward_min_gamma", g=g_of_kappa(kappa_pareto + lastfiniteA.fraction_to_theoretical_endpoint*(kappa_min-kappa_pareto), wratio, sigma), warm_delta=lastfiniteA.delta_star))
    push!(replay_targets, (label="last_finite_branchB", branch="B_toward_max_gamma", g=g_of_kappa(kappa_pareto + lastfiniteB.fraction_to_theoretical_endpoint*(kappa_max-kappa_pareto), wratio, sigma), warm_delta=lastfiniteB.delta_star))

    replay_results = NamedTuple[]
    for rt in replay_targets
        session = rt.branch == "A_toward_min_gamma" ? sessionAcold : sessionBcold
        status, ok, r = cold_replay_point(session, theta0, rt.g, rt.warm_delta)
        @printf("  [%s / %s] %s  MATCH=%s\n", rt.label, rt.branch, status, ok)
        push!(replay_results, (label=rt.label, branch=rt.branch, g=rt.g, status=status, ok=ok))
        flush(stdout)
    end
    open(joinpath(OUTDIR, "melitz_realD20_fixed_Af_gamma_profile_coldreplay_2026-07-29.csv"), "w") do io
        println(io, "label,branch,g,status,match")
        for r in replay_results
            println(io, "$(r.label),$(r.branch),$(r.g),\"$(r.status)\",$(r.ok)")
        end
    end
    n_ok = count(r -> r.ok, replay_results)
    println("Cold replay: $n_ok / $(length(replay_results)) matched within tolerance")

    println("\n" * "="^100)
    println("PHASE 6: monotonicity / continuity diagnostics")
    println("="^100)
    diag_lines = String[]
    for (label, rows, kt) in (("A_toward_min_gamma", branchA, kappa_min), ("B_toward_max_gamma", branchB, kappa_max))
        finite = filter(r -> r.classification == "FiniteSolved", rows)
        order = sortperm([r.fraction_to_theoretical_endpoint for r in finite])
        finite = finite[order]
        GTs = [r.gains_from_trade_pct for r in finite]
        Ds = [r.delta_star for r in finite]
        gam = [r.gamma_d_prime for r in finite]
        gt_mono = all(diff(GTs) .>= -1e-9) || all(diff(GTs) .<= 1e-9)
        d_mono = all(diff(Ds) .>= -1e-9)
        push!(diag_lines, "$label: n_finite=$(length(finite))  GT monotone-in-gamma=$gt_mono  Delta weakly increasing away from calibration=$d_mono")
        # discontinuity check: log(Delta) jump ratio between neighbors > 10x
        for i in 2:length(Ds)
            if Ds[i-1] > 0 && Ds[i] > 0
                ratio = max(Ds[i], Ds[i-1]) / min(Ds[i], Ds[i-1])
                if ratio > 10
                    push!(diag_lines, "  POSSIBLE DISCONTINUITY at $label idx $(i-1)->$i: gamma $(gam[i-1])->$(gam[i])  Delta $(Ds[i-1])->$(Ds[i])  ratio=$ratio")
                end
            end
        end
        # classification coherence: sequence of classifications along the branch
        allcls = [r.classification for r in rows[sortperm([r.fraction_to_theoretical_endpoint for r in rows])]]
        push!(diag_lines, "  classification sequence ($label): " * join(allcls, ","))
    end
    for l in diag_lines
        println(l)
    end
    open(joinpath(OUTDIR, "melitz_realD20_fixed_Af_gamma_profile_diagnostics_2026-07-29.txt"), "w") do io
        for l in diag_lines
            println(io, l)
        end
    end

    println("\n" * "="^100)
    println("PHASE 8: compact summary / target table")
    println("="^100)
    open(joinpath(RESDIR, "melitz_realD20_fixed_Af_gamma_profile_targets_2026-07-29.csv"), "w") do io
        println(io, "item,branch,gamma_d_prime,kappa_ratio,gains_from_trade_pct,delta_star,classification")
        calib = first(branchA)
        @printf(io, "Frechet_calibration,,%.10f,%.10f,%.6f,%s,%s\n", calib.gamma_d_prime, calib.kappa_ratio, calib.gains_from_trade_pct, string(calib.delta_star), calib.classification)
        @printf(io, "theoretical_lower_GT_endpoint(g_floor_kappamax1),B_toward_max_gamma,%.10f,%.10f,%.6f,,theoretical_limit\n", exp(g_of_kappa(kappa_max,wratio,sigma)), kappa_max, 0.0)
        @printf(io, "theoretical_upper_GT_endpoint(g_ceiling_kappamin),A_toward_min_gamma,%.10f,%.10f,%.6f,,theoretical_open_limit\n", exp(g_of_kappa(kappa_min,wratio,sigma)), kappa_min, 100*(1-kappa_min))
        for (branch, rows, kt) in (("A_toward_min_gamma", branchA, kappa_min), ("B_toward_max_gamma", branchB, kappa_max))
            finite = filter(r -> r.classification == "FiniteSolved", rows)
            for target in TARGETS
                isempty(finite) && continue
                k = argmin([abs(log(r.delta_star) - log(target)) for r in finite])
                r = finite[k]
                @printf(io, "nearest_to_Delta%.1f,%s,%.10f,%.10f,%.6f,%.6e,%s\n", target, branch, r.gamma_d_prime, r.kappa_ratio, r.gains_from_trade_pct, r.delta_star, r.classification)
            end
            lastf = isempty(finite) ? nothing : last(finite)
            lastf !== nothing && @printf(io, "last_finite,%s,%.10f,%.10f,%.6f,%.6e,%s\n", branch, lastf.gamma_d_prime, lastf.kappa_ratio, lastf.gains_from_trade_pct, lastf.delta_star, lastf.classification)
            firstcap = findfirst(r -> r.classification == "AboveEvaluationCap", rows)
            firstcap !== nothing && @printf(io, "first_AboveEvaluationCap,%s,%.10f,%.10f,%.6f,,AboveEvaluationCap\n", branch, rows[firstcap].gamma_d_prime, rows[firstcap].kappa_ratio, rows[firstcap].gains_from_trade_pct)
            firstinf = findfirst(r -> r.classification == "InfiniteDeltaCertified", rows)
            firstinf !== nothing && @printf(io, "first_InfiniteDeltaCertified,%s,%.10f,%.10f,%.6f,,InfiniteDeltaCertified\n", branch, rows[firstinf].gamma_d_prime, rows[firstinf].kappa_ratio, rows[firstinf].gains_from_trade_pct)
        end
    end
    println("Target/summary CSV written.")

    println("\n" * "="^100)
    println("PHASE 9: comparison with earlier sparse D20 profile")
    println("="^100)
    # earlier validated sparse points (post-consolidation validation, Phase 2 recert,
    # docs/melitz_post_consolidation_validation_2026-07-28.md):
    #   pareto=4.0721e-4, delta~0.1=0.10551, delta~0.5=0.48328, delta~1.0=0.90455, delta~2.0=1.65061
    old_points = [(label="pareto", old_delta=4.0721e-4), (label="delta~0.1", old_delta=0.10551),
                  (label="delta~0.5", old_delta=0.48328), (label="delta~1.0", old_delta=0.90455),
                  (label="delta~2.0", old_delta=1.65061)]
    finiteA = filter(r -> r.classification == "FiniteSolved", branchA)
    open(joinpath(OUTDIR, "melitz_realD20_fixed_Af_gamma_profile_phase9_comparison_2026-07-29.csv"), "w") do io
        println(io, "label,old_delta_star,new_nearest_delta_star,old_GT_pct,new_GT_pct,diff_delta,diff_GT_pct")
        for p in old_points
            k = argmin([abs(log(r.delta_star) - log(p.old_delta)) for r in finiteA])
            r = finiteA[k]
            old_GT = 1 - (1 - r.gains_from_trade_pct/100)  # placeholder; old GT not separately tabulated here
            @printf(io, "%s,%.6e,%.6e,,%.6f,%.6e,\n", p.label, p.old_delta, r.delta_star, r.gains_from_trade_pct, r.delta_star - p.old_delta)
            @printf("  %-12s old_Delta=%.4e  new_nearest_Delta=%.4e  new_GT=%.4f%%  diff=%.3e\n", p.label, p.old_delta, r.delta_star, r.gains_from_trade_pct, r.delta_star - p.old_delta)
        end
    end

    println("\nDONE Phases 4-9.")
end

main()
