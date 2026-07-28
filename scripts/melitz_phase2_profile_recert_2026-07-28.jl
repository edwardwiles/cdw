# Post-consolidation validation, Phase 2: re-certify the fixed-A/f gamma profiles (D4 + real
# D20) at economically relevant delta targets {0.1, 0.5, 1, 2} using the NEW consolidated
# inner-solve API. No root-finding: for each target, interpolate g from the EXISTING saved
# profile CSVs (docs/key_results/melitz_phase2_gamma_profile_{d4,realD20}_2026-07-28.csv,
# log-linear in Delta, same method `scripts/melitz_anomaly_phase1_reconstruct_2026-07-28.jl`
# used), then DIRECTLY solve at that interpolated g via `solve_melitz_delta!` and compare the
# freshly-verified DeltaStar to the profile's own value at that fraction (or the log-linear
# interpolation, if the exact fraction is not a grid point).

using Pkg
REPO = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
Pkg.activate(REPO)
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
include(joinpath(REPO, "scripts", "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const OUTDIR = joinpath(REPO, "docs", "key_results")
println("Julia threads: ", Threads.nthreads(), "   BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

CAP = 10.0
policy = CappedEvaluation(CAP)

function interp_g_at_target(profile, target)
    finite = filter(r -> r.classification == "FiniteSolved", profile)
    gs = [r.g for r in finite]; ds = [r.DeltaStar for r in finite]
    order = sortperm(gs); gs, ds = gs[order], ds[order]
    if target > maximum(ds)
        return nothing, maximum(ds)
    end
    k = findfirst(i -> ds[i] <= target <= ds[i+1] || ds[i] >= target >= ds[i+1], 1:length(ds)-1)
    k === nothing && return nothing, maximum(ds)
    t = (log(target) - log(ds[k])) / (log(ds[k+1]) - log(ds[k]))
    g = gs[k] + t * (gs[k+1] - gs[k])
    return g, nothing
end

function run_fixture_profile(name, profile_csv, build_fn; targets=[0.0, 0.1, 0.5, 1.0, 2.0])
    println("\n" * "="^100)
    println("FIXTURE: $name")
    println("="^100)
    profile = load_gamma_profile_csv(profile_csv)
    d = build_fn(; policy=policy)
    ctx, obj, theta0 = d.ctx, d.obj, d.theta0
    session = MelitzInnerSession(obj, ctx, policy)
    rows = NamedTuple[]
    for target in targets
        if target == 0.0
            g = theta0[1]
            label = "pareto"
        else
            g, maxd = interp_g_at_target(profile, target)
            if g === nothing
                println("\n-- target=$target: NOT REACHABLE in finite fixed-A/f corridor at this fixture " *
                        "(max profile FiniteSolved Delta = $maxd) -- SKIPPED, not forced via a higher cap " *
                        "(the original profile itself hit NumericalFailure/cap territory beyond this point, " *
                        "not a valid finite interpolation target).")
                continue
            end
            label = "delta~$target"
        end
        theta = copy(theta0); theta[1] = g
        t0 = time()
        r = solve_melitz_delta!(session, theta, policy; origin_block_screen=false,
                                 warm_start_source=:previous)
        wall = time() - t0
        Delta = r isa FiniteSolved ? r.Delta : NaN
        println("\n-- $label: g=$g  ->  result=$(typeof(r))  Delta=$Delta  wall=$(round(wall,digits=3))s")
        if r isa FiniteSolved
            println("   nStatus=$(r.nStatus)  ||x||=$(norm(r.x))")
        end
        push!(rows, (fixture=name, label=label, g=g, result_type=string(typeof(r)),
                      Delta=Delta, nStatus=(r isa FiniteSolved ? r.nStatus : -1), wall_s=wall))
        flush(stdout)
    end
    return rows
end

d4_rows = run_fixture_profile("D4_seed29_W20000",
    joinpath(OUTDIR, "melitz_phase2_gamma_profile_d4_2026-07-28.csv"), build_d4_fixture)

d20_rows = run_fixture_profile("realD20_seed1_W80000",
    joinpath(OUTDIR, "melitz_phase2_gamma_profile_realD20_2026-07-28.csv"), build_realD20_fixture)

allrows = vcat(d4_rows, d20_rows)
open(joinpath(OUTDIR, "melitz_post_consolidation_phase2_profile_recert_2026-07-28.csv"), "w") do io
    println(io, "fixture,label,g,result_type,Delta,nStatus,wall_s")
    for r in allrows
        println(io, "$(r.fixture),$(r.label),$(r.g),$(r.result_type),$(r.Delta),$(r.nStatus),$(r.wall_s)")
    end
end
println("\nWrote docs/key_results/melitz_post_consolidation_phase2_profile_recert_2026-07-28.csv")
println("\nDONE Phase 2 (post-consolidation).")
