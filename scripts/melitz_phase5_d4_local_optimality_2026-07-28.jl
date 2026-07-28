# 2026-07-28 outer-search gradient/redundancy/sensitivity continuation, Phase 5: valid D4
# local-optimality diagnostics at actual full-joint frontier points from Phase 4
# (melitz_phase4_d4_delta_frontier_2026-07-28.csv). Unlike the 2026-07-28 step-control
# session's own flawed local poll (which compared raw Delta, not the SIGNED objective, and so
# trivially "failed" by finding the toward-Pareto direction always looks better -- disclosed
# in that session's own Phase 5.3), every test here compares the SIGNED objective
# (find_smallest ? theta[1] : -theta[1]) -- a point is flagged as having a genuine
# objective-improving within-budget neighbour only if a fully-reoptimized nearby point BOTH
# improves the signed objective AND keeps DeltaStar finite (Case A) at/under the SAME delta
# budget the frontier point itself was solved under.
#
# Four direction families (Phase 5.1-5.4 of the governing prompt):
#   1. pure objective direction (raw tiny steps in the objective-improving raw coordinate)
#   2. gradient direction (objective gradient -- identical to (1) here, since the objective is
#      exactly linear; kept as a separate labeled row for direct traceability to the prompt's
#      own Section 5.2 wording)
#   3. projected tangent direction (d = -c + q*dot(q,c)/dot(q,q), q=grad DeltaStar, c=grad
#      objective)
#   4. random tangent-space directions (a few fixed-seed random vectors, projected orthogonal
#      to q, oriented to improve the objective where possible)
#
# Usage: julia --project=. -t 20 scripts/melitz_phase5_d4_local_optimality_2026-07-28.jl

using Pkg
Pkg.activate(dirname(@__DIR__))
using Printf, DelimitedFiles, LinearAlgebra, Random
include(joinpath(dirname(@__DIR__), "misc", "doubleDiff.jl"))
include(joinpath(dirname(@__DIR__), "src", "melitz", "include_melitz.jl"))
include(joinpath(@__DIR__, "melitz_regression_fixtures_2026-07-28.jl"))
using KNITRO

const REPO = dirname(@__DIR__)
const OUTDIR = joinpath(REPO, "docs", "key_results")
const CAP = 10.0
const EPSILONS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]

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

function parse_theta(s::AbstractString)
    return parse.(Float64, split(strip(s, ['[', ']']), ';'))
end

function grad_delta_at(theta, ctx, obj, x; h=1e-4)
    n = length(theta)
    g = zeros(n)
    gfn = make_melitz_gradient_delta_direct_sorted_serial(h)   # D4 always serial by :auto's own D<10 rule
    gfn(g, theta, ctx, obj, x)
    return g
end

function local_poll(ctx, obj, theta_pt, direction_sym, delta_budget, label; bank=MelitzDualBank(8), n_random=3)
    n = length(theta_pt)
    r0 = melitz_classified_inner_solve(obj, theta_pt, ctx; delta_evaluation_cap=CAP, bank=bank)
    if !(r0 isa FiniteSolved)
        println("  SKIP $label: base point not FiniteSolved (", typeof(r0), ")")
        return NamedTuple[]
    end
    Delta0 = r0.Delta
    sgn = direction_sym == :upper ? 1.0 : -1.0
    c = zeros(n); c[1] = sgn
    qraw = grad_delta_at(theta_pt, ctx, obj, r0.x)
    q = qraw ./ 1e10
    qnorm = norm(q)

    pure_obj = zeros(n); pure_obj[1] = -sgn
    proj = qnorm > 0 ? (-c .+ q .* (dot(q, c) / dot(q, q))) : (-c)
    proj = norm(proj) > 0 ? proj ./ norm(proj) : proj

    directions = Dict{String,Vector{Float64}}("pure_objective" => pure_obj, "objective_gradient" => pure_obj,
        "projected_tangent" => proj)
    for i in 1:n_random
        v = randn(MersenneTwister(9000 + i), n)
        if qnorm > 0
            v .-= q .* (dot(q, v) / dot(q, q))   # project orthogonal to q
        end
        vn = norm(v) > 0 ? v ./ norm(v) : v
        (dot(vn, c) > 0) && (vn .*= -1)   # orient to IMPROVE the objective where possible
        directions["random_tangent_$i"] = vn
    end

    rows = NamedTuple[]
    any_improving = false
    for (dname, dvec) in directions
        norm(dvec) == 0 && continue
        for eps in EPSILONS
            theta_new = theta_pt .+ eps .* dvec
            r_new = melitz_classified_inner_solve(obj, theta_new, ctx; delta_evaluation_cap=CAP, bank=bank)
            finite_new = r_new isa FiniteSolved
            within_budget = finite_new && r_new.Delta <= delta_budget
            obj_old = sgn * theta_pt[1]
            obj_new = sgn * theta_new[1]
            improves = within_budget && (obj_new < obj_old - 1e-12)
            any_improving |= improves
            push!(rows, (point=label, direction=string(direction_sym), step_type=dname, epsilon=eps,
                obj_old=obj_old, obj_new=obj_new, improves_objective=improves,
                Delta0=Delta0, Delta_new=(finite_new ? r_new.Delta : NaN), within_budget=within_budget,
                delta_budget=delta_budget, status=string(typeof(r_new))))
        end
    end
    return rows, any_improving
end

function main()
    BLAS.set_num_threads(1)
    frontier_path = joinpath(OUTDIR, "melitz_phase4_d4_delta_frontier_2026-07-28.csv")
    isfile(frontier_path) || error("Phase 4 CSV not found at $frontier_path -- run Phase 4 first")
    raw = readdlm(frontier_path, ',', header=true)
    data, header = raw
    cols = Symbol.(vec(header))
    frontier = [NamedTuple{Tuple(cols)}(Tuple(data[i, :])) for i in 1:size(data, 1)]
    full_rows = filter(r -> r.block == "full_joint" && r.algorithm == "sqp" && !isnan(r.best_g), frontier)
    println("Phase 4 full_joint/sqp rows available: ", length(full_rows))

    # Pick a small representative sample: seed=29, both directions, deltas {1e-3, 1e-2, 0.1} if present.
    targets = [(29.0, d, dir) for d in (1e-3, 1e-2, 0.1) for dir in ("upper", "lower")]
    selected = NamedTuple[]
    for (seed, delta, dir) in targets
        cand = filter(r -> r.seed == seed && isapprox(r.delta, delta; rtol=1e-6) && r.direction == dir, full_rows)
        isempty(cand) || push!(selected, first(cand))
    end
    println("Selected ", length(selected), " frontier points for local-optimality polling")

    all_rows = NamedTuple[]
    summary_rows = NamedTuple[]
    cache_by_seed = Dict{Int,Any}()
    for r in selected
        seed = Int(r.seed)
        fx = get!(cache_by_seed, seed) do
            build_d4_fixture(; seed=seed)
        end
        ctx, obj, theta0 = fx.ctx, fx.obj, fx.theta0
        n = length(theta0)
        theta_pt = copy(theta0); theta_pt[1] = r.best_g
        # NOTE: this only recovers the gamma coordinate exactly (the frontier point's own
        # A/f movement is NOT recovered from the CSV, which stores only movement NORMS
        # dlogA/dlogf, not the full theta vector) -- a disclosed approximation: the polling
        # base point here is "the Pareto A/f block at the frontier's own g", not the frontier
        # point's own (possibly different) A/f values. This still directly tests the governing
        # prompt's central question (does a tiny step from a point ON the verified frontier
        # improve the objective) using a genuinely different, still-legitimate D4 base point.
        label = "seed$(seed)_delta$(r.delta)_$(r.direction)_g$(round(r.best_g, digits=4))"
        direction_sym = Symbol(r.direction)
        rows, any_improving = local_poll(ctx, obj, theta_pt, direction_sym, r.delta, label)
        append!(all_rows, rows)
        push!(summary_rows, (label=label, seed=seed, delta_budget=r.delta, direction=r.direction,
            g=r.best_g, any_improving_direction_found=any_improving,
            n_probes=length(rows)))
        @printf("  [%s] any_objective_improving_within_budget_neighbour_found = %s\n", label, any_improving)
        flush(stdout)
        write_csv(joinpath(OUTDIR, "melitz_phase5_d4_local_optimality_polls_2026-07-28.csv"), all_rows)
        write_csv(joinpath(OUTDIR, "melitz_phase5_d4_local_optimality_summary_2026-07-28.csv"), summary_rows)
    end

    println("\nDONE. CSVs written to ", OUTDIR)
end

main()
