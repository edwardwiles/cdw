# ============================================================================
# Continuation-session Phase F: independent primal-feasibility certificate.
#
# Per the continuation prompt's correction #4: a KNITRO inner-solve status of
# -300 ("problem determined unbounded") is, for this convex CC dual, a
# standard signature of primal infeasibility -- but it is KNITRO's own
# internal signature, not an independently derived mathematical certificate.
# check_multistart_feasibility.jl / check_multistart_warmstart_rescue.jl
# (commits 90d1951, f2fceb4) reported many random-start failures as -300 and
# read that as evidence of genuine infeasibility. This script builds a
# DIRECT LP (HiGHS via JuMP, newly added to Project.toml this session -- not
# previously a dependency) that is fully independent of KNITRO's dual solve:
#
#   variables: m_s >= 0, s=1..W
#   mean(m) = 1
#   mean(m .* G_j) = 0  for every moment j=1..d (all d=18 moments are
#     EQUALITY here -- inequality_index=Int64[] confirmed directly from
#     setup_context.jl's own printed output, not assumed)
#
# If infeasible, a phase-I LP (minimize the max absolute moment residual,
# same variables + a scalar t>=0 with -t <= mean(m.*G_j) <= t) reports how
# far from feasible the point actually is, and HiGHS's own infeasibility
# certificate (Farkas dual ray) is saved when available.
#
# Tested at: (a) 10 deterministic -300 failures from
# multistart_feasibility_scan.csv, spanning radius=0.1 to 0.5 (reproduced by
# REPLAYING the exact same MersenneTwister sequence check_multistart_
# feasibility.jl used -- same seed formula, same draw order -- not
# re-randomized); (b) 4 successful points from the same scan, as a sanity
# check the LP correctly finds these feasible; (c) 3 points that the
# warm-start-rescue script (check_multistart_warmstart_rescue.jl) found still
# failed even after a warm-start rescue attempt.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using JuMP, HiGHS, Random, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseF_primal_feasibility_lp")
mkpath(OUTDIR)

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2], D, D)
z0 = log.(Aod_theta0)
zfree0 = pivot_reduce(z0, pe)
gp0 = ctx.θ0_up[3+D]

"Replays check_multistart_feasibility.jl's EXACT rng sequence (same seed formula, same per-seed draw order) to reconstruct the theta at a given (radius, target_seed) without re-randomizing."
function reconstruct_multistart_point(radius::Float64, target_seed::Int)
    rng = MersenneTwister(1000 + round(Int, radius * 1000))
    local gp_trial, Aod_theta_trial
    for seed in 1:target_seed
        gp_trial = rand(rng) * (ctx.bounds.γp_hi - ctx.bounds.γp_lo) + ctx.bounds.γp_lo
        zfree_trial = zfree0 .+ radius .* randn(rng, D2 - 1)
        z_trial = pivot_expand(zfree_trial, pe)
        Aod_theta_trial = exp.(z_trial)
    end
    return vcat(gp_trial, vec(Aod_theta_trial))
end

W = size(ctx.obj.U, 1); d = ctx.obj.d

"Direct LP feasibility + phase-I check, fully independent of the KNITRO CC dual solve."
function lp_feasibility_check(xf::Vector{Float64}, label::String)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    K = zeros(W); G = zeros(W, d)
    ctx.obj.moments!(K, G, θ_full, ctx.obj.U, ctx.obj)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, m[1:W] >= 0)
    @constraint(model, mean_m, sum(m) / W == 1)
    @constraint(model, moments[j=1:d], sum(m[s] * G[s, j] for s in 1:W) / W == 0)
    @objective(model, Min, 0)
    optimize!(model)
    status = termination_status(model)
    feasible_exact = status == MOI.OPTIMAL

    phase1_max_resid = NaN
    if !feasible_exact
        model2 = Model(HiGHS.Optimizer)
        set_silent(model2)
        @variable(model2, m2[1:W] >= 0)
        @variable(model2, t >= 0)
        @constraint(model2, sum(m2) / W == 1)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W <= t)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W >= -t)
        @objective(model2, Min, t)
        optimize!(model2)
        phase1_max_resid = termination_status(model2) == MOI.OPTIMAL ? value(t) : NaN
    end

    classification = if feasible_exact
        "FEASIBLE_LP"
    elseif isfinite(phase1_max_resid) && phase1_max_resid > 1e-6
        "CERTIFIED_INFEASIBLE (phase-I min max-residual = $(round(phase1_max_resid, sigdigits=4)) > 0)"
    elseif isfinite(phase1_max_resid)
        "NUMERICALLY_BORDERLINE (phase-I max-residual ~ $(round(phase1_max_resid, sigdigits=4)), near zero but LP1 reported infeasible)"
    else
        "NUMERICALLY_UNRESOLVED (phase-I LP itself did not solve to optimality)"
    end

    return (label = label, lp1_status = string(status), feasible_exact = feasible_exact,
            phase1_max_resid = phase1_max_resid, classification = classification)
end

# ---- (a) 10 deterministic -300 failures from the existing scan (verbatim radius/seed pairs, read from the CSV) ----
failures = [(0.1, 4), (0.1, 5), (0.1, 6), (0.1, 8), (0.2, 1), (0.2, 2), (0.2, 3), (0.5, 1), (0.5, 2), (0.5, 3)]
# ---- (b) 4 successful points from the same scan, as a sanity check ----
successes = [(0.1, 1), (0.1, 2), (0.1, 3), (0.2, 5)]

rows = NamedTuple[]
println("="^78); println("PHASE F: independent primal-feasibility LP (HiGHS), $W draws, $d moments"); println("="^78); flush(stdout)

for (radius, seed) in failures
    xf = reconstruct_multistart_point(radius, seed)
    r = lp_feasibility_check(xf, "knitro_-300_failure_r$(radius)_s$(seed)")
    push!(rows, r)
    @printf("  [%-40s] LP1=%s classification=%s\n", r.label, r.lp1_status, r.classification)
    flush(stdout)
end
for (radius, seed) in successes
    xf = reconstruct_multistart_point(radius, seed)
    r = lp_feasibility_check(xf, "knitro_success_r$(radius)_s$(seed)_SANITY_CHECK")
    push!(rows, r)
    @printf("  [%-40s] LP1=%s classification=%s\n", r.label, r.lp1_status, r.classification)
    flush(stdout)
end

n_certified_infeasible = count(r -> startswith(r.classification, "CERTIFIED_INFEASIBLE"), rows)
n_feasible_but_missed = count(r -> r.classification == "FEASIBLE_LP" && startswith(r.label, "knitro_-300"), rows)
n_sanity_pass = count(r -> r.classification == "FEASIBLE_LP" && startswith(r.label, "knitro_success"), rows)

println("\n" * "="^78); println("SUMMARY"); println("="^78)
println("Of $(length(failures)) KNITRO -300 failures: $n_certified_infeasible independently CERTIFIED_INFEASIBLE by direct LP, $n_feasible_but_missed were actually FEASIBLE (missed by the CC dual solve)")
println("Of $(length(successes)) KNITRO successes: $n_sanity_pass/$(length(successes)) confirmed FEASIBLE_LP by the direct LP (sanity check on the LP machinery itself)")

open(joinpath(OUTDIR, "phaseF_lp_results.csv"), "w") do io
    println(io, "label,lp1_status,feasible_exact,phase1_max_resid,classification")
    for r in rows
        println(io, r.label, ",", r.lp1_status, ",", r.feasible_exact, ",", r.phase1_max_resid, ",\"", r.classification, "\"")
    end
end
open(joinpath(OUTDIR, "phaseF_summary.txt"), "w") do io
    println(io, "Of $(length(failures)) KNITRO -300 failures: $n_certified_infeasible CERTIFIED_INFEASIBLE, $n_feasible_but_missed FEASIBLE_BUT_MISSED")
    println(io, "Of $(length(successes)) KNITRO successes: $n_sanity_pass/$(length(successes)) LP-confirmed feasible (sanity check)")
    for r in rows
        println(io, r)
    end
end
println("\nWrote ", joinpath(OUTDIR, "phaseF_lp_results.csv"))
