# ============================================================================
# Continuation 10 (branch c10-stratified-marginal): targeted follow-up to
# c10_stratmarg_screen_sweep.jl's main sweep result.
#
# The main sweep found a REAL, substantial reduction in zero-winner-pair
# incidence from stratified-marginal sampling at W=20000 (calibration: plain
# 10/15 structurally-feasible reps vs stratified 14/15; upper_candidate:
# plain 12/15 vs stratified 14/15) but neither scheme hits literal 100%
# reliability at W=20000, so the main sweep's strict "did this W FLIP from
# unusable to reliably usable" trigger correctly did not fire (see
# followup_real_solves.csv from that run: only the always-run W=80000 plain
# benchmark rows are there).
#
# This script picks out a SPECIFIC, genuinely discriminating replicate at
# W=20000 -- rep=12, where the exact screen found plain pseudorandom
# STRUCTURALLY INFEASIBLE (2 zero-win positive-share pairs) but stratified-
# marginal STRUCTURALLY FEASIBLE (0 zero-win pairs), for BOTH points (see
# screen_sweep.csv rows point=calibration/upper_candidate, W=20000, rep=12) --
# and asks the two follow-up questions the task brief calls for:
#   (a) does the REAL inner CC dual solve (evaluate_fullA_fast, unmodified)
#       actually succeed at that (W, scheme, point), not just the
#       necessary-condition screen?
#   (b) how close does the resulting divergence (Delta_dual) come to the
#       W=80,000 plain-pseudorandom benchmark (already run by the main
#       sweep, results in followup_real_solves.csv) -- a bias check, since
#       fewer total draws can still bias the estimate even where feasible.
#       kappa is reported too but is trivially invariant whenever feasible
#       (gp is part of the FIXED evaluation point, not re-solved here).
# ============================================================================
include(joinpath(@__DIR__, "context_real_d20.jl"))   # unify-random-draw-production-pipeline 2026-07-30: qmc_context_real_d20.jl deleted, d20_real_setup now takes U= directly
include(joinpath(@__DIR__, "qmc_draws.jl"))
include(joinpath(@__DIR__, "c10_stratmarg_draws.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using KNITRO, Random, Statistics, LinearAlgebra, Dates, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const RUN_ID = "c10_stratmarg_followup_w20000_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_stratmarg_followup_w20000.jl starting ", now(), " commit=", COMMIT)

const FEASIBLE_CODES = (0, -100, -101, -103)
kappa_of(gp, σ) = 1 - gp^(σ / (σ - 1))
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function load_w_csv(path)
    vals = Float64[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        push!(vals, parse(Float64, line))
    end
    return vals
end

# ---- Same reference structural context as the main sweep (W=80000, production draws) ----
ctx_probe = d20_real_setup(W = 80000, find_smallest = true)
D = ctx_probe.D; σ = ctx_probe.σ
pe = build_pivot_elimination(ctx_probe)
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+D^2]
zfree0_natural = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx_probe.θ0_up[3+D]
w_calib = vcat(gp0, zfree0_natural)
w_upper = load_w_csv(joinpath(@__DIR__, "qmc_fixed_points", "upper_candidate_w.csv"))

const POINTS = [("calibration", w_calib, true), ("upper_candidate", w_upper, false)]
draw_fn(scheme::Symbol) = scheme == :plain ? pseudorandom_U : stratified_marginal_U

function run_real_solve(label, Wt, scheme, seed, xf, find_smallest)
    f = draw_fn(scheme)
    U = f(Wt, D; seed = seed)
    ctx = d20_real_setup(W = Wt, U = U, find_smallest = find_smallest)
    # confirm the screen's verdict independently, right before the real solve
    θ_full = CS.reconstruct_full(xf, ctx.m)
    Pmat = target_shares(ctx)
    pc = precompute_pairwise_M(ctx)
    a = compute_a_od(θ_full, ctx)
    pres = pairwise_certificate(a, pc, Pmat)
    order = order_destinations(pres, D)
    wres = screen_hard_winners(θ_full, ctx, Pmat; order = order, full_scan = true)
    n_zero = sum((Pmat .> 0) .& (wres.win_counts .== 0))

    t0 = time()
    r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, use_cache = false, warm = false)
    t_solve = time() - t0
    kap = r.inner_status in FEASIBLE_CODES ? kappa_of(r.gamma_focal_prime, σ) : NaN
    logprint("  [", label, " W=", Wt, " ", scheme, " seed=", seed, "] screen n_zero_win_pairs=", n_zero,
             " (pairwise_infeasible=", pres.infeasible, ")  REAL inner_status=", r.inner_status,
             " Delta_dual=", r.Delta_dual, " kappa=", kap, " wall=", round(t_solve, digits = 2), "s")
    return (point = label, W = Wt, scheme = String(scheme), seed = seed, screen_n_zero_win_pairs = n_zero,
            screen_pairwise_infeasible = pres.infeasible, inner_status = r.inner_status,
            Delta_dual = r.Delta_dual, kappa = kap, wall_s = t_solve)
end

rows = NamedTuple[]
for (label, w, find_smallest) in POINTS
    xf = x_free_from_w(w, pe)
    logprint("\n", "="^90); logprint("POINT: ", label); logprint("="^90)
    # rep=12 at W=20000: plain is screen-infeasible (2 zero-win pairs), stratified is screen-feasible (0) -- see
    # screen_sweep.csv from c10_stratmarg_screen_sweep_20260719_130454.
    seed_plain = 100_000 * 20000 + 1_000 * 12 + 1
    seed_strat = 100_000 * 20000 + 1_000 * 12 + 2
    push!(rows, run_real_solve(label, 20000, :plain, seed_plain, xf, find_smallest))
    push!(rows, run_real_solve(label, 20000, :stratified, seed_strat, xf, find_smallest))
end

function write_csv_rows(path, rows)
    isempty(rows) && return
    keys_ = collect(propertynames(rows[1]))
    open(path, "w") do io
        println(io, join(string.(keys_), ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in keys_], ","))
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "followup_w20000_rep12.csv"), rows)
logprint("\nOUTDIR = ", OUTDIR)
close(LOGIO)
