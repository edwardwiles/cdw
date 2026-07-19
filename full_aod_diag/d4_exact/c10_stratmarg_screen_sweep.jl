# ============================================================================
# Continuation 10 (branch c10-stratified-marginal): does per-country
# stratified-marginal (Latin Hypercube) sampling reduce/eliminate zero-winner
# bilateral-pair incidence at W smaller than the current 80,000 production
# default, vs plain pseudorandom MC at the SAME W?
#
# Background: docs/fullA_D20_infeasibility_screening_report.md sec 5 found
# W=8,000 uniformly infeasible at D=20 real data because 5/400 positive-share
# (origin,destination) pairs get literally ZERO winning draws on that draw
# support (confirmed exactly by the pairwise draw-free certificate) -- a
# TAIL-COVERAGE problem in each country's own marginal, not a smoothness
# problem. The prior QMC investigation (docs/fullA_D20_qmc_investigation_report.md,
# branch c10-qmc-is) already asked and answered "does a smarter design reduce
# gradient/kappa NOISE" (no) using scrambled Halton/Sobol -- THIS script asks
# a genuinely different question (coverage/feasibility, not precision) using a
# genuinely different design (per-country stratified marginals / Latin
# Hypercube, c10_stratmarg_draws.jl::stratified_marginal_U -- NOT Halton/Sobol).
#
# ADDITIVE ONLY. Reuses, does not modify: qmc_context_real_d20.jl's real
# inverse-transform + draw-injection fork, infeasibility_screen.jl's exact
# pairwise certificate + tie-safe winner scan (both read-only, unmodified),
# qmc_draws.jl's pseudorandom_U baseline generator.
#
# Test matrix: W in {8000, 20000, 40000, 80000} x scheme in {plain
# pseudorandom, stratified-marginal} x point in {calibration (gp0,A0),
# delta=1 upper candidate (gamma'=0.955701, its A, from
# qmc_fixed_points/upper_candidate_w.csv -- same provenance as c10_phase7's
# QMC comparison)}. Replicate count is TIERED by per-W setup cost (measured:
# d20_real_setup(W=80000) wall ~54s vs W=8000 wall ~3.6s in
# results/fullA_d4/b200eda/c9_infscreen_d20/harness_log.txt), same tiering
# principle c10_phase7_qmc_precision_comparison.jl's SCRAMBLE_PLAN used:
#   W=8000: 15 reps, W=20000: 15 reps, W=40000: 12 reps, W=80000: 10 reps
#   (80,000 is a REGRESSION check here -- stratification should not hurt the
#   current production point -- not the focus, so fewer reps is fine).
#
# Metrics per (point, W, scheme), aggregated across replicates:
#   - zero-winner-pair incidence (count out of 400 positive-share pairs with
#     0 winning draws), mean/std/min/max across replicates
#   - pairwise-certificate infeasibility verdict (fraction of replicates
#     rejected) and winner-scan feasibility verdict (should agree exactly at
#     D=20 per the screening report's own finding that pairwise catches
#     100% of genuine D=20 failures in every sweep tried so far)
#   - draw-generation wall time (t_draw) and context-build wall time (t_ctx)
#
# Follow-up (only if some W<80000 shows a measurable improvement): real
# inner CC dual solve status (evaluate_fullA_fast, unmodified) at that
# W/point/scheme, plus a W=80000 plain-pseudorandom benchmark solve for the
# bias comparison (kappa is trivially invariant whenever feasible, since gp
# is part of the FIXED evaluation point, not re-solved -- Delta_dual, the
# actual estimated divergence, is the metric that can genuinely shift with
# fewer draws; both are reported).
# ============================================================================
include(joinpath(@__DIR__, "qmc_context_real_d20.jl"))   # -> d20_real_setup, d20_real_setup_qmc, exp_from_uniform01, screen fns (transitively, via context_real_d20.jl)
include(joinpath(@__DIR__, "qmc_draws.jl"))               # -> pseudorandom_U (baseline)
include(joinpath(@__DIR__, "c10_stratmarg_draws.jl"))     # -> stratified_marginal_U (this task's new generator)
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
const RUN_ID = "c10_stratmarg_screen_sweep_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_stratmarg_screen_sweep.jl starting ", now(), " commit=", COMMIT, " nthreads=", Threads.nthreads())

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

# ============================================================================
# Reference structural context (ONE-TIME, W=80000, production draws) -- gives
# D, sigma, the pivot-elimination map `pe` (STRUCTURAL: pivot = argmax|gravity
# tangent coeff|, a pure function of real data tau/N_obs, confirmed
# W/U-independent by reading gravity_elimination.jl::build_pivot_elimination
# -- safe to reuse across every test W below), and the natural-theta
# calibration point (gp0, A0). Point held FIXED across all draw realizations
# tested below (same experimental design c10_phase7_qmc_precision_comparison.jl
# used) -- we are testing whether the SAME outer-loop point stays
# winner-feasible under different draw sets, not re-deriving a new
# calibration point per draw set.
# ============================================================================
logprint("\n", "="^90); logprint("Building reference structural context (W=80000, production draws)"); logprint("="^90)
const W_REFERENCE = 80000
t0 = time()
ctx_probe = d20_real_setup(W = W_REFERENCE, find_smallest = true)
logprint("d20_real_setup(W=", W_REFERENCE, ") wall=", round(time() - t0, digits = 1), "s")
D = ctx_probe.D; σ = ctx_probe.σ
pe = build_pivot_elimination(ctx_probe)
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+D^2]
zfree0_natural = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx_probe.θ0_up[3+D]
w_calib = vcat(gp0, zfree0_natural)
logprint("Calibration: gp0=", gp0, " norm(zfree0)=", norm(zfree0_natural))

w_upper = load_w_csv(joinpath(@__DIR__, "qmc_fixed_points", "upper_candidate_w.csv"))
@assert length(w_upper) == D^2 "expected $(D^2) (gp + $(D^2-1) zfree), got $(length(w_upper))"
logprint("Loaded upper candidate: gp=", w_upper[1], " (expect 0.955701), norm(zfree)=", norm(w_upper[2:end]))

const POINTS = [
    ("calibration", w_calib, true),
    ("upper_candidate", w_upper, false),
]

draw_fn(scheme::Symbol) = scheme == :plain ? pseudorandom_U : stratified_marginal_U
const SCHEMES = (:plain, :stratified)
const W_LIST = [8000, 20000, 40000, 80000]
const REP_PLAN = Dict(8000 => 15, 20000 => 15, 40000 => 12, 80000 => 10)

# ============================================================================
# MAIN SWEEP: pairwise certificate + full destination-major winner scan
# (screen_hard_winners with full_scan=true, so every destination is always
# scanned and we get the EXACT total zero-win-pair count, not just the
# first-failure early-exit point) at each (point, W, scheme, replicate).
# ============================================================================
rows = NamedTuple[]
for (label, w, find_smallest) in POINTS
    xf = x_free_from_w(w, pe)
    logprint("\n", "="^90); logprint("POINT: ", label, "  gp=", w[1], "  find_smallest=", find_smallest)
    logprint("="^90)
    for Wt in W_LIST
        n_rep = REP_PLAN[Wt]
        for scheme in SCHEMES
            f = draw_fn(scheme)
            for r in 1:n_rep
                seed = 100_000 * Wt + 1_000 * r + (scheme == :plain ? 1 : 2)
                t_draw0 = time(); U = f(Wt, D; seed = seed); t_draw = time() - t_draw0
                t_ctx0 = time(); ctx = d20_real_setup_qmc(W = Wt, U_injected = U, find_smallest = find_smallest); t_ctx = time() - t_ctx0

                θ_full = CS.reconstruct_full(xf, ctx.m)
                Pmat = target_shares(ctx)
                pc = precompute_pairwise_M(ctx)
                a = compute_a_od(θ_full, ctx)
                pres = pairwise_certificate(a, pc, Pmat)
                order = order_destinations(pres, D)
                wres = screen_hard_winners(θ_full, ctx, Pmat; order = order, full_scan = true)
                n_zero = sum((Pmat .> 0) .& (wres.win_counts .== 0))
                n_pos = count(>(0), Pmat)

                push!(rows, (point = label, W = Wt, scheme = String(scheme), rep = r, seed = seed,
                    pairwise_infeasible = pres.infeasible, winner_scan_feasible = wres.feasible,
                    n_zero_win_pairs = n_zero, n_positive_share_pairs = n_pos,
                    worst_slack = pres.worst_slack, t_draw = t_draw, t_ctx = t_ctx))
            end
            zeros_this = [r.n_zero_win_pairs for r in rows if r.point == label && r.W == Wt && r.scheme == String(scheme)]
            logprint("  [", label, " W=", Wt, " ", scheme, "] n_zero_win_pairs across ", n_rep, " reps: mean=",
                     round(mean(zeros_this), digits = 2), " std=", round(std(zeros_this), digits = 2),
                     " min=", minimum(zeros_this), " max=", maximum(zeros_this),
                     " (", count(==(0), zeros_this), "/", n_rep, " fully-feasible reps)")
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "screen_sweep.csv"), rows)
logprint("\nWrote ", joinpath(OUTDIR, "screen_sweep.csv"), " (", length(rows), " rows)")

# ============================================================================
# AGGREGATE TABLE (mean/std/range per point/W/scheme) -- written separately
# for easy consumption by the report.
# ============================================================================
agg_rows = NamedTuple[]
for (label, w, find_smallest) in POINTS, Wt in W_LIST, scheme in SCHEMES
    sub = [r for r in rows if r.point == label && r.W == Wt && r.scheme == String(scheme)]
    isempty(sub) && continue
    zeros_ = [r.n_zero_win_pairs for r in sub]
    push!(agg_rows, (point = label, W = Wt, scheme = String(scheme), n_reps = length(sub),
        mean_n_zero = mean(zeros_), std_n_zero = std(zeros_), min_n_zero = minimum(zeros_), max_n_zero = maximum(zeros_),
        frac_fully_feasible = count(==(0), zeros_) / length(sub),
        frac_pairwise_infeasible = count(r -> r.pairwise_infeasible, sub) / length(sub),
        frac_winner_scan_feasible = count(r -> r.winner_scan_feasible, sub) / length(sub),
        mean_t_draw = mean([r.t_draw for r in sub]), mean_t_ctx = mean([r.t_ctx for r in sub])))
end
write_csv_rows(joinpath(OUTDIR, "screen_sweep_aggregate.csv"), agg_rows)
logprint("\n", "="^90); logprint("AGGREGATE TABLE"); logprint("="^90)
for a in agg_rows
    logprint("  ", a.point, " W=", a.W, " ", a.scheme, ": mean_n_zero=", round(a.mean_n_zero, digits = 2),
             " std=", round(a.std_n_zero, digits = 2), " range=[", a.min_n_zero, ",", a.max_n_zero, "]",
             " frac_fully_feasible=", round(a.frac_fully_feasible, digits = 2),
             " frac_pairwise_infeasible=", round(a.frac_pairwise_infeasible, digits = 2))
end

# ============================================================================
# FOLLOW-UP: for each point, does stratification make some W < 80000 FULLY
# feasible (frac_fully_feasible == 1.0) where plain does not? If so, run a
# REAL inner CC dual solve there (both schemes, first replicate's seed) plus
# a W=80000 plain-pseudorandom benchmark solve, for the bias comparison.
# ============================================================================
logprint("\n", "="^90); logprint("FOLLOW-UP: real inner-solve check where stratification changes feasibility"); logprint("="^90)
followup_rows = NamedTuple[]

function run_real_solve(label, Wt, scheme, seed, xf, find_smallest)
    f = draw_fn(scheme)
    U = f(Wt, D; seed = seed)
    ctx = d20_real_setup_qmc(W = Wt, U_injected = U, find_smallest = find_smallest)
    t0 = time()
    r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, use_cache = false, warm = false)
    t_solve = time() - t0
    kap = r.inner_status in FEASIBLE_CODES ? kappa_of(r.gamma_focal_prime, σ) : NaN
    logprint("  [", label, " W=", Wt, " ", scheme, " seed=", seed, "] inner_status=", r.inner_status,
             " Delta_dual=", r.Delta_dual, " kappa=", kap, " wall=", round(t_solve, digits = 2), "s")
    return (point = label, W = Wt, scheme = String(scheme), seed = seed, inner_status = r.inner_status,
            Delta_dual = r.Delta_dual, kappa = kap, wall_s = t_solve)
end

for (label, w, find_smallest) in POINTS
    xf = x_free_from_w(w, pe)
    # Always run the W=80000 plain benchmark for the bias comparison.
    seed80k = 100_000 * 80000 + 1_000 * 1 + 1
    push!(followup_rows, run_real_solve(label, 80000, :plain, seed80k, xf, find_smallest))

    for Wt in (8000, 20000, 40000)
        agg_plain = only([a for a in agg_rows if a.point == label && a.W == Wt && a.scheme == "plain"])
        agg_strat = only([a for a in agg_rows if a.point == label && a.W == Wt && a.scheme == "stratified"])
        newly_feasible = agg_strat.frac_fully_feasible == 1.0 && agg_plain.frac_fully_feasible < 1.0
        logprint("  ", label, " W=", Wt, ": plain frac_fully_feasible=", agg_plain.frac_fully_feasible,
                 " stratified frac_fully_feasible=", agg_strat.frac_fully_feasible,
                 " -> newly_feasible_via_stratification=", newly_feasible)
        if newly_feasible
            seed = 100_000 * Wt + 1_000 * 1 + 2   # first stratified replicate's seed
            push!(followup_rows, run_real_solve(label, Wt, :stratified, seed, xf, find_smallest))
            seed_plain = 100_000 * Wt + 1_000 * 1 + 1
            push!(followup_rows, run_real_solve(label, Wt, :plain, seed_plain, xf, find_smallest))
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "followup_real_solves.csv"), followup_rows)

logprint("\nOUTDIR = ", OUTDIR)
close(LOGIO)
