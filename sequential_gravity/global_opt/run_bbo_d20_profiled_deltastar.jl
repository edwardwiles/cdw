# ============================================================================
# Global (population-based) version of the "profiled delta*" reformulation.
#
# The original formulation (run_bbo_d20_real.jl, this session's first pass) directly
# extremizes gamma'_focal over the JOINT (gamma'_focal, A_od) subject to delta*<=budget --
# the same structure the local KNITRO search uses, just gradient-free. That fights the same
# structural problem the local search does: a tight, expensive, IMPLICIT divergence
# constraint, handled here via a smooth infeasibility penalty (bbo_common.jl) rather than
# KNITRO's own scale-mismatched Lagrangian, but still a constrained search.
#
# REFORMULATION (proposed and pre-validated at D=4/D=20 with a LOCAL pattern-search method
# in derivative_diagnostics/run_profiled_delta_star_min.jl /
# run_profiled_delta_star_min_d20_real.jl -- read those first): FIX gamma'_focal at a target
# GT (it drops out as a free variable entirely) and MINIMIZE delta*(A_od) over A_od alone.
# This is a genuinely box-bounded, UNCONSTRAINED scalar minimization in D variables, with no
# gamma'-vs-A_od scale mismatch anywhere (gamma' is not a decision variable), and no penalty
# machinery needed for a budget constraint (there isn't one in the objective at all -- delta*
# itself IS the objective). The D=4 local-search version already found real headroom below
# the delta=1 budget at the constrained search's own gamma'* -- direct evidence the original
# constrained search (local OR the joint-global one) was leaving value on the table.
#
# This driver swaps :generating_set_search (local pattern search, the D=4/D=20 scripts' own
# choice) for a genuine population method (adaptive DE), per the explicit suggestion that
# "the same objective/bounds pair should work directly with any of BlackBoxOptim's population
# methods" -- and reuses this session's own log-space Acol parameterization (bbo_common.jl)
# instead of run_profiled_delta_star_min*.jl's raw +-1e4x focal_bounds box, since that raw
# box is a poor fit for population methods (see bbo_common.jl's own header comment).
#
# The OBJECTIVE itself is exactly `exact_inner_divergence_at` (fixed_A_incumbent.jl,
# already included by run_profiled_production.jl, already validated in Part 10 of
# full_d2_correction_report.md) -- a fresh gravity-linearization freeze plus a real KNITRO
# inner CC-dual solve at every candidate A_od. No new inner-dual machinery here.
#
# GT is reused from the ALREADY-COMPUTED D=20 scaled local-search result (kappa=0.081518,
# see full_d2_correction_report.md section 7.2) -- exactly matching
# run_profiled_delta_star_min_d20_real.jl's own reuse choice, so results are comparable.
# gamma'_focal is NOT searched by this driver at all -- it only asks "starting from this
# already-known-good gamma', is there an A_od with strictly more slack than the local search
# found?" (If real headroom is found, the natural follow-up -- not yet built here -- is to
# bisect on GT itself using this same box-only inner minimization as the feasibility check;
# see section 5 of the writeup for why that step is out of scope for this run.)
#
#   FAKEDATA=3 DVAL=20 WVAL=80000 PARALLEL_INVERSION=true \
#     BBO_MAXTIME=10800 BBO_POPSIZE=16 \
#     julia -t 19 --project=. sequential_gravity/global_opt/run_bbo_d20_profiled_deltastar.jl
# ============================================================================
ENV["SKIP_BATCH_LOOP"] = "true"
ENV["FAKEDATA"] = get(ENV, "FAKEDATA", "3")
ENV["DVAL"] = get(ENV, "DVAL", "20")
ENV["WVAL"] = get(ENV, "WVAL", "80000")
ENV["PARALLEL_INVERSION"] = get(ENV, "PARALLEL_INVERSION", "true")
include(joinpath(@__DIR__, "..", "run_profiled_production.jl"))
using BlackBoxOptim, Printf, Dates, JLD2, LinearAlgebra, Random
using BlackBoxOptim: num_func_evals, f_calls
include(joinpath(@__DIR__, "bbo_common.jl"))

const SAVED_PATH = joinpath(@__DIR__, "..", "batch_out_realD20_W80000_fixeddualfdfull_scaled05", "seq_upper_delta1.0.jld2")
const MAXTIME = parse(Float64, get(ENV, "BBO_MAXTIME", "10800"))     # default 3h
const POPSIZE = parse(Int, get(ENV, "BBO_POPSIZE", "16"))
# SEED controls BlackBoxOptim's initial (random-uniform-in-box) population and its whole
# subsequent mutation/crossover trajectory (confirmed directly: same seed -> bit-identical
# best_candidate trajectory; different seed -> a genuinely different run). Used for the
# multistart/seed-sensitivity comparison -- default 1 preserves a single deterministic run if
# unset (the section-4.2 production run itself did NOT set this, so it used whatever the
# process's own default RNG state was; not reproducible by seed alone, but its own log is saved).
const SEED = parse(Int, get(ENV, "SEED", "1"))
Random.seed!(SEED)

@assert D == 20 "this driver is D=20-specific"
@assert W == 80000 "this driver is W=80000-specific (matches the saved scaled run)"

saved = JLD2.load(SAVED_PATH)
@assert saved["D"] == 20 && saved["delta"] == 1.0 && saved["bound"] == "upper" &&
        saved["gradient_method"] == "fixed_dual_fd_full" && saved["use_var_scaling"] == true "saved jld2 spec mismatch -- not the section-7.2 scaled run"

const GT = saved["best_feasible_gp"]
const κ_saved = saved["best_feasible_kappa"]
const θ_saved_best = Float64.(saved["best_feasible_theta"])
const Aod_constrained = θ_saved_best[4:3+D]

@assert isapprox(θr0[1], θ_saved_best[1]; rtol = 1e-8) && isapprox(θr0[2], θ_saved_best[2]; rtol = 1e-8) "MISMATCH: this driver's real-data setup does not match the saved D=20 scaled run"

@printf("\n===== Profiled global delta*(A_od) minimization, D=20 real data =====\n")
@printf("Reusing saved GT (NOT searching gamma'_focal): gamma'_focal* = %.12f -> kappa = %.6f\n", GT, κ_saved)
@printf("  (source: %s)\n", SAVED_PATH)
@printf("PopulationSize=%d MaxTime=%.0fs LOGBOUND=%.2f SEED=%d\n\n", POPSIZE, MAXTIME, LOGBOUND, SEED)
flush(stdout)

function build_theta_profiled(logratio::AbstractVector)
    θ = copy(θr0)
    θ[3] = GT
    θ[4:3+D] .= Acol_star .* exp.(logratio)
    θ
end

# exact_inner_divergence_at's own inner KNITRO CC-dual solve (cold-started, no warm start
# unlike the production sequential loop's own warm-started calls) can fail outright at
# far-from-A* candidates and return a huge sentinel value (observed: exactly 1e10, matching
# the `δ_star_initial = 1.0e10` fallback visible in this repo's own setup diagnostics) rather
# than throwing -- NOT caught by `gravity_ok` (gravity itself can converge fine even when the
# dual solve on top of it fails). Any δ_star at or above this threshold is treated as a solve
# failure, not a genuine achieved divergence -- no legitimate candidate near a delta=1 budget
# should ever score in the hundreds given how smoothly divergence scaled with distance from A*
# in multistart_screening_d20.jl (max observed div(p)~1.5 out to relΔA~20).
const MAX_SANE_DELTA_STAR = 50.0

function profiled_fitness(logratio::AbstractVector)
    θ = build_theta_profiled(logratio)
    local r
    try
        r = exact_inner_divergence_at(θ)
    catch
        return 1.0e4 + 1.0e-3 * norm(logratio)
    end
    (r.gravity_ok && isfinite(r.δ_star) && r.δ_star < MAX_SANE_DELTA_STAR) || return 1.0e4 + 1.0e-3 * norm(logratio)
    r.δ_star
end

logratio_range() = [(-LOGBOUND, LOGBOUND) for _ in 1:D]

const CKPT_DIR = get(ENV, "BBO_CKPT_DIR", joinpath(@__DIR__, "d20_checkpoints"))
isdir(CKPT_DIR) || mkpath(CKPT_DIR)
const CKPT_PATH = joinpath(CKPT_DIR, "bbo_d20_profiled_deltastar_upper_seed$(SEED).jld2")

function save_checkpoint(x, f, nevals, wall, done::Bool)
    Acol_best = Acol_star .* exp.(x)
    relΔA = norm(Acol_best .- Acol_star) / norm(Acol_star)
    JLD2.save(CKPT_PATH, Dict(
        "logratio_best" => x, "delta_star_best" => f, "num_evals" => nevals,
        "Acol_best" => Acol_best, "gammap_target" => GT, "kappa_target" => κ_saved,
        "relDeltaA" => relΔA, "delta_budget" => 1.0, "D" => D, "W" => W, "wall" => wall,
        "done" => done, "timestamp" => string(Dates.now())))
    relΔA
end

function checkpoint_callback(oc)
    x = best_candidate(oc); f = best_fitness(oc)
    relΔA = save_checkpoint(x, f, num_func_evals(oc), NaN, false)
    @printf("[checkpoint %s] fevals=%d best_delta_star=%.6f  gap_to_budget=%.6f  relΔA=%.3f\n",
            Dates.format(Dates.now(), "HH:MM:SS"), num_func_evals(oc), f, 1.0 - f, relΔA)
    flush(stdout)
end

# ---- Baseline checks: exact delta* AT A* and AT the (scaled) constrained search's own
# endpoint, before optimizing -- same as run_profiled_delta_star_min_d20_real.jl ----
t0 = time()
δ_at_Astar = profiled_fitness(zeros(D))
@printf("delta*(A_od = A*)                    = %.8f  (%.1fs)\n", δ_at_Astar, time() - t0)
t0 = time()
δ_at_constrained = profiled_fitness(log.(Aod_constrained ./ Acol_star))
@printf("delta*(A_od = scaled-search endpoint) = %.8f  (%.1fs)\n", δ_at_constrained, time() - t0)
flush(stdout)

t0 = time()
res = bboptimize(profiled_fitness;
    SearchRange = logratio_range(), NumDimensions = D,
    Method = :adaptive_de_rand_1_bin_radiuslimited,
    PopulationSize = POPSIZE, MaxTime = MAXTIME,
    TraceMode = :compact, TraceInterval = 30.0,
    CallbackFunction = checkpoint_callback, CallbackInterval = 0.0)
wall = time() - t0

xbest = best_candidate(res); δbest = best_fitness(res)
Acol_best = Acol_star .* exp.(xbest)
relΔA = norm(Acol_best .- Acol_star) / norm(Acol_star)

@printf("\n===== PROFILED GLOBAL delta* MINIMIZATION DONE: wall=%.1fs fevals=%d =====\n", wall, f_calls(res))
@printf("  delta*_min                 = %.8f  (original budget = 1.0)\n", δbest)
@printf("  gap = budget - delta*_min  = %.6f\n", 1.0 - δbest)
@printf("  relΔA vs A*                = %.3f\n", relΔA)
@printf("  GT (gamma'_focal, FIXED throughout) = %.9f -> kappa = %.6f  (unchanged by this search)\n", GT, κ_saved)
if (1.0 - δbest) > 0.02
    println("  => MEANINGFULLY LOWER than the budget: real headroom left on the table even after KNITRO-scaling.")
elseif (1.0 - δbest) < -0.02
    println("  => the global minimum EXCEEDS the budget -- unexpected (the scaled-search endpoint is itself")
    println("     delta*<=1-feasible by construction) -- check optimizer convergence / MaxTime before trusting this.")
else
    println("  => approximately EQUAL to the budget: the scaled constrained search already found (close to) the true optimum.")
end

save_checkpoint(xbest, δbest, f_calls(res), wall, true)
println("\nBBO_D20_PROFILED_DELTASTAR_DONE")
