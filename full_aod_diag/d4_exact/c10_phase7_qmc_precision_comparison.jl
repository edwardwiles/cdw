# ============================================================================
# Continuation 10, Phase 7 (Section 7 of the standing brief): the MAIN
# randomized-QMC-vs-pseudorandom-MC precision comparison at fixed W=80000,
# real D=20 data. Isolated (uses qmc_context_real_d20.jl's forked context
# builder, touches no production file). Does NOT run any full outer-loop
# optimization per scramble (far too expensive) -- per task spec, evaluates
# objective+gradient at FIXED points across independent scrambles, plus one
# short (maxit-capped) continuation from the known upper candidate.
#
# Fixed points (natural-theta w=[gamma', pivot-reduced zfree], D^2-1=399 free
# A-directions):
#   1. calibration: gp0, A0 (natural-theta calibration A_od)
#   2. upper candidate: gamma'=0.955701 (delta=1, large-kappa branch, this
#      investigation's headline "upper" result), A from the ACTUAL polish2
#      best_feasible point of results/fullA_d4/b200eda/c9_phase8_d20_pilot_.../
#      summary.txt (parsed verbatim from that run's saved text, not
#      re-derived/re-optimized -- see qmc_fixed_points/upper_candidate_w.csv
#      for provenance).
#   3. "extreme" point: a directional extrapolation PAST the upper candidate,
#      2x the calibration->upper displacement in w-space (w_calib + 2*(w_upper
#      - w_calib)) -- picked because it is cheap to construct from data
#      already in hand (no new optimization needed) and plausibly probes a
#      more constraint-violating/boundary-adjacent region than the upper
#      candidate itself, analogous in spirit to a larger-delta direction.
#      Explicitly a judgment call, flagged as such in the report.
#   4-5. two random directional perturbations around the upper candidate in
#      pivot-reduced z-space (h=0.05), same style as Phase 6/c9_phase8's
#      directional secant diagnostic.
# ============================================================================
include(joinpath(@__DIR__, "qmc_context_real_d20.jl"))
include(joinpath(@__DIR__, "qmc_draws.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "bandwidth_cache_policy.jl"))
using KNITRO, Sobol, Random, Statistics, LinearAlgebra, Dates, Printf

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const RUN_ID = "c10_phase7_qmc_precision_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const LOGPATH = joinpath(OUTDIR, "harness_log.txt")
const LOGIO = open(LOGPATH, "w")
function logprint(xs...)
    println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO)
end
logprint("c10_phase7_qmc_precision_comparison.jl starting ", now(), " commit=", COMMIT, " nthreads=", Threads.nthreads())

const W_REAL = 80000
const FEASIBLE_CODES = (0, -100, -101, -103)
kappa_of(gp, σ) = 1 - gp^(σ / (σ - 1))
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

# ---- Load the fixed points ----
function load_w_csv(path)
    vals = Float64[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        push!(vals, parse(Float64, line))
    end
    return vals
end
w_upper = load_w_csv(joinpath(@__DIR__, "qmc_fixed_points", "upper_candidate_w.csv"))
@assert length(w_upper) == 400 "expected 400 (gp + 399 zfree), got $(length(w_upper))"
logprint("Loaded upper candidate: gp=", w_upper[1], " (expect 0.955701), norm(zfree)=", norm(w_upper[2:end]))

# ---- Build a throwaway ctx just to get D, gp0/zfree0_natural, and pe (structural, U-independent) ----
ctx_probe = d20_real_setup(W = W_REAL, find_smallest = true)
pe = build_pivot_elimination(ctx_probe)
D = ctx_probe.D; σ = ctx_probe.σ
Aod_theta_natural = ctx_probe.θ0_up[ctx_probe.Aod_offset+1:ctx_probe.Aod_offset+D^2]
zfree0_natural = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx_probe.θ0_up[3+D]
w_calib = vcat(gp0, zfree0_natural)
logprint("Calibration: gp0=", gp0, " norm(zfree0)=", norm(zfree0_natural))

w_extreme = w_calib .+ 2.0 .* (w_upper .- w_calib)
logprint("Extreme (2x calib->upper extrapolation): gp=", w_extreme[1], " norm(zfree)=", norm(w_extreme[2:end]))

Random.seed!(13579)
dir1 = randn(length(w_upper) - 1); dir1 ./= norm(dir1)
dir2 = randn(length(w_upper) - 1); dir2 ./= norm(dir2)
const H_PERTURB = 0.05
w_perturb1 = vcat(w_upper[1], w_upper[2:end] .+ H_PERTURB .* dir1)
w_perturb2 = vcat(w_upper[1], w_upper[2:end] .+ H_PERTURB .* dir2)

const POINTS = [
    ("calibration", w_calib, true),
    ("upper_candidate", w_upper, false),
    ("extreme_2x", w_extreme, false),
    ("perturb1", w_perturb1, false),
    ("perturb2", w_perturb2, false),
]

# ---- Scramble plan: (drawtype, n_scrambles) per point label, tiered by cost/importance ----
const SCRAMBLE_PLAN = Dict(
    "calibration" => 6, "upper_candidate" => 6, "extreme_2x" => 4,
    "perturb1" => 4, "perturb2" => 4,
)
const GRADIENT_POINTS = ("calibration", "upper_candidate")   # gradient is expensive -- only these two
const DRAWTYPES = (:pseudorandom, :halton, :sobol)

draw_fn(kind::Symbol) = kind == :pseudorandom ? pseudorandom_U : kind == :halton ? halton_U : sobol_U

rows = NamedTuple[]
grad_store = Dict{Tuple{String,Symbol,Int}, Vector{Float64}}()

for (label, w, find_smallest) in POINTS
    n_scr = SCRAMBLE_PLAN[label]
    logprint("\n", "="^90); logprint("POINT: ", label, "  gp=", w[1], "  find_smallest=", find_smallest, "  n_scrambles=", n_scr)
    logprint("="^90)
    for kind in DRAWTYPES
        f = draw_fn(kind)
        for s in 1:n_scr
            seed = 1000 * s + (kind == :pseudorandom ? 1 : kind == :halton ? 2 : 3)
            t_draw0 = time()
            U = f(W_REAL, D; seed = seed)
            t_draw = time() - t_draw0

            t_ctx0 = time()
            ctx = d20_real_setup_qmc(W = W_REAL, U_injected = U, find_smallest = find_smallest)
            t_ctx = time() - t_ctx0

            xf = x_free_from_w(w, pe)
            t_eval0 = time()
            r = evaluate_fullA(xf, ctx; warm = false)
            t_eval = time() - t_eval0

            winner_zero_incidence = NaN; winner_min_share = NaN
            if r.inner_status in FEASIBLE_CODES
                winner, price_, gap_ = compute_winners(r.θ_full, ctx)
                # per-destination win COUNT by origin (D x D), zero-winner incidence = how many
                # (origin, destination) cells never won across all W draws.
                wcount = zeros(Int, D, D)
                for dcol in 1:D, wi in 1:W_REAL
                    wcount[winner[wi, dcol], dcol] += 1
                end
                winner_zero_incidence = count(==(0), wcount)
                winner_min_share = minimum(wcount) / W_REAL
            end

            kappa_here = r.inner_status in FEASIBLE_CODES ? kappa_of(r.gamma_focal_prime, σ) : NaN

            g_cos_stub = NaN
            if label in GRADIENT_POINTS && r.inner_status in FEASIBLE_CODES
                t_grad0 = time()
                gfull, meta = composite_gradient_at_fast(xf, ctx, pe; threaded = true, h_mode = :adaptive)
                t_grad = time() - t_grad0
                grad_store[(label, kind, s)] = gfull
            else
                t_grad = 0.0
            end

            push!(rows, (label = label, kind = String(kind), scramble = s, seed = seed,
                inner_status = r.inner_status, Delta_dual = r.Delta_dual, kappa = kappa_here,
                gravity_value = r.gravity_value, max_abs_moment_resid = r.max_abs_moment_resid,
                max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
                m_mean = r.m_mean, m_min = r.m_min, m_max = r.m_max,
                weight_norm_resid = r.weight_norm_resid,
                winner_zero_incidence = winner_zero_incidence, winner_min_share = winner_min_share,
                t_draw = t_draw, t_ctx = t_ctx, t_eval = t_eval, t_grad = t_grad))

            logprint("  [", label, "/", kind, "/scr", s, "] status=", r.inner_status, " Delta=", r.Delta_dual,
                     " kappa=", kappa_here, " zero_incidence=", winner_zero_incidence,
                     " t_draw=", round(t_draw,digits=3), " t_ctx=", round(t_ctx,digits=1), " t_eval=", round(t_eval,digits=2),
                     " t_grad=", round(t_grad,digits=2))
        end
    end
end

# ---- write raw rows to CSV ----
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
write_csv_rows(joinpath(OUTDIR, "fixed_point_comparison.csv"), rows)
logprint("\nWrote ", joinpath(OUTDIR, "fixed_point_comparison.csv"), " (", length(rows), " rows)")

# ---- cross-scramble cosine similarity of gradients, per (point, kind) ----
logprint("\n", "="^90); logprint("GRADIENT COSINE SIMILARITY (within-kind, across scrambles)"); logprint("="^90)
cos_rows = NamedTuple[]
for label in GRADIENT_POINTS, kind in DRAWTYPES
    gs = [grad_store[(label, kind, s)] for s in 1:SCRAMBLE_PLAN[label] if haskey(grad_store, (label, kind, s))]
    length(gs) < 2 && continue
    coss = Float64[]
    for i in 1:length(gs), j in i+1:length(gs)
        push!(coss, dot(gs[i], gs[j]) / (norm(gs[i]) * norm(gs[j])))
    end
    logprint("  ", label, "/", kind, ": n_pairs=", length(coss), " mean_cos=", round(mean(coss), digits=6),
             " min_cos=", round(minimum(coss), digits=6))
    push!(cos_rows, (label = label, kind = String(kind), n_pairs = length(coss), mean_cos = mean(coss), min_cos = minimum(coss)))
end
# ---- ALSO: cross-kind cosine (halton/sobol vs pseudorandom baseline) ----
for label in GRADIENT_POINTS
    base_gs = [grad_store[(label, :pseudorandom, s)] for s in 1:SCRAMBLE_PLAN[label] if haskey(grad_store, (label, :pseudorandom, s))]
    isempty(base_gs) && continue
    for kind in (:halton, :sobol)
        other_gs = [grad_store[(label, kind, s)] for s in 1:SCRAMBLE_PLAN[label] if haskey(grad_store, (label, kind, s))]
        isempty(other_gs) && continue
        coss = Float64[dot(bg, og) / (norm(bg) * norm(og)) for bg in base_gs for og in other_gs]
        logprint("  ", label, "/", kind, "-vs-pseudorandom: mean_cos=", round(mean(coss), digits=6), " min_cos=", round(minimum(coss), digits=6))
        push!(cos_rows, (label = label, kind = String(kind) * "_vs_pseudorandom", n_pairs = length(coss), mean_cos = mean(coss), min_cos = minimum(coss)))
    end
end
write_csv_rows(joinpath(OUTDIR, "gradient_cosine.csv"), cos_rows)

logprint("\nOUTDIR = ", OUTDIR)
close(LOGIO)
