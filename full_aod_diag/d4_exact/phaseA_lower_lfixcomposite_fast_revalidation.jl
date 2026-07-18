# ============================================================================
# Continuation 5, Priority 4: canonical revalidation of the new best-feasible
# LOWER incumbent (results/fullA_d4/9e03706/optfd_lower_lfix_composite_fast_sr1_20260718_095249/
# summary.txt, best_feasible_tracked row: kappa=0.005428799948779983). Mirrors
# phaseA_lfixcomposite_sr1_revalidation.jl's exact recipe, find_smallest=false
# (per test_stationarity_lower.jl's established convention for lower-direction checks).
# ============================================================================
include(joinpath(@__DIR__, "stationarity_check.jl"))
using Random, LinearAlgebra, Printf

function write_csv_rows(path::AbstractString, rows::Vector{<:NamedTuple})
    isempty(rows) && (open(path, "w") do io; println(io, "(no rows)"); end; return)
    cols = keys(rows[1])
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((r[c] for c in cols), ","))
        end
    end
end

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseA_lower_lfixcomposite_fast_revalidation")
mkpath(OUTDIR)

println("Running Priority 4 revalidation (lower_lfixcomposite_fast_sr1_300s) at commit ", COMMIT)

const W_CAND = [0.9967391744173478, 0.33826763364911505, 0.2756097423805949, 0.3168080759212972,
    0.28291467586724917, 1.124214996356822, 1.0586966677239436, 1.0353970705480537,
    1.0651065385723435, 0.7972795304932372, 0.7498744164437179, 0.797300832305142,
    0.7520482961532734, 1.464240420901755, 1.3955928233005424, 1.4031151818251653]

const W_OLD_STALLED = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916,
    -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819,
    0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236,
    0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

ctx_tight = d4_exact_setup(find_smallest = false)
ctx_loose = d4_exact_setup(find_smallest = false,
    inner_loop_opt = joinpath(@__DIR__, "ek_inner_loose.opt"))
pe = build_pivot_elimination(ctx_tight)
D = ctx_tight.D; D2 = D^2
w_lo = vcat(ctx_tight.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx_tight.bounds.γp_hi, fill(8.0, D2 - 1))

function x_free_from_w(w::AbstractVector)
    z = pivot_expand(w[2:end], pe)
    return vcat(w[1], vec(exp.(z)))
end

# ============================================================================
# 1. Fresh cold+warm recheck at 2 tolerances
# ============================================================================
println("="^78); println("STEP 1: fresh cold+warm recheck at 2 tolerances"); println("="^78)
recheck_rows = NamedTuple[]
for (label, w) in (("lower_lfixcomposite_fast_best_feasible", W_CAND), ("lower_stalled_old (for comparison)", W_OLD_STALLED))
    xf = x_free_from_w(w)
    for (tol_label, ctx) in (("tight_1e-12", ctx_tight), ("loose_1e-08", ctx_loose))
        for (warm_label, warm) in (("cold", false), ("warm", true))
            r = evaluate_fullA(xf, ctx; cache = nothing, warm = warm)
            κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
            push!(recheck_rows, (point = label, tol = tol_label, start = warm_label,
                kappa = κ, Delta_dual = r.Delta_dual, Delta_minus_delta = r.Delta_minus_delta,
                gravity_value = r.gravity_value, max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
                mean_m_resid = r.mean_m_resid, inner_status = r.inner_status, winner_hash = r.winner_hash))
            @printf("  [%s | %s | %s] kappa=%.10f Delta-delta=%.3e gravity=%.3e kkt=%.3e status=%d\n",
                label, tol_label, warm_label, κ, r.Delta_minus_delta, r.gravity_value, r.max_abs_moment_kkt_resid, r.inner_status)
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "step1_recheck.csv"), recheck_rows)
n_pass_step1 = count(r -> r.point == "lower_lfixcomposite_fast_best_feasible" && r.inner_status == 0 && isfinite(r.kappa) && r.Delta_minus_delta <= 1e-6, recheck_rows)
println("STEP 1 result: $(n_pass_step1)/4 cold/warm x tight/loose rechecks (new candidate only) are feasible")

# ============================================================================
# 2. External KKT check at h=0.01
# ============================================================================
println("\n" * "="^78); println("STEP 2: external KKT check, h=0.01"); println("="^78)
result_h01 = external_stationarity_check(W_CAND, ctx_tight, pe; find_smallest = false, h = 0.01, w_lo = w_lo, w_hi = w_hi)
println("Delta = ", result_h01.Delta, "  Delta-delta = ", result_h01.Delta_minus_delta)
println("eta = ", result_h01.eta, "  eta_nonneg = ", result_h01.eta_nonneg)
println("KKT residual (relative) = ", result_h01.residual_relative)
println("complementary slackness = ", result_h01.complementary_slackness)
println("active bounds = ", result_h01.n_active_bounds, "  nonfinite probes = ", result_h01.n_nonfinite_probes)
verified_h01 = result_h01.eta_nonneg && result_h01.residual_relative < 0.05 &&
    abs(result_h01.complementary_slackness) < 0.05 && result_h01.n_active_bounds == 0 &&
    result_h01.n_nonfinite_probes == 0
println("H_BANDWIDTH_KKT_CANDIDATE at h=0.01: ", verified_h01)

# ============================================================================
# 3. Multi-h gradient grid
# ============================================================================
println("\n" * "="^78); println("STEP 3: multi-h gradient grid"); println("="^78)
function Delta_of_w(ctx, w)
    r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = true)
    return r.Delta_dual, r.winner_hash, isfinite(r.Delta_dual)
end
const H_GRID = [0.02, 0.01, 0.005, 0.0025, 0.001]
w0 = W_CAND
n = length(w0)
Δ0, wh0, ok0 = Delta_of_w(ctx_tight, w0)
h_grid_results = NamedTuple[]
grads_by_h = Dict{Float64,Vector{Float64}}()
for h in H_GRID
    grad_c = zeros(n); n_switch = 0; n_nonfinite = 0
    for i in 1:n
        wp = copy(w0); wp[i] += h; wm = copy(w0); wm[i] -= h
        Δp, whp, okp = Delta_of_w(ctx_tight, wp)
        Δm, whm, okm = Delta_of_w(ctx_tight, wm)
        if okp && okm
            grad_c[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
        end
        (whp != wh0) && (n_switch += 1)
        (whm != wh0) && (n_switch += 1)
    end
    grads_by_h[h] = grad_c
    grad_f = zeros(n); grad_f[1] = -1.0   # find_smallest=false
    eta = -dot(grad_f, grad_c) / max(dot(grad_c, grad_c), 1e-300)
    resid = norm(grad_f .+ eta .* grad_c) / norm(grad_f)
    push!(h_grid_results, (h = h, n_winner_switches = n_switch, n_nonfinite = n_nonfinite,
        eta = eta, kkt_residual_relative = resid, norm_grad_central = norm(grad_c)))
    @printf("  h=%.5f: switches=%d nonfinite=%d eta=%.5f kkt_resid=%.5f |grad_c|=%.4f\n",
        h, n_switch, n_nonfinite, eta, resid, norm(grad_c))
end
write_csv_rows(joinpath(OUTDIR, "step3_h_grid.csv"), h_grid_results)

# ============================================================================
# 4. Deterministic poll for an exact-feasible improvement (LARGER w[1] is better here)
# ============================================================================
println("\n" * "="^78); println("STEP 4: deterministic poll"); println("="^78)
rng = MersenneTwister(20260718)
random_dirs = [normalize(randn(rng, n)) for _ in 1:20]
const POLL_RADII = [0.0001, 0.001, 0.005, 0.02]
poll_dirs = vcat([begin; e = zeros(n); e[i] = 1.0; e; end for i in 1:n], random_dirs)
poll_rows = NamedTuple[]
n_improvements = 0
for r in POLL_RADII, (di, d) in enumerate(poll_dirs), sgn in (1.0, -1.0)
    w_try = w0 .+ sgn .* r .* d
    Δ, wh, ok = Delta_of_w(ctx_tight, w_try)
    feasible = ok && Δ <= ctx_tight.δ + 1e-6
    improved = feasible && w_try[1] > w0[1] + 1e-12   # find_smallest=false: LARGER w[1] is better
    global n_improvements
    improved && (n_improvements += 1)
    push!(poll_rows, (radius = r, dir_idx = di, sign = sgn, gp = w_try[1], Delta = Δ, feasible = feasible, improved = improved))
end
write_csv_rows(joinpath(OUTDIR, "step4_poll.csv"), poll_rows)
println("poll: $(length(poll_rows)) probes, $(n_improvements) exact-feasible improvements found")

# ============================================================================
# Final classification
# ============================================================================
println("\n" * "="^78); println("PRIORITY 4 CLASSIFICATION SUMMARY (lower_lfixcomposite_fast_sr1_300s)"); println("="^78)
robust_local = verified_h01 && all(row.kkt_residual_relative < 0.10 for row in h_grid_results) &&
    all(row.n_nonfinite == 0 for row in h_grid_results) && n_improvements == 0
exact_feasible = n_pass_step1 == 4
println("EXACT_FEASIBLE_CANDIDATE: ", exact_feasible)
println("H_BANDWIDTH_KKT_CANDIDATE (h=0.01): ", verified_h01)
println("ROBUST_LOCAL_CANDIDATE: ", robust_local)
open(joinpath(OUTDIR, "phaseA_classification.txt"), "w") do io
    println(io, "point: lower_lfixcomposite_fast_sr1_300s, gamma_focal_prime=$(w0[1]), kappa=$(1 - w0[1]^(ctx_tight.σ/(ctx_tight.σ-1)))")
    println(io, "EXACT_FEASIBLE_CANDIDATE = ", exact_feasible)
    println(io, "H_BANDWIDTH_KKT_CANDIDATE(h=0.01) = ", verified_h01)
    println(io, "ROBUST_LOCAL_CANDIDATE = ", robust_local)
    println(io, "n_poll_improvements_found = ", n_improvements, " / ", length(poll_rows))
end
println("\nWrote all Priority 4 lower-revalidation artifacts to ", OUTDIR)
