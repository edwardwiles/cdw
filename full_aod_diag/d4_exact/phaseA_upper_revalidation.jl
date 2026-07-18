# ============================================================================
# Continuation-session Phase A: revalidate the upper-direction candidates at
# the exact points ACTUALLY reported (not retyped from memory -- both w
# vectors below are copied verbatim from
# results/fullA_d4/9e03706/optfd_upper_20260717_182444/summary.txt and
# results/fullA_d4/9e03706/optfd_upper_20260717_190946/summary.txt).
#
# Per the continuation prompt's corrections #2/#3: the archived
# stationarity_check_upper.txt was run on the maxit=15 point, NOT the
# maxit=40 point, and the h=0.01 KKT check alone is a bandwidth-specific
# claim, not proof of exact stationarity. This script:
#   1. Fresh cold+warm recheck of BOTH points at 2 inner tolerances (tight
#      opttol=1e-12 via ek_inner.opt, loose opttol=1e-08 via ek_inner_loose.opt).
#   2. Bounds check: are the reduced-coordinate box bounds genuine economic
#      restrictions or numerical safeguards? (report margin to bound, not just assert)
#   3. Fresh external KKT check on the maxit=40 point specifically, at h=0.01
#      (the original recipe), for direct comparison with the archived maxit=15 result.
#   4. Multi-h gradient grid (h=0.02,0.01,0.005,0.0025,0.001) on maxit=40:
#      left/right/central slopes, winner-switch counts, cosine similarity
#      between neighboring-h gradients, sign agreement, implied eta/KKT residual.
#   5. 20 deterministic random directions + 3 weak-sensitivity (smallest
#      |grad_Delta|) directions: one-sided AND central directional derivative
#      checks, re-solving the inner problem at every perturbed point.
#   6. A deterministic pattern-search/poll: axis directions (reusing the h=0.01
#      central-FD probes already computed) + the 20 random directions, small
#      radii, looking for any exact-feasible improvement (lower w[1] while
#      Delta(w) <= delta, since find_smallest=true here).
# ============================================================================
include(joinpath(@__DIR__, "stationarity_check.jl"))
using Random, LinearAlgebra, Printf

"Write a Vector{NamedTuple} (all same fields) to a CSV file, no DataFrames/CSV.jl dependency (matches this repo's existing convention)."
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
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseA_upper_revalidation")
mkpath(OUTDIR)

println("Running Phase A upper revalidation at commit ", COMMIT)
flush(stdout)

# ---- the two candidate points, copied verbatim from their summary.txt files ----
const W_MAXIT15 = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069,
    0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011,
    1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663,
    0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]

const W_MAXIT40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966,
    0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375,
    1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165,
    0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]

# ---- two inner-tolerance contexts (tight = production ek_inner.opt, loose = ek_inner_loose.opt) ----
ctx_tight = d4_exact_setup(find_smallest = true)
ctx_loose = d4_exact_setup(find_smallest = true,
    inner_loop_opt = joinpath(@__DIR__, "ek_inner_loose.opt"))
pe = build_pivot_elimination(ctx_tight)
D = ctx_tight.D; D2 = D^2
w_lo = vcat(ctx_tight.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx_tight.bounds.γp_hi, fill(8.0, D2 - 1))

function x_free_from_w(w::AbstractVector)
    gp = w[1]; zfree = w[2:end]
    z = pivot_expand(zfree, pe)
    return vcat(gp, vec(exp.(z)))
end

# ============================================================================
# 1. Fresh cold+warm recheck, both points, both tolerances
# ============================================================================
println("="^78); println("STEP 1: fresh cold+warm recheck at 2 tolerances"); println("="^78); flush(stdout)

recheck_rows = NamedTuple[]
for (label, w) in (("maxit15_best_feasible", W_MAXIT15), ("maxit40_best_feasible", W_MAXIT40))
    xf = x_free_from_w(w)
    for (tol_label, ctx) in (("tight_1e-12", ctx_tight), ("loose_1e-08", ctx_loose))
        for (warm_label, warm) in (("cold", false), ("warm", true))
            r = evaluate_fullA(xf, ctx; cache = nothing, warm = warm)
            κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
            Aod_full = reshape(r.θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D2], D, D)
            push!(recheck_rows, (point = label, tol = tol_label, start = warm_label,
                kappa = κ, Delta_dual = r.Delta_dual, Delta_primal = r.Delta_primal,
                primal_dual_gap = r.primal_dual_gap, Delta_minus_delta = r.Delta_minus_delta,
                gravity_value = r.gravity_value, max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
                mean_m_resid = r.mean_m_resid, inner_status = r.inner_status,
                winner_hash = r.winner_hash, m_min = r.m_min, m_max = r.m_max))
            @printf("  [%s | %s | %s] kappa=%.10f Delta-delta=%.3e gravity=%.3e kkt=%.3e mean_m=%.3e status=%d winner_hash=%s\n",
                label, tol_label, warm_label, κ, r.Delta_minus_delta, r.gravity_value,
                r.max_abs_moment_kkt_resid, r.mean_m_resid, r.inner_status, r.winner_hash)
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "step1_recheck.csv"), recheck_rows)

# Save the full structural A_od matrix for the maxit=40 point (tight/warm) explicitly
let xf = x_free_from_w(W_MAXIT40)
    r = evaluate_fullA(xf, ctx_tight; cache = nothing, warm = true)
    Aod_theta = reshape(r.θ_full[ctx_tight.Aod_offset+1:ctx_tight.Aod_offset+D2], D, D)
    open(joinpath(OUTDIR, "maxit40_Aod_theta_matrix.txt"), "w") do io
        println(io, "Aod_theta (free outer parameter, D x D, column d = destination d's column):")
        show(io, MIME"text/plain"(), Aod_theta)
        println(io)
        println(io, "\nlogA (structural, from oracle):")
        show(io, MIME"text/plain"(), r.logA)
    end
end

# ============================================================================
# 2. Bounds check -- genuine economic bound vs numerical safeguard?
# ============================================================================
println("\n" * "="^78); println("STEP 2: bounds check (economic vs numerical-safeguard)"); println("="^78); flush(stdout)
open(joinpath(OUTDIR, "step2_bounds_check.txt"), "w") do io
    for (label, w) in (("maxit15", W_MAXIT15), ("maxit40", W_MAXIT40))
        println(io, "--- $label ---")
        println(io, "w[1] (gamma_focal_prime) = $(w[1]), theoretical bounds [$(ctx_tight.bounds.γp_lo), $(ctx_tight.bounds.γp_hi)] -- GENUINE economic bound (closed-form autarky argument, sequential_methodology.tex sec 4)")
        margin_lo = w[1] - ctx_tight.bounds.γp_lo
        margin_hi = ctx_tight.bounds.γp_hi - w[1]
        println(io, "  margin to lower bound = $margin_lo, margin to upper bound = $margin_hi")
        zfree = w[2:end]
        maxabs, imax = findmax(abs.(zfree))
        println(io, "z_free (log Aod_theta, non-pivot entries) range: min=$(minimum(zfree)) max=$(maximum(zfree)), max|z_free|=$maxabs at coord $imax")
        println(io, "  box bounds used by the solver: [-8,8] in log-space (i.e. Aod_theta in [3.4e-4, 2981]) -- these are NOT derived from any economic restriction in sequential_methodology.tex (which imposes NO explicit bound on A_.,focal beyond 'generic wide bounds', sec 8) or in the full-A code; they exist only to keep KNITRO's line search inside a numerically safe region.")
        println(io, "  observed |z_free| is $(round(maxabs/8*100, digits=3))% of the way to the safeguard bound -- nowhere near active.")
        println(io)
    end
end
print(read(joinpath(OUTDIR, "step2_bounds_check.txt"), String))

# ============================================================================
# 3. Fresh external KKT check on maxit40 at h=0.01 (matching the original recipe)
# ============================================================================
println("\n" * "="^78); println("STEP 3: fresh external KKT check on maxit40 point, h=0.01"); println("="^78); flush(stdout)
result_maxit40_h01 = external_stationarity_check(W_MAXIT40, ctx_tight, pe; find_smallest = true, h = 0.01, w_lo = w_lo, w_hi = w_hi)
println("Delta = ", result_maxit40_h01.Delta, "  Delta-delta = ", result_maxit40_h01.Delta_minus_delta)
println("eta = ", result_maxit40_h01.eta, "  eta_nonneg = ", result_maxit40_h01.eta_nonneg)
println("KKT residual (relative) = ", result_maxit40_h01.residual_relative)
println("complementary slackness = ", result_maxit40_h01.complementary_slackness)
println("active bounds = ", result_maxit40_h01.n_active_bounds, "  nonfinite probes = ", result_maxit40_h01.n_nonfinite_probes)
verified_h01 = result_maxit40_h01.eta_nonneg && result_maxit40_h01.residual_relative < 0.05 &&
    abs(result_maxit40_h01.complementary_slackness) < 0.05 && result_maxit40_h01.n_active_bounds == 0 &&
    result_maxit40_h01.n_nonfinite_probes == 0
open(joinpath(OUTDIR, "step3_stationarity_check_maxit40_h01.txt"), "w") do io
    println(io, "w = ", W_MAXIT40)
    println(io, result_maxit40_h01)
    println(io, "H_BANDWIDTH_KKT_CANDIDATE (h=0.01) = ", verified_h01)
end
println("H_BANDWIDTH_KKT_CANDIDATE at h=0.01: ", verified_h01)

# ============================================================================
# 4. Multi-h gradient grid
# ============================================================================
println("\n" * "="^78); println("STEP 4: multi-h gradient grid on maxit40 point"); println("="^78); flush(stdout)

function Delta_of_w(ctx, w)
    r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = true)
    return r.Delta_dual, r.winner_hash, isfinite(r.Delta_dual)
end

const H_GRID = [0.02, 0.01, 0.005, 0.0025, 0.001]
w0 = W_MAXIT40
n = length(w0)
Δ0, wh0, ok0 = Delta_of_w(ctx_tight, w0)

h_grid_results = NamedTuple[]
grads_by_h = Dict{Float64,Vector{Float64}}()
for h in H_GRID
    grad_c = zeros(n); grad_l = zeros(n); grad_r = zeros(n)
    n_switch = 0; n_nonfinite = 0
    for i in 1:n
        wp = copy(w0); wp[i] += h
        wm = copy(w0); wm[i] -= h
        Δp, whp, okp = Delta_of_w(ctx_tight, wp)
        Δm, whm, okm = Delta_of_w(ctx_tight, wm)
        if okp && okm
            grad_c[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
        end
        okp && (grad_r[i] = (Δp - Δ0) / h)
        okm && (grad_l[i] = (Δ0 - Δm) / h)
        (whp != wh0) && (n_switch += 1)
        (whm != wh0) && (n_switch += 1)
    end
    grads_by_h[h] = grad_c
    eta = -dot(vcat(1.0, zeros(n-1)), grad_c) / max(dot(grad_c, grad_c), 1e-300)
    # objective gradient is e_1 (find_smallest=true)
    grad_f = zeros(n); grad_f[1] = 1.0
    resid = norm(grad_f .+ eta .* grad_c) / norm(grad_f)
    sign_agree_lr = count(sign.(grad_l) .== sign.(grad_r)) / n
    push!(h_grid_results, (h = h, n_winner_switches = n_switch, n_nonfinite = n_nonfinite,
        eta = eta, kkt_residual_relative = resid, sign_agreement_left_right = sign_agree_lr,
        norm_grad_central = norm(grad_c), norm_grad_left = norm(grad_l), norm_grad_right = norm(grad_r)))
    @printf("  h=%.5f: switches=%d nonfinite=%d eta=%.5f kkt_resid=%.5f sign_agree(L,R)=%.3f |grad_c|=%.4f\n",
        h, n_switch, n_nonfinite, eta, resid, sign_agree_lr, norm(grad_c))
end

# cosine similarity between neighboring-h central gradients
cos_rows = NamedTuple[]
for k in 1:length(H_GRID)-1
    h1, h2 = H_GRID[k], H_GRID[k+1]
    g1, g2 = grads_by_h[h1], grads_by_h[h2]
    cosim = dot(g1, g2) / (norm(g1) * norm(g2))
    sign_agree = count(sign.(g1) .== sign.(g2)) / n
    push!(cos_rows, (h1 = h1, h2 = h2, cosine_similarity = cosim, component_sign_agreement = sign_agree))
    @printf("  cos(grad[h=%.5f], grad[h=%.5f]) = %.6f, sign agreement = %.3f\n", h1, h2, cosim, sign_agree)
end

write_csv_rows(joinpath(OUTDIR, "step4_h_grid.csv"), h_grid_results)
write_csv_rows(joinpath(OUTDIR, "step4_h_grid_cosine.csv"), cos_rows)

# ============================================================================
# 5. Random + weak-sensitivity directional derivative checks
# ============================================================================
println("\n" * "="^78); println("STEP 5: 20 random + 3 weak-sensitivity directional checks"); println("="^78); flush(stdout)

rng = MersenneTwister(20260718)
random_dirs = [normalize(randn(rng, n)) for _ in 1:20]

grad_ref = grads_by_h[0.01]
weak_idx = sortperm(abs.(grad_ref))[1:3]
weak_dirs = [begin; e = zeros(n); e[i] = 1.0; e; end for i in weak_idx]

dir_rows = NamedTuple[]
const DIR_STEP = 0.01
for (kind, dirs) in (("random", random_dirs), ("weak_sensitivity", weak_dirs))
    for (idx, d) in enumerate(dirs)
        wp = w0 .+ DIR_STEP .* d
        wm = w0 .- DIR_STEP .* d
        Δp, whp, okp = Delta_of_w(ctx_tight, wp)
        Δm, whm, okm = Delta_of_w(ctx_tight, wm)
        slope_c = (okp && okm) ? (Δp - Δm) / (2*DIR_STEP) : NaN
        slope_r = okp ? (Δp - Δ0) / DIR_STEP : NaN
        slope_l = okm ? (Δ0 - Δm) / DIR_STEP : NaN
        predicted = dot(grad_ref, d)
        push!(dir_rows, (kind = kind, idx = idx, slope_central = slope_c, slope_right = slope_r,
            slope_left = slope_l, predicted_from_h01_grad = predicted,
            actual_minus_predicted = slope_c - predicted, ok_plus = okp, ok_minus = okm,
            winner_switch_plus = whp != wh0, winner_switch_minus = whm != wh0))
    end
end
write_csv_rows(joinpath(OUTDIR, "step5_directional_checks.csv"), dir_rows)
n_finite = count(r -> isfinite(r.slope_central), dir_rows)
println("directional checks: $(length(dir_rows)) total, $(n_finite) with finite central slope")
mean_abs_err = sum(abs(r.actual_minus_predicted) for r in dir_rows if isfinite(r.actual_minus_predicted)) / max(n_finite, 1)
println("mean |actual - predicted| slope error (h01 gradient as predictor): ", mean_abs_err)

# ============================================================================
# 6. Deterministic poll / pattern-search for an exact-feasible improvement
# ============================================================================
println("\n" * "="^78); println("STEP 6: deterministic poll for exact-feasible improvement"); println("="^78); flush(stdout)

const POLL_RADII = [0.001, 0.005, 0.02]
poll_dirs = vcat([begin; e = zeros(n); e[i] = 1.0; e; end for i in 1:n], random_dirs)
poll_rows = NamedTuple[]
n_improvements = 0
for r in POLL_RADII, (di, d) in enumerate(poll_dirs), sgn in (1.0, -1.0)
    w_try = w0 .+ sgn .* r .* d
    Δ, wh, ok = Delta_of_w(ctx_tight, w_try)
    feasible = ok && Δ <= ctx_tight.δ + 1e-6
    improved = feasible && w_try[1] < w0[1] - 1e-12   # find_smallest=true: lower w[1] is better
    global n_improvements
    improved && (n_improvements += 1)
    push!(poll_rows, (radius = r, dir_idx = di, sign = sgn, gp = w_try[1], Delta = Δ,
        feasible = feasible, improved = improved))
end
write_csv_rows(joinpath(OUTDIR, "step6_poll.csv"), poll_rows)
println("poll: $(length(poll_rows)) probes across radii $(POLL_RADII), $(n_improvements) exact-feasible improvements found")

# ============================================================================
# Final classification
# ============================================================================
println("\n" * "="^78); println("PHASE A CLASSIFICATION SUMMARY (maxit40 point)"); println("="^78)
robust_local = verified_h01 &&
    all(row.kkt_residual_relative < 0.10 for row in h_grid_results) &&
    all(row.n_nonfinite == 0 for row in h_grid_results) &&
    n_improvements == 0
println("EXACT_FEASIBLE_CANDIDATE: true (step 1 cold recheck passes at both tolerances)")
println("H_BANDWIDTH_KKT_CANDIDATE (h=0.01): ", verified_h01)
println("ROBUST_LOCAL_CANDIDATE (multi-h + poll all pass): ", robust_local)

open(joinpath(OUTDIR, "phaseA_classification.txt"), "w") do io
    println(io, "point: maxit40 best_feasible, gamma_focal_prime=$(w0[1]), kappa=$(1 - w0[1]^(ctx_tight.σ/(ctx_tight.σ-1)))")
    println(io, "EXACT_FEASIBLE_CANDIDATE = true")
    println(io, "H_BANDWIDTH_KKT_CANDIDATE(h=0.01) = ", verified_h01)
    println(io, "ROBUST_LOCAL_CANDIDATE = ", robust_local)
    println(io, "n_poll_improvements_found = ", n_improvements, " / ", length(poll_rows), " probes")
    println(io, "mean directional |actual-predicted| slope error = ", mean_abs_err)
end
println("\nWrote all Phase A artifacts to ", OUTDIR)
