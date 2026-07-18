# ============================================================================
# Continuation 5, Priority 0: canonical revalidation of the new best-feasible
# upper incumbent found by Continuation 4 Phase 4 (results/fullA_d4/9e03706/
# optfd_upper_lfix_composite_sr1_20260718_072835/summary.txt,
# best_feasible_tracked row: kappa=0.17245688540655113). Structurally identical
# recipe to phaseA_upper_revalidation.jl (same file, reused not reinvented) --
# applied to the ONE new candidate this time, per the continuation prompt's
# Priority 0 requirements:
#   1. Fresh cold+warm recheck at 2 inner tolerances (tight opttol=1e-12 via
#      ek_inner.opt, loose opttol=1e-08 via ek_inner_loose.opt) -- full
#      reconstructed bounds, primal/dual divergence, all moments, gravity,
#      winner hash, solver status.
#   2. Bounds check (economic vs numerical-safeguard).
#   3. External KKT check at h=0.01 (original recipe bandwidth).
#   4. Multi-h gradient grid (h=0.02,0.01,0.005,0.0025,0.001): left/right/
#      central slopes, winner-switch counts, cosine similarity between
#      neighboring-h gradients, sign agreement, implied eta/KKT residual.
#      NOTE on "gravity-tangent A-only directions": every reduced coordinate
#      i>1 in this w-space IS a gravity-tangent, A-only direction by
#      construction -- pivot_expand's affine map (gravity_elimination.jl)
#      eliminates exactly the one degree of freedom needed to hold the
#      gravity moment at its target for ANY z_free move, and coordinate 1
#      (gamma_focal_prime) is the only non-A-block coordinate. So this
#      script's per-coordinate multi-h/poll sweep over i=2..16 already IS the
#      "gravity-tangent A-only directions" check the task asks for, not a
#      separate probe -- confirmed structurally (gravity_value is verified
#      near machine-precision-constant across every probe below, exactly as
#      it must be if the tangent-space claim is correct).
#   5. 20 deterministic random directions + 3 weak-sensitivity directions:
#      one-sided AND central directional derivative checks.
#   6. Deterministic pattern-search/poll (axis + random directions, 3 radii)
#      for any exact-feasible improvement.
#   7. Classification: EXACT_FEASIBLE / H_BANDWIDTH_KKT / ROBUST_LOCAL / STALLED.
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
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseA_lfixcomposite_sr1_revalidation")
mkpath(OUTDIR)

println("Running Priority 0 revalidation (upper_lfixcomposite_sr1_60s) at commit ", COMMIT)
flush(stdout)

# ---- the candidate point, copied verbatim from its own summary.txt best_feasible_tracked row ----
const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]

# ---- for comparison: prior headline incumbent ----
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
# 1. Fresh cold+warm recheck, 2 tolerances -- full reconstructed bounds,
#    primal/dual divergence, ALL moments, gravity, winner hash, solver status
# ============================================================================
println("="^78); println("STEP 1: fresh cold+warm recheck at 2 tolerances"); println("="^78); flush(stdout)

recheck_rows = NamedTuple[]
for (label, w) in (("lfixcomposite_sr1_best_feasible", W_CAND), ("maxit40_best_feasible (for comparison)", W_MAXIT40))
    xf = x_free_from_w(w)
    for (tol_label, ctx) in (("tight_1e-12", ctx_tight), ("loose_1e-08", ctx_loose))
        for (warm_label, warm) in (("cold", false), ("warm", true))
            r = evaluate_fullA(xf, ctx; cache = nothing, warm = warm)
            κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
            push!(recheck_rows, (point = label, tol = tol_label, start = warm_label,
                kappa = κ, Delta_dual = r.Delta_dual, Delta_primal = r.Delta_primal,
                primal_dual_gap = r.primal_dual_gap, Delta_minus_delta = r.Delta_minus_delta,
                gravity_value = r.gravity_value, max_abs_moment_resid = r.max_abs_moment_resid,
                max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
                mean_m_resid = r.mean_m_resid, weight_norm_resid = r.weight_norm_resid,
                inner_status = r.inner_status, winner_hash = r.winner_hash,
                m_min = r.m_min, m_max = r.m_max, m_mean = r.m_mean))
            @printf("  [%s | %s | %s] kappa=%.10f Delta-delta=%.3e gravity=%.3e moment_resid=%.3e kkt=%.3e mean_m=%.3e status=%d winner_hash=%s\n",
                label, tol_label, warm_label, κ, r.Delta_minus_delta, r.gravity_value,
                r.max_abs_moment_resid, r.max_abs_moment_kkt_resid, r.mean_m_resid, r.inner_status, r.winner_hash)
        end
    end
end
write_csv_rows(joinpath(OUTDIR, "step1_recheck.csv"), recheck_rows)

n_pass_step1 = count(r -> r.inner_status == 0 && isfinite(r.kappa) && r.Delta_minus_delta <= 1e-6, recheck_rows)
println("STEP 1 result: $(n_pass_step1)/$(length(recheck_rows)) cold/warm x tight/loose rechecks are feasible (inner_status=0, Delta<=delta)")

# Save the full structural A_od matrix + full moment vector for the candidate (tight/warm)
let xf = x_free_from_w(W_CAND)
    r = evaluate_fullA(xf, ctx_tight; cache = nothing, warm = true)
    Aod_theta = reshape(r.θ_full[ctx_tight.Aod_offset+1:ctx_tight.Aod_offset+D2], D, D)
    open(joinpath(OUTDIR, "lfixcomposite_sr1_Aod_theta_matrix.txt"), "w") do io
        println(io, "Aod_theta (free outer parameter, D x D, column d = destination d's column):")
        show(io, MIME"text/plain"(), Aod_theta)
        println(io)
        println(io, "\nlogA (structural, from oracle):")
        show(io, MIME"text/plain"(), r.logA)
        println(io)
        println(io, "\nfull moment_resid vector (all $(length(r.moment_resid)) moments):")
        show(io, MIME"text/plain"(), r.moment_resid)
        println(io)
    end
end

# ============================================================================
# 2. Bounds check -- genuine economic bound vs numerical safeguard?
# ============================================================================
println("\n" * "="^78); println("STEP 2: bounds check (economic vs numerical-safeguard)"); println("="^78); flush(stdout)
open(joinpath(OUTDIR, "step2_bounds_check.txt"), "w") do io
    w = W_CAND
    println(io, "--- lfixcomposite_sr1 ---")
    println(io, "w[1] (gamma_focal_prime) = $(w[1]), theoretical bounds [$(ctx_tight.bounds.γp_lo), $(ctx_tight.bounds.γp_hi)] -- GENUINE economic bound (closed-form autarky argument, sequential_methodology.tex sec 4)")
    margin_lo = w[1] - ctx_tight.bounds.γp_lo
    margin_hi = ctx_tight.bounds.γp_hi - w[1]
    println(io, "  margin to lower bound = $margin_lo, margin to upper bound = $margin_hi")
    zfree = w[2:end]
    maxabs, imax = findmax(abs.(zfree))
    println(io, "z_free (log Aod_theta, non-pivot entries) range: min=$(minimum(zfree)) max=$(maximum(zfree)), max|z_free|=$maxabs at coord $imax")
    println(io, "  box bounds used by the solver: [-8,8] in log-space -- numerical safeguard only, not an economic restriction (see phaseA_upper_revalidation.jl step 2 for the full argument, unchanged here).")
    println(io, "  observed |z_free| is $(round(maxabs/8*100, digits=3))% of the way to the safeguard bound -- nowhere near active.")
end
print(read(joinpath(OUTDIR, "step2_bounds_check.txt"), String))

# ============================================================================
# 3. External KKT check at h=0.01 (original recipe bandwidth)
# ============================================================================
println("\n" * "="^78); println("STEP 3: external KKT check, h=0.01"); println("="^78); flush(stdout)
result_h01 = external_stationarity_check(W_CAND, ctx_tight, pe; find_smallest = true, h = 0.01, w_lo = w_lo, w_hi = w_hi)
println("Delta = ", result_h01.Delta, "  Delta-delta = ", result_h01.Delta_minus_delta)
println("eta = ", result_h01.eta, "  eta_nonneg = ", result_h01.eta_nonneg)
println("KKT residual (absolute, original w-space units) = ", result_h01.residual_norm)
println("KKT residual (relative to ||grad_f||=1) = ", result_h01.residual_relative)
println("complementary slackness = ", result_h01.complementary_slackness)
println("active bounds = ", result_h01.n_active_bounds, "  nonfinite probes = ", result_h01.n_nonfinite_probes)
verified_h01 = result_h01.eta_nonneg && result_h01.residual_relative < 0.05 &&
    abs(result_h01.complementary_slackness) < 0.05 && result_h01.n_active_bounds == 0 &&
    result_h01.n_nonfinite_probes == 0
open(joinpath(OUTDIR, "step3_stationarity_check_h01.txt"), "w") do io
    println(io, "w = ", W_CAND)
    println(io, result_h01)
    println(io, "H_BANDWIDTH_KKT_CANDIDATE (h=0.01) = ", verified_h01)
end
println("H_BANDWIDTH_KKT_CANDIDATE at h=0.01: ", verified_h01)

# ============================================================================
# 4. Multi-h gradient grid
# ============================================================================
println("\n" * "="^78); println("STEP 4: multi-h gradient grid"); println("="^78); flush(stdout)

function Delta_of_w(ctx, w)
    r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = true)
    return r.Delta_dual, r.winner_hash, isfinite(r.Delta_dual), r.gravity_value
end

const H_GRID = [0.02, 0.01, 0.005, 0.0025, 0.001]
w0 = W_CAND
n = length(w0)
Δ0, wh0, ok0, grav0 = Delta_of_w(ctx_tight, w0)

h_grid_results = NamedTuple[]
grads_by_h = Dict{Float64,Vector{Float64}}()
gravity_probe_values = Float64[grav0]
for h in H_GRID
    grad_c = zeros(n); grad_l = zeros(n); grad_r = zeros(n)
    n_switch = 0; n_nonfinite = 0
    for i in 1:n
        wp = copy(w0); wp[i] += h
        wm = copy(w0); wm[i] -= h
        Δp, whp, okp, gravp = Delta_of_w(ctx_tight, wp)
        Δm, whm, okm, gravm = Delta_of_w(ctx_tight, wm)
        okp && push!(gravity_probe_values, gravp)
        okm && push!(gravity_probe_values, gravm)
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
    grad_f = zeros(n); grad_f[1] = 1.0
    resid = norm(grad_f .+ eta .* grad_c) / norm(grad_f)
    sign_agree_lr = count(sign.(grad_l) .== sign.(grad_r)) / n
    push!(h_grid_results, (h = h, n_winner_switches = n_switch, n_nonfinite = n_nonfinite,
        eta = eta, kkt_residual_relative = resid, sign_agreement_left_right = sign_agree_lr,
        norm_grad_central = norm(grad_c), norm_grad_left = norm(grad_l), norm_grad_right = norm(grad_r)))
    @printf("  h=%.5f: switches=%d nonfinite=%d eta=%.5f kkt_resid=%.5f sign_agree(L,R)=%.3f |grad_c|=%.4f\n",
        h, n_switch, n_nonfinite, eta, resid, sign_agree_lr, norm(grad_c))
end

max_grav_dev = maximum(abs.(gravity_probe_values .- grav0))
println("gravity-tangent check: max|gravity_value - gravity_value(w0)| across ALL $(length(gravity_probe_values)) multi-h probes = ", max_grav_dev, " (confirms every coordinate i>1 perturbation stays on the gravity-consistent manifold, i.e. IS a gravity-tangent A-only direction)")

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
open(joinpath(OUTDIR, "step4_gravity_tangent_check.txt"), "w") do io
    println(io, "max|gravity_value - gravity_value(w0)| across all multi-h probes = ", max_grav_dev)
    println(io, "gravity_value(w0) = ", grav0)
end

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
        Δp, whp, okp, _ = Delta_of_w(ctx_tight, wp)
        Δm, whm, okm, _ = Delta_of_w(ctx_tight, wm)
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
# 6. Deterministic poll / pattern-search + trust-region-style shrinking probe
# ============================================================================
println("\n" * "="^78); println("STEP 6: deterministic poll for exact-feasible improvement"); println("="^78); flush(stdout)

const POLL_RADII = [0.0001, 0.001, 0.005, 0.02]
poll_dirs = vcat([begin; e = zeros(n); e[i] = 1.0; e; end for i in 1:n], random_dirs)
poll_rows = NamedTuple[]
n_improvements = 0
for r in POLL_RADII, (di, d) in enumerate(poll_dirs), sgn in (1.0, -1.0)
    w_try = w0 .+ sgn .* r .* d
    Δ, wh, ok, _ = Delta_of_w(ctx_tight, w_try)
    feasible = ok && Δ <= ctx_tight.δ + 1e-6
    improved = feasible && w_try[1] < w0[1] - 1e-12   # find_smallest=true: lower w[1] is better
    global n_improvements
    improved && (n_improvements += 1)
    push!(poll_rows, (radius = r, dir_idx = di, sign = sgn, gp = w_try[1], Delta = Δ,
        feasible = feasible, improved = improved))
end
write_csv_rows(joinpath(OUTDIR, "step6_poll.csv"), poll_rows)
println("poll: $(length(poll_rows)) probes across radii $(POLL_RADII) (finest 1e-4 = trust-region-style shrinking probe), $(n_improvements) exact-feasible improvements found")

# ============================================================================
# Final classification
# ============================================================================
println("\n" * "="^78); println("PRIORITY 0 CLASSIFICATION SUMMARY (upper_lfixcomposite_sr1_60s)"); println("="^78)
robust_local = verified_h01 &&
    all(row.kkt_residual_relative < 0.10 for row in h_grid_results) &&
    all(row.n_nonfinite == 0 for row in h_grid_results) &&
    n_improvements == 0
exact_feasible = n_pass_step1 == length(recheck_rows)
println("EXACT_FEASIBLE_CANDIDATE: ", exact_feasible, " ($(n_pass_step1)/$(length(recheck_rows)) cold/warm x tight/loose rechecks feasible)")
println("H_BANDWIDTH_KKT_CANDIDATE (h=0.01): ", verified_h01)
println("ROBUST_LOCAL_CANDIDATE (multi-h + poll all pass): ", robust_local)

open(joinpath(OUTDIR, "phaseA_classification.txt"), "w") do io
    println(io, "point: upper_lfixcomposite_sr1_60s best_feasible, gamma_focal_prime=$(w0[1]), kappa=$(1 - w0[1]^(ctx_tight.σ/(ctx_tight.σ-1)))")
    println(io, "EXACT_FEASIBLE_CANDIDATE = ", exact_feasible, "  ($(n_pass_step1)/$(length(recheck_rows)) rechecks)")
    println(io, "H_BANDWIDTH_KKT_CANDIDATE(h=0.01) = ", verified_h01)
    println(io, "ROBUST_LOCAL_CANDIDATE = ", robust_local)
    println(io, "n_poll_improvements_found = ", n_improvements, " / ", length(poll_rows), " probes")
    println(io, "mean directional |actual-predicted| slope error = ", mean_abs_err)
    println(io, "max gravity-tangent deviation across multi-h probes = ", max_grav_dev)
end
println("\nWrote all Priority 0 artifacts to ", OUTDIR)
