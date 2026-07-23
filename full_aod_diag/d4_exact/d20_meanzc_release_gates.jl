# Real D=20/W=80,000/L=50 release gates for the CM+moments(+ZC) production integration
# (2026-07-23), per the release addendum's finite gate list (main prompt Section 5 + the
# multiple-K addendum Section 4.2). NOT a new research program -- exactly two real outer
# points, exactly two (K_mean,K_pair) configs (one per point), matching the addendum's own
# bounded battery:
#   Point A (benchmark/calibration start):        (K_mean,K_pair) = (1,1)
#   Point B (recovered cold-verified CM incumbent, delta=1, chain1): (K_mean,K_pair) = (2,2)
#
# At each point:
#   1. Fixed-point gate: cold solve (fresh process, no cache), record Delta_dual, typed status,
#      primal-dual gap, weighted moment/pair residuals, gravity residual, eta_nu/nu.
#   2. Nesting check at the SAME point/nu: Delta_CM <= Delta_CM+mean <= Delta_CM+mean+ZC.
#   3. C+ equality gate: build the SAME verified base state, evaluate full gradient under both
#      :reference and :cplus, compare economic block + every eta_nu_k coordinate.
#   4. One independently reoptimized central-FD check of the analytic eta_nu derivative (at
#      Point B only, K=2, one component -- "one check, not a bandwidth study" per the brief).
#
# Usage: julia --project=. full_aod_diag/d4_exact/d20_meanzc_release_gates.jl
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
using Printf, LinearAlgebra, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const L = 50
const DRAW_SEED = 20260719
const CONTRASTS = :orthonormal

lp(">>> [", now(), "] building D=20/W=", W, " real-data context (draw_seed=", DRAW_SEED, ")...")
t_ctx = @elapsed ctx = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED)
pe = build_pivot_elimination(ctx)
lp(">>> context built in ", round(t_ctx, digits = 1), "s. D=", ctx.D, " bi=", ctx.bi)

snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[L]

W_actual = size(ctx.U, 1)
pool = build_grad_workspace_pool(W_actual)
ws = build_lfix_factorized_workspace(ctx.D, W_actual)

x_free_calib = ctx.θ0_up[ctx.free_idx]
D = ctx.D
w_calib = vcat(x_free_calib[1], pivot_reduce(log.(reshape(x_free_calib[2:end], D, D)), pe))

# Point B: recovered cold-verified CM incumbent, delta=1, chain1 (production campaign 2026-07-22)
const SEED_PATH = "/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/cold_verified_seed.jls"
seedB = deserialize(SEED_PATH)
@assert seedB.W == W && seedB.draw_seed == DRAW_SEED && seedB.cm_L == L && seedB.contrasts == CONTRASTS "Point B provenance mismatch"
x_free_B = x_free_from_w(seedB.w, pe)

results = Dict{String,Any}()

function run_point(label::String, x_free0::Vector{Float64}, K_mean::Int, K_pair::Int, ν0::Vector{Float64})
    lp()
    lp("="^100)
    lp(">>> POINT ", label, "  (K_mean=", K_mean, ", K_pair=", K_pair, ")  nu0=", ν0)
    lp("="^100)

    # ---- 1. Fixed-point gate: CM-only baseline ----
    pcx_cm = build_cm_production_context(ctx, CS; L = L, contrasts = CONTRASTS, probs = probs)
    t_cm = @elapsed base_cm, verify_cm = archC_verified_state(x_free0, pcx_cm.ctx_cm, pcx_cm.cctx)
    Δ_cm = verify_cm.Delta_dual
    lp("  [CM-only]      Delta_dual=", Δ_cm, "  status=", classify_inner_result(verify_cm),
       "  gap=", verify_cm.primal_dual_gap, "  kkt=", verify_cm.max_abs_moment_kkt_resid, " (", round(t_cm, digits=1), "s)")

    # ---- 1. Fixed-point gate: CM+mean (K_mean levels, K_pair=0) ----
    aug_mean = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = 0, contrasts = CONTRASTS,
        meanzc_basis = :direct)
    ctx_mean = merge(ctx, (obj = aug_mean.obj_cm,))
    cctx_mean = build_cm_meanzc_bin_ctx(ctx, aug_mean)
    t_mean = @elapsed base_mean, verify_mean = archC_meanzc_verified_state(x_free0, ν0, ctx_mean, cctx_mean)
    Δ_mean = verify_mean.Delta_dual
    lp("  [CM+mean]      Delta_dual=", Δ_mean, "  status=", classify_inner_result(verify_mean),
       "  gap=", verify_mean.primal_dual_gap, "  kkt=", verify_mean.max_abs_moment_kkt_resid, " (", round(t_mean, digits=1), "s)")
    resid_mean = [recovered_mean_residuals(base_mean.m_star, aug_mean.Zraw_all[k], ν0[k]) for k in 1:K_mean]
    lp("     mean residuals (per level, should be ~0): ", [maximum(abs.(r)) for r in resid_mean])

    # ---- 1. Fixed-point gate: CM+mean+ZC (K_mean levels, K_pair levels) ----
    aug_zc = build_cm_meanzc_augmented_obj(ctx, CS; L = L, K_mean = K_mean, K_pair = K_pair, contrasts = CONTRASTS,
        meanzc_basis = :direct)
    ctx_zc = merge(ctx, (obj = aug_zc.obj_cm,))
    cctx_zc = build_cm_meanzc_bin_ctx(ctx, aug_zc)
    bins_zc = cm_bin_indices_for(ctx, aug_zc)
    t_zc = @elapsed base_zc, verify_zc = archC_meanzc_verified_state(x_free0, ν0, ctx_zc, cctx_zc)
    Δ_zc = verify_zc.Delta_dual
    lp("  [CM+mean+ZC]   Delta_dual=", Δ_zc, "  status=", classify_inner_result(verify_zc),
       "  gap=", verify_zc.primal_dual_gap, "  kkt=", verify_zc.max_abs_moment_kkt_resid, " (", round(t_zc, digits=1), "s)")
    resid_pair = [recovered_pair_residuals(base_zc.m_star, aug_zc.Zpairraw_all[k], ν0[k]) for k in 1:K_pair]
    lp("     pair residuals (per level, should be ~0): ", [maximum(abs.(r)) for r in resid_pair])
    gravity_resid = abs(sum(base_zc.m_star) / W_actual - 1.0)
    lp("     gravity/mean-weight residual |mean(m*)-1|=", gravity_resid)

    # ---- 2. Nesting check ----
    nest_tol = 1e-4
    nest1_ok = Δ_cm <= Δ_mean + nest_tol
    nest2_ok = Δ_mean <= Δ_zc + nest_tol
    lp("  NESTING: CM(", Δ_cm, ") <= CM+mean(", Δ_mean, ")? ", nest1_ok,
       "   CM+mean(", Δ_mean, ") <= CM+mean+ZC(", Δ_zc, ")? ", nest2_ok)

    # ---- 3. C+ equality gate on the CM+mean+ZC base state ----
    pcx_zc = (ctx_cm = ctx_zc, aug = aug_zc, cctx = cctx_zc, bins = bins_zc)
    t_gref = @elapsed g_ref, _ = cm_meanzc_production_gradient(x_free0, ν0, pcx_zc, ctx, pe; base = base_zc, verify = verify_zc,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    t_gcplus = @elapsed g_cplus, _ = cm_meanzc_production_gradient_cplus(x_free0, ν0, pcx_zc, ctx, pe, pool, ws; base = base_zc, verify = verify_zc,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    D2 = D^2
    econ_ref = g_ref[1:D2]; econ_cplus = g_cplus[1:D2]
    maxdiff = maximum(abs.(econ_ref .- econ_cplus))
    relmax = maximum(abs.(econ_ref .- econ_cplus) ./ max.(abs.(econ_ref), 1e-8))
    cossim = dot(econ_ref, econ_cplus) / (norm(econ_ref) * norm(econ_cplus))
    nsign = sum((econ_ref .> 1e-8) .& (econ_cplus .< -1e-8)) + sum((econ_ref .< -1e-8) .& (econ_cplus .> 1e-8))
    eta_ref = g_ref[D2+1:end]; eta_cplus = g_cplus[D2+1:end]
    eta_maxdiff = maximum(abs.(eta_ref .- eta_cplus))
    lp("  C+ EQUALITY: econ max|diff|=", maxdiff, " max relative=", relmax, " cosine=", cossim,
       " sign_mismatches=", nsign, "  (:reference ", round(t_gref, digits=1), "s, :cplus ", round(t_gcplus, digits=1), "s, speedup=", round(t_gref/t_gcplus, digits=1), "x)")
    lp("     eta_nu: reference=", eta_ref, " cplus=", eta_cplus, " max|diff|=", eta_maxdiff)

    results[label] = (Δ_cm = Δ_cm, Δ_mean = Δ_mean, Δ_zc = Δ_zc, nest1_ok = nest1_ok, nest2_ok = nest2_ok,
        econ_maxdiff = maxdiff, econ_relmax = relmax, cossim = cossim, nsign = nsign, eta_maxdiff = eta_maxdiff,
        eta_ref = eta_ref, eta_cplus = eta_cplus, gravity_resid = gravity_resid,
        gap_zc = verify_zc.primal_dual_gap, kkt_zc = verify_zc.max_abs_moment_kkt_resid,
        status_zc = classify_inner_result(verify_zc), t_gref = t_gref, t_gcplus = t_gcplus)

    return base_zc, verify_zc, pcx_zc, ν0
end

nu0_A = [Float64(factorial(k)) for k in 1:1]   # K=1: nu_1 = 1! = 1 (Exp(1) mean)
base_A, verify_A, pcx_A, ν0_A = run_point("A_calibration", w_calib, 1, 1, nu0_A)

nu0_B = [Float64(factorial(k)) for k in 1:2]   # K=2: nu_1=1, nu_2=2 (Exp(1) 2nd raw moment)
base_B, verify_B, pcx_B, ν0_B = run_point("B_cm_incumbent_delta1", x_free_B, 2, 2, nu0_B)

lp()
lp("="^100)
lp("One independently reoptimized central-FD check of the analytic eta_nu derivative (Point B, K=2, level 1)")
lp("="^100)
hη = 1e-5
η0_B = log.(ν0_B)
ηp = copy(η0_B); ηp[1] += hη
ηm = copy(η0_B); ηm[1] -= hη
_, _, vp = cm_meanzc_production_value_verified(x_free_B, exp.(ηp), pcx_B)
_, _, vm = cm_meanzc_production_value_verified(x_free_B, exp.(ηm), pcx_B)
fd_eta1 = (vp.Delta_dual - vm.Delta_dual) / (2hη)
analytic_eta1 = results["B_cm_incumbent_delta1"].eta_ref[1]
lp("  analytic d(Delta)/d(eta_nu_1)=", analytic_eta1, "  FD(reoptimized)=", fd_eta1, "  |diff|=", abs(analytic_eta1 - fd_eta1))

lp()
lp("="^100)
lp("Peak RSS")
lp("="^100)
peak_rss_kb = try
    parse(Int, split(read(`grep VmHWM /proc/self/status`, String))[2])
catch
    -1
end
lp("  VmHWM (peak RSS) = ", peak_rss_kb, " kB = ", round(peak_rss_kb / 1024 / 1024, digits = 2), " GB")

lp()
lp("="^100)
lp("SUMMARY")
lp("="^100)
for (label, r) in sort(collect(results); by = first)
    lp("  ", label, ": nest1_ok=", r.nest1_ok, " nest2_ok=", r.nest2_ok, " econ_max|diff|=", r.econ_maxdiff,
       " cosine=", r.cossim, " sign_mismatches=", r.nsign, " eta_max|diff|=", r.eta_maxdiff,
       " status=", r.status_zc, " gap=", r.gap_zc, " kkt=", r.kkt_zc)
end
lp(">>> D20_MEANZC_RELEASE_GATES_DONE")
