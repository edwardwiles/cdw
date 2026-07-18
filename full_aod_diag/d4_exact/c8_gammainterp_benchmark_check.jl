# ============================================================================
# Continuation 8, Workstream 1: benchmark-framing correction for the
# gamma_profile non-monotonicity report. Read-only diagnostic (no production
# code touched). Answers:
#   1. What is g_F (the calibration/factual gamma'_focal value), exactly, and
#      where does it come from in the code?
#   2. Delta(g_F, A*) at the raw calibration A (no reoptimization).
#      profile_Delta(g_F) = min_A Delta(g_F,A) via the SAME exact per-point
#      minimizer gamma_profile.jl already uses (reused, not re-derived).
#      Confirmatory re-check of the reported profile minimum near g*=0.96.
#   3. Does the raw calibration A satisfy the gravity moment R(A)=0 exactly,
#      or does the pivot-elimination projection (which the driver always
#      applies) have to move it, and by how much?
# ============================================================================
ENV["GP_RUN_GRID"] = "0"   # load gamma_profile.jl's functions/ctx/pe without running its own grid
include(joinpath(@__DIR__, "gamma_profile.jl"))
using Printf

# ZFREE_INCUMBENT is only defined inside gamma_profile.jl's `if RUN_GRID` block, which we
# disabled above -- copy verbatim (same literal used by gamma_profile.jl / gamma_profile_multistart.jl,
# = upper_lfixcomposite_sr1_60s's own A-block terminal point) so this script doesn't redefine it.
const ZFREE_INCUMBENT = [0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]

println("="^78)
println("STEP 1: g_F provenance")
println("="^78)

# Raw (unclamped) calibration gamma'_focal, straight from build_theta_gammanorm
# -- context.jl:34 calls this, then context.jl:36 clamps into [gamma_p_lo, gamma_p_hi]
# before storing into ctx.theta0_up. Recompute the UNCLAMPED value directly here
# to see whether clamping actually bites.
so, pp = build_ad_context()
D = so.D; bi = AD_PARAMS.baseIndex; σ = AD_PARAMS.σHat; μHat = pp.γ.μHat
θ0_up_unclamped = build_theta_gammanorm(pp.θ_initial_up, D, bi, μHat, σ)
gF_raw = θ0_up_unclamped[3+D]
gF_ctx = ctx.θ0_up[3+D]     # what's actually stored/used everywhere else (post-clamp)
@printf("  gF_raw (build_theta_gammanorm(...)[3+D], moments_gammanorm.jl:370, pre-clamp)      = %.15f\n", gF_raw)
@printf("  gF_ctx (ctx.theta0_up[3+D], context.jl:36 post-clamp, what the rest of code uses)   = %.15f\n", gF_ctx)
@printf("  bounds: gamma_p_lo=%.15f  gamma_p_hi=%.15f\n", ctx.bounds.γp_lo, ctx.bounds.γp_hi)
@printf("  clamp changed the value? %s (|diff|=%.3e)\n", gF_raw != gF_ctx, abs(gF_raw - gF_ctx))
const G_F = gF_ctx

println()
println("="^78)
println("STEP 3 (done before 2 -- need A* first): raw calibration A vs gravity-projected A*")
println("="^78)
Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
z0 = log.(Aod_theta0)
g_raw_gravity = gravity_from_logz(z0, ctx)
@printf("  gravity_value at RAW calibration Aod_theta0 (no projection)   = %.6e\n", g_raw_gravity)

ZFREE_CALIB = pivot_reduce(z0, pe)
z0_proj = pivot_expand(ZFREE_CALIB, pe)
g_proj_gravity = gravity_from_logz(z0_proj, ctx)
@printf("  gravity_value at PROJECTED calibration A* (pivot_expand(pivot_reduce(z0)))  = %.6e  (should be ~0 by exact-elimination construction)\n", g_proj_gravity)
pivot = pe.pivot_lin
@printf("  pivot coordinate (linear idx %d): raw z0=%.6f  projected z0=%.6f  |diff|=%.3e\n",
    pivot, vec(z0)[pivot], vec(z0_proj)[pivot], abs(vec(z0)[pivot]-vec(z0_proj)[pivot]))
relL2_A = norm(vec(exp.(z0_proj)) .- vec(Aod_theta0)) / norm(vec(Aod_theta0))
@printf("  relL2(A*_projected - A*_raw) = %.3e\n", relL2_A)

println()
println("="^78)
println("STEP 2: Delta(g_F, A*) raw point, profile_Delta(g_F) via solver, and g*~0.96 re-check")
println("="^78)

# (a) raw point, no reoptimization: g=g_F, A = projected calibration A* (this IS
# the A actually consumed downstream -- x_free_from_w always routes zfree through
# pivot_expand, so "the calibration A" in this pipeline means the projected one).
w_raw = vcat(G_F, ZFREE_CALIB)
xf_raw = x_free_from_w(w_raw)
r_raw = evaluate_fullA(xf_raw, ctx; cache = nothing, warm = true)
@printf("  (a) Delta(g_F, A*_raw-calib-projected), NO reoptimization:\n")
@printf("      Delta_dual = %.6e   inner_status=%d   gravity_value=%.3e   max_abs_moment_kkt_resid=%.3e\n",
    r_raw.Delta_dual, r_raw.inner_status, r_raw.gravity_value, r_raw.max_abs_moment_kkt_resid)

# (b) profile_Delta(g_F) = min_A Delta(g_F, A): reuse gamma_profile.jl's exact
# per-point KNITRO A-block minimizer, small multistart (3 starts, not the full 9)
# just to make sure we're not reporting a single-path artifact at this one point.
starts_gF = [("calib", copy(ZFREE_CALIB)), ("incumbent", copy(ZFREE_INCUMBENT))]
best_gF = Inf; best_kind = ""
for (kind, zf0) in starts_gF
    res = profile_delta_at_gamma(G_F, zf0, ctx, pe; maxtime_real = 30.0, hessopt_tag = "sr1")
    @printf("  (b) [%-9s start] status=%d n_eval=%3d wall=%5.1fs best_Delta=%.6e gravity=%.3e kkt=%.3e\n",
        kind, res.knitro_status, res.n_eval, res.wall, res.best_Delta, res.best_gravity, res.best_kkt)
    if isfinite(res.best_Delta) && res.best_Delta < best_gF
        global best_gF = res.best_Delta; global best_kind = kind
    end
end
@printf("  --> profile_Delta(g_F) = min_A Delta(g_F,A) = %.6e  (best start: %s)\n", best_gF, best_kind)

# (c) confirmatory single run at g=0.96 (previously reported multistart-min = 9.354e-05)
println()
res96 = profile_delta_at_gamma(0.96, copy(ZFREE_INCUMBENT), ctx, pe; maxtime_real = 30.0, hessopt_tag = "sr1")
@printf("  (c) g=0.96 confirmatory run [incumbent start]: status=%d n_eval=%3d wall=%5.1fs best_Delta=%.6e gravity=%.3e kkt=%.3e\n",
    res96.knitro_status, res96.n_eval, res96.wall, res96.best_Delta, res96.best_gravity, res96.best_kkt)
@printf("  (previously reported multistart-min at g=0.9600 was 9.354e-05)\n")

println()
println("="^78)
println("SUMMARY")
println("="^78)
@printf("  g_F                                  = %.10f\n", G_F)
@printf("  Delta(g_F, A*) raw (no reopt)         = %.6e\n", r_raw.Delta_dual)
@printf("  profile_Delta(g_F) = min_A Delta       = %.6e\n", best_gF)
@printf("  profile_Delta(0.96) [reconfirm]        = %.6e\n", res96.best_Delta)
@printf("  gravity residual at raw calib A        = %.6e\n", g_raw_gravity)
@printf("  gravity residual at projected calib A* = %.6e\n", g_proj_gravity)
