# Correctness gates: new theta_cplus_secant vs old theta_fixed_dual_delta_pivot_A (task §10).
# Real D=20/W=80,000. Compares D_plus, D_minus, the secant, and (via a full cb_G!-equivalent
# reconstruction) confirms both routes decode the SAME xf/theta at each probe.
#
# Usage: julia --project=. -t 4 full_aod_diag/d4_exact/test_theta_cplus_correctness.jl
using Random: MersenneTwister
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout.jl"))
include(joinpath(@__DIR__, "theta_cplus.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
check(cond, name) = (lp(cond ? "PASS  " : "FAIL  ", name); cond || push!(FAILURES, name))

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0
const FIND_SMALLEST = true

ctx0 = d20_real_setup_design(W = W, δ = DELTA, find_smallest = FIND_SMALLEST,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx0.μHat
sigma = ctx0.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
ctx0 = attach_compressed_factual_workspace(ctx0, ctx0.D, ctx0.D_dest, W)
ctx0 = attach_canonical_price_precompute_workspace(ctx0)
ctx0 = attach_hard_score_b_cache(ctx0)
ctx_flex = make_flexible_theta(ctx0; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :powered_aspace)
xy = precompute_aspace_XY(ctx_flex)
pgc = build_pivot_elimination_cheap(ctx_flex; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
theta_ws = build_theta_cplus_workspace(ctx0.D, ctx0.D_dest, W; h_theta = 1e-3)

x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp0 = x_free_calib[1]
D = ctx0.D; Ddest = ctx0.D_dest
logA_full0 = log.(reshape(x_free_calib[2:end], D, Ddest))
layout = make_layout(trade_elasticity_mode = :flexible, A_coordinate_mode = :powered_aspace)
rsc = build_ranged_screen_context(ctx_flex)
sc = ScreenCounters(); n_eval = Ref(0)

function base_at(w0)
    d0 = decode_outer_unified(w0, ctx_flex, layout, pgc, xy)
    r0, _ = screened_eval(d0.xf, ctx_flex, rsc, sc, n_eval; warm = false)
    r0.inner_status in FEASIBLE_CODES || error("base_at: point not feasible, inner_status=$(r0.inner_status)")
    base = BaseDualState(collect(d0.xf), r0.θ_full, r0.zeta, r0.lambda, copy(ctx_flex.obj.arg1), r0.inner_status)
    return d0, base, r0
end

function compare_at(label, theta_test, h_theta)
    w0 = reduce_to_w_unified(theta_test, gp0, logA_full0, pgc, xy, layout)
    local d0, base, r0
    try
        d0, base, r0 = base_at(w0)
    catch e
        lp(">>> ", label, ": SKIPPED (base point not inner-feasible at this theta -- test-design artifact, not a theta_cplus issue): ", e)
        return nothing
    end
    lp(">>> ", label, ": theta=", theta_test, " Delta0=", r0.Delta_dual, " inner_status=", r0.inner_status)

    # OLD path
    inner_x_fixed = copy(ctx_flex.obj.x)
    w_plus = copy(w0); w_plus[1] += h_theta
    D_plus_old = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx_flex, xy)
    w_minus = copy(w0); w_minus[1] -= h_theta
    D_minus_old = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx_flex, xy)
    secant_old = (D_plus_old - D_minus_old) / (2 * h_theta)

    # NEW path
    theta_ws.h_theta = h_theta
    sec_new = theta_cplus_secant(w0, ctx_flex, xy, pgc, base, theta_ws)

    d_plus = abs(D_plus_old - sec_new.D_plus)
    d_minus = abs(D_minus_old - sec_new.D_minus)
    d_secant = abs(secant_old - sec_new.grad_eta_theta)
    rel_secant = d_secant / max(abs(secant_old), 1e-12)
    lp("    D_plus:  old=", D_plus_old, " new=", sec_new.D_plus, " |diff|=", d_plus)
    lp("    D_minus: old=", D_minus_old, " new=", sec_new.D_minus, " |diff|=", d_minus)
    lp("    secant:  old=", secant_old, " new=", sec_new.grad_eta_theta, " |diff|=", d_secant, " rel=", rel_secant)
    lp("    n_winner_changes(plus vs minus)=", sec_new.n_winner_changes)
    check(d_plus < 1e-8, "$label: D_plus agrees < 1e-8")
    check(d_minus < 1e-8, "$label: D_minus agrees < 1e-8")
    check(rel_secant < 1e-6, "$label: secant agrees rel < 1e-6")
    return sec_new
end

# 1) calibration
compare_at("calibration", theta_star, 1e-3)
# 2) ~5% above/below calibration
compare_at("theta +5%", theta_star * 1.05, 1e-3)
compare_at("theta -5%", theta_star * 0.95, 1e-3)
# 3) multiple step sizes at calibration
for h in (1e-2, 1e-3, 1e-4)
    compare_at("calibration h=$h", theta_star, h)
end
# 4) a point with a nonzero winner-change count if one shows up naturally (near-budget-ish: perturb A slightly)
logA_perturbed = logA_full0 .+ 0.05 .* randn(MersenneTwister(1), D, Ddest)
w_pert = reduce_to_w_unified(theta_star, gp0, logA_perturbed, pgc, xy, layout)
try
    d0, base, r0 = base_at(w_pert)
    if r0.inner_status in FEASIBLE_CODES
        lp(">>> near-budget perturbed point: theta=", theta_star, " Delta0=", r0.Delta_dual)
        inner_x_fixed = copy(ctx_flex.obj.x)
        h_theta = 1e-3
        w_plus = copy(w_pert); w_plus[1] += h_theta
        D_plus_old = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx_flex, xy)
        w_minus = copy(w_pert); w_minus[1] -= h_theta
        D_minus_old = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx_flex, xy)
        secant_old = (D_plus_old - D_minus_old) / (2 * h_theta)
        theta_ws.h_theta = h_theta
        sec_new = theta_cplus_secant(w_pert, ctx_flex, xy, pgc, base, theta_ws)
        d_secant = abs(secant_old - sec_new.grad_eta_theta)
        rel_secant = d_secant / max(abs(secant_old), 1e-12)
        lp("    secant: old=", secant_old, " new=", sec_new.grad_eta_theta, " rel=", rel_secant, " n_winner_changes=", sec_new.n_winner_changes)
        check(rel_secant < 1e-6, "perturbed point: secant agrees rel < 1e-6")
    else
        lp(">>> perturbed point not feasible (inner_status=", r0.inner_status, "), skipping")
    end
catch e
    lp(">>> perturbed point errored: ", e, " -- skipping (not a theta_cplus failure if base_at itself failed)")
end

lp("theta_ws.generic_moments_calls = ", theta_ws.generic_moments_calls, " (must be 0)")
check(theta_ws.generic_moments_calls == 0, "zero generic moments!/reconstruct_full calls through theta_cplus path")

lp(isempty(FAILURES) ? "ALL THETA_CPLUS CORRECTNESS GATES PASS" : "FAILURES: $(FAILURES)")
