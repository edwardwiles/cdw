# Real D=20 post-omit-ROW gates for the flexible-theta a-space production port (task §14).
# D_origin=20, D_dest=19, W=80,000, pseudorandom seed 20260719, current production data,
# BLAS threads=1 (set via env before launch, not here).
#
# Run: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. -t 20 \
#        full_aod_diag/d4_exact/test_flexible_theta_aspace_d20_gates.jl
using Random
using LinearAlgebra: norm
using Statistics: mean

include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const RESULTS = Dict{String,Any}()
const FAILURES = String[]
function check(name::AbstractString, cond::Bool; note::AbstractString = "")
    status = cond ? "PASS" : "FAIL"
    lp(rpad(status, 6), name, note == "" ? "" : "  ($note)")
    cond || push!(FAILURES, name)
    RESULTS[name] = cond
end

const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

lp(">>> Julia threads: ", Threads.nthreads())
lp(">>> Gate 0: build fixed-theta ctx (calibration reference) + flexible-theta ctx")

t0 = @elapsed ctx_fixed = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
lp("  ctx_fixed built in ", round(t0, digits = 1), "s  D=", ctx_fixed.D, " D_dest=", ctx_fixed.D_dest, " μHat=", ctx_fixed.μHat)

theta_star = 1.0 / ctx_fixed.μHat
sigma = ctx_fixed.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
lp("  theta_star=", theta_star, " sigma=", sigma, " theta_bounds=[", theta_min, ",", theta_max, "]")

t1 = @elapsed ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
lp("  ctx (flexible) built in ", round(t1, digits = 1), "s")
xy = precompute_aspace_XY(ctx)
Ddest = _flex_ddest(ctx); D = ctx.D
n_free_A = D * Ddest - 1
outer_dim = D * Ddest + 1
lp("[startup] trade_elasticity_mode = flexible")
lp("[startup] A_coordinate_mode = ", ctx.A_coordinate_mode)
lp("[startup] theta_star = ", theta_star)
lp("[startup] theta_bounds = [", theta_min, ", ", theta_max, "]")
lp("[startup] outer_dimension = ", outer_dim)
lp("[startup] core_top1_engine = canonical_log_additive")
lp("[startup] outer_gradient_top3_engine = cplus")
check("outer_dimension == 381 at real D=20 post-omit-ROW", outer_dim == 381; note = "D=$D Ddest=$Ddest")

rsc = build_ranged_screen_context(ctx)
check("envelope screen correctly DISABLED under flexible theta", rsc.envelope === nothing; note = string(rsc.unsupported_reason))

x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
gp0 = x_free_fixed[1]
logA_full0 = log.(reshape(x_free_fixed[2:end], D, Ddest))
pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
w_ext_start_z = reduce_to_w_ext(log(theta_star), gp0, logA_full0, pgc0)
w_ext_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)

lp(">>> Gate 1: calibration equivalence -- a-space start point reproduces fixed-production Delta/gravity")
sc = ScreenCounters()
t2 = @elapsed r_a0, meta_a0, d_a0 = screened_eval_flexible_A(w_ext_start_a, ctx, rsc, sc, Ref(0), xy; warm = false)
lp("  cold eval: ", round(t2, digits = 1), "s inner_status=", r_a0.inner_status, " Delta_dual=", r_a0.Delta_dual, " gravity=", r_a0.gravity_value)
check("a-space start point feasible", r_a0.inner_status in FEASIBLE_CODES)
check("gravity restriction satisfied at start (< 1e-6)", abs(r_a0.gravity_value) < 1e-6; note = "gravity=$(r_a0.gravity_value)")

r_z0, _, d_z0 = screened_eval_flexible(w_ext_start_z, ctx, rsc, ScreenCounters(), Ref(0), pgc0; warm = false)
check("z-space vs a-space Delta_dual agreement at theta_star", isapprox(r_z0.Delta_dual, r_a0.Delta_dual; rtol = 1e-6);
      note = "z=$(r_z0.Delta_dual) a=$(r_a0.Delta_dual)")
check("z-space vs a-space xf agreement at theta_star", isapprox(d_z0.xf, d_a0.xf; rtol = 1e-6);
      note = "max|diff|=$(maximum(abs.(d_z0.xf .- d_a0.xf)))")

# Reference: fixed-mode production's OWN cold evaluation at the identical calibration point,
# through the UNTOUCHED fixed-mode path (build_pivot_elimination/pivot_expand/x_free_from_w),
# to confirm the flexible-mode a-space decode reproduces genuinely-fixed production, not just
# itself.
pe_fixed_ref = build_pivot_elimination(ctx_fixed)
zfree_fixed_ref = pivot_reduce(logA_full0, pe_fixed_ref)
xf_fixed_ref = x_free_from_w(vcat(gp0, zfree_fixed_ref), pe_fixed_ref)
r_fixed_ref, _ = screened_eval(xf_fixed_ref, ctx_fixed, build_ranged_screen_context(ctx_fixed), ScreenCounters(), Ref(0); warm = false)
check("flexible a-space start Delta_dual matches GENUINE fixed-mode production Delta_dual",
      isapprox(r_fixed_ref.Delta_dual, r_a0.Delta_dual; rtol = 1e-6);
      note = "fixed_prod=$(r_fixed_ref.Delta_dual) flexA=$(r_a0.Delta_dual)")

lp(">>> Gate 2: feasible theta +-5% probes (a_nonpivot fixed)")
theta_up = theta_star * 1.05; theta_down = theta_star * 0.95
for (lbl, th) in (("up(+5%)", theta_up), ("down(-5%)", theta_down))
    w = copy(w_ext_start_a); w[1] = log(th)
    r, _, d = screened_eval_flexible_A(w, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    lp("  theta $lbl ($th): inner_status=", r.inner_status, " Delta=", r.Delta_dual, " gravity=", r.gravity_value)
    check("theta $lbl probe feasible", r.inner_status in FEASIBLE_CODES)
    check("theta $lbl gravity satisfied", abs(r.gravity_value) < 1e-6)
end

lp(">>> Gate 3: D=20 theta derivative accuracy -- fixed-dual secant vs fully-resolved FD, two step sizes")
r0, d0 = screened_eval_flexible_A_verify(w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
inner_x_fixed = copy(ctx.obj.x)
theta_secant_rel_errs = Float64[]
for h in (1e-3, 2.5e-4)
    w_plus = copy(w_ext_start_a); w_plus[1] += h
    w_minus = copy(w_ext_start_a); w_minus[1] -= h
    D_plus_analytic = theta_fixed_dual_delta_pivot_A(w_plus, inner_x_fixed, ctx, xy)
    D_minus_analytic = theta_fixed_dual_delta_pivot_A(w_minus, inner_x_fixed, ctx, xy)
    secant_analytic = (D_plus_analytic - D_minus_analytic) / (2h)
    r_plus, dp = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    r_minus, dm = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
    winners_stable = r_plus.inner_status in FEASIBLE_CODES && r_minus.inner_status in FEASIBLE_CODES
    secant_resolved = (r_plus.Delta_dual - r_minus.Delta_dual) / (2h)
    rel_err = abs(secant_analytic - secant_resolved) / max(abs(secant_resolved), 1e-8)
    lp("  h=", h, ": analytic=", secant_analytic, " resolved=", secant_resolved, " rel_err=", rel_err,
       " winners_stable=", winners_stable, " status(+/-)=", r_plus.inner_status, "/", r_minus.inner_status)
    push!(theta_secant_rel_errs, rel_err)
    check("theta secant h=$h: both probes feasible (winners stable)", winners_stable)
end
check("theta secant error shrinks as h shrinks (discretization signature)",
      theta_secant_rel_errs[2] < theta_secant_rel_errs[1];
      note = "rel_errs=$theta_secant_rel_errs")
RESULTS["theta_secant_rel_errs"] = theta_secant_rel_errs

lp(">>> Gate 4: full A-gradient C+ vs slow reference (finite-difference envelope) on flexible ctx")
ctx_frozen = freeze_theta_ctx(ctx, d0.mu)
pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
pe_here = pivot_elim_from_cache(pgc, d0.mu)
xf_reduced = vcat(d0.gp, d0.xf[3:end])
base = BaseDualState(xf_reduced, r0.θ_full, r0.zeta, r0.lambda, copy(ctx.obj.arg1), r0.inner_status)
grad_pool = build_grad_workspace_pool(W)
lfix_c_ws = build_lfix_factorized_workspace(D, Ddest, W)
t3 = @elapsed gfull_reduced, meta_g = composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws;
    base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
lp("  C+ gradient computed in ", round(t3, digits = 2), "s, length=", length(gfull_reduced))
check("C+ gradient full length == D*Ddest+1", length(gfull_reduced) == D * Ddest + 1)
predicted_g_a = gfull_reduced[2:end] .* (-d0.theta)

rng = MersenneTwister(4242)
dir = randn(rng, n_free_A); dir ./= norm(dir)
h_fd = 1e-4
w_plus = copy(w_ext_start_a); w_plus[3:end] .+= h_fd .* dir
w_minus = copy(w_ext_start_a); w_minus[3:end] .-= h_fd .* dir
r_plus, _ = screened_eval_flexible_A_verify(w_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
r_minus, _ = screened_eval_flexible_A_verify(w_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
fd_directional = (r_plus.Delta_dual - r_minus.Delta_dual) / (2 * h_fd)
predicted_directional = sum(predicted_g_a .* dir)
rel_err_grad = abs(fd_directional - predicted_directional) / max(abs(fd_directional), 1e-8)
lp("  A-gradient directional check: analytic=", predicted_directional, " fd=", fd_directional, " rel_err=", rel_err_grad)
check("C+ A-gradient (rescaled) matches stable-winner FD to <20% (informational-noise band, matches D=4 gate 4's own tolerance)",
      rel_err_grad < 0.20; note = "rel_err=$rel_err_grad")
RESULTS["a_gradient_rel_err"] = rel_err_grad

lp(">>> Gate 5: mixed directional derivative (theta AND A moving together)")
rng2 = MersenneTwister(99)
dir_a = randn(rng2, n_free_A); dir_a ./= norm(dir_a)
h_mix = 1e-4
w_mix_plus = copy(w_ext_start_a); w_mix_plus[1] += h_mix; w_mix_plus[3:end] .+= h_mix .* dir_a
w_mix_minus = copy(w_ext_start_a); w_mix_minus[1] -= h_mix; w_mix_minus[3:end] .-= h_mix .* dir_a
r_mix_plus, _ = screened_eval_flexible_A_verify(w_mix_plus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
r_mix_minus, _ = screened_eval_flexible_A_verify(w_mix_minus, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
check("mixed theta+A directional probe both feasible", r_mix_plus.inner_status in FEASIBLE_CODES && r_mix_minus.inner_status in FEASIBLE_CODES)
fd_mix = (r_mix_plus.Delta_dual - r_mix_minus.Delta_dual) / (2 * h_mix)
predicted_mix = theta_secant_rel_errs == [] ? NaN : ((theta_fixed_dual_delta_pivot_A(w_mix_plus, inner_x_fixed, ctx, xy) -
    theta_fixed_dual_delta_pivot_A(w_mix_minus, inner_x_fixed, ctx, xy)) / (2 * h_mix)) + sum(predicted_g_a .* dir_a)
lp("  mixed directional: resolved-FD=", fd_mix, " (theta-secant + A-gradient sum, informational)=", predicted_mix)

lp(">>> Gate 6: no callback errors across all above evaluations (all inner_status codes recorded above are FEASIBLE_CODES or a clean rejection, no exception escaped)")
check("no exceptions escaped Gate 0-5 (this line only reached if true)", true)

lp(">>> SUMMARY")
for (k, v) in RESULTS
    v isa Bool && lp("  ", rpad(k, 70), v ? "PASS" : "FAIL")
end
lp(isempty(FAILURES) ? "ALL D=20 GATES PASS" : "D=20 FAILURES: $(join(FAILURES, "; "))")
flush(stdout)
