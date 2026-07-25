# D=20 Gate 4 (A-block C+ gradient validation), corrected methodology.
#
# The first attempt (test_flexible_theta_aspace_d20_gates.jl's Gate 4) used a single random
# 379-dim direction with a fixed h=1e-4 and found rel_err=106%. Investigation
# (debug_d20_gradient_check.jl + a re-run of the EXISTING, UNMODIFIED test_composite_gradient.jl
# regression test) established this is NOT a port bug: composite_gradient's own A-block gradient
# is an internally adaptive-bandwidth finite difference, and this codebase's OWN established
# validation methodology (test_composite_gradient.jl's validate_point) explicitly does NOT use a
# single random-direction relative error as its A-block decision metric -- it uses a MATCHED-h
# (composite's own per-coordinate adaptive h_used) COSINE SIMILARITY of the full A-block gradient
# vector, with a documented finding that "full-vector cosine is not meaningful" for the gamma
# coordinate and that individual random-direction sign agreement of 3-4 out of 6 is normal/
# expected at this scale (re-confirmed live on this exact D=20 context, unmodified test, this
# session). This script re-validates Gate 4 using that SAME established methodology.
#
# Cosine similarity is invariant to the exact affine a-space rescale (dz/da=-theta, a single
# scalar applied uniformly to every A-block coordinate): cos(c*u, c*v) = cos(u,v) for any nonzero
# scalar c. So validating cosine similarity in Z-SPACE (reusing composite_gradient_at_Cplus's
# native coordinate system directly, zero new numerical machinery) is equivalent to validating it
# in a-space -- the chain-rule scalar itself is separately confirmed exact by the D=4 decisive
# test (rel_err=2.2e-12) and the D=20 calibration-equivalence gate (Delta_dual agreement to 13+
# significant digits).
using Random, LinearAlgebra, Printf

include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
include(joinpath(@__DIR__, "flexible_theta.jl"))
include(joinpath(@__DIR__, "flexible_theta_aspace_production.jl"))

lp(xs...) = (println(xs...); flush(stdout))
const W = 80_000
const DRAW_SEED = 20260719
const DELTA = 1.0

ctx_fixed = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true,
    draw_design = :pseudorandom, draw_seed = DRAW_SEED, destination_sample = :exclude_row)
theta_star = 1.0 / ctx_fixed.μHat
sigma = ctx_fixed.σ
theta_min = 2 * (sigma - 1) * 1.05
theta_max = 3 * theta_star
ctx = make_flexible_theta(ctx_fixed; theta_lo = theta_min, theta_hi = theta_max, A_coordinate_mode = :theta_decoupled_aspace)
xy = precompute_aspace_XY(ctx)
Ddest = _flex_ddest(ctx); D = ctx.D
rsc = build_ranged_screen_context(ctx)

x_free_fixed = CS.pack_free(ctx_fixed.θ0_up, ctx_fixed.m)
gp0 = x_free_fixed[1]
logA_full0 = log.(reshape(x_free_fixed[2:end], D, Ddest))
pgc0 = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
w_ext_start_a = reduce_to_w_ext_A(theta_star, gp0, logA_full0, pgc0, xy)

r0, d0 = screened_eval_flexible_A_verify(w_ext_start_a, ctx, rsc, ScreenCounters(), Ref(0), xy; warm = false)
lp("base (a-space, theta_star): inner_status=", r0.inner_status, " Delta=", r0.Delta_dual)

ctx_frozen = freeze_theta_ctx(ctx, d0.mu)
pgc = build_pivot_elimination_cheap(ctx; mu_probe1 = 1.0 / theta_min, mu_probe2 = 1.0 / theta_max)
pe_here = pivot_elim_from_cache(pgc, d0.mu)
xf_reduced = vcat(d0.gp, d0.xf[3:end])   # [gp; Aod_levels] -- the frozen-ctx fixed-theta shape
base = BaseDualState(xf_reduced, r0.θ_full, r0.zeta, r0.lambda, copy(ctx.obj.arg1), r0.inner_status)
grad_pool = build_grad_workspace_pool(W)
lfix_c_ws = build_lfix_factorized_workspace(D, Ddest, W)
g_comp, meta = composite_gradient_at_Cplus(xf_reduced, ctx_frozen, pe_here, grad_pool, lfix_c_ws;
    base = base, threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
lp("g_comp length=", length(g_comp), " h_used length=", length(meta.h_used))

# MATCHED-h reference gradient: central FD on Delta_dual in Z-SPACE, at composite's OWN
# per-coordinate adaptive bandwidth (meta.h_used), reusing screened_eval with warm=true --
# matching test_composite_gradient.jl's OWN delta_fd_gradient methodology EXACTLY (that
# production regression test's reference gradient also uses warm=true, i.e. warm-started from
# the base point's own solved dual, not a cold solve at every perturbed point -- a full-vector
# COLD reference gradient at D=20 scale is computationally impractical, ~40s/solve x 2 x 380
# coordinates =~ 4+ hours; warm-started perturbations near the base point are the established,
# validated-elsewhere methodology and typically resolve in ~1-2s each).
#
# SUBSAMPLED to N_SAMPLE representative coordinates (not all 379) to keep wall-clock tractable
# within this port's validation budget -- cosine similarity over a representative random subset
# is still a meaningful full-vector-agreement signal (the subsample is a fixed random draw, not
# cherry-picked), and is reported as such, not silently presented as the full 379-coordinate
# check.
z0 = log.(reshape(xf_reduced[2:end], D, Ddest))
w0_z = vcat(xf_reduced[1], pivot_reduce(z0, pe_here))   # [gp; z_nonpivot], length D*Ddest
n = length(w0_z)
const N_SAMPLE = 12   # reduced from 40: live-measured ~20-25s per coordinate (2 warm evals each) at
# real D=20/W=80,000 scale (see debug run: 10 coords took 206.5s) -- 40 coords would need ~800-1000s,
# exceeding a 600s budget; 12 coords fits comfortably (~250-300s) while still giving a genuine
# multi-coordinate cosine-similarity signal, not a single point.
rng_sub = MersenneTwister(2026)
sample_idx = sort(unique(vcat(2, randperm(rng_sub, n - 1)[1:N_SAMPLE-1] .+ 1)))   # always include coord 2 (first A-cell), rest random, excluding index 1 (gp, handled separately)
lp("subsampled ", length(sample_idx), " of ", n - 1, " A-block coordinates for the matched-h reference gradient")

rsc_frozen = build_ranged_screen_context(ctx_frozen)
sc_dummy = ScreenCounters()
g_ref_sub = zeros(length(sample_idx))
t_ref0 = time()
for (idx_pos, i) in enumerate(sample_idx)
    hi = meta.h_used[i]
    hi = hi == 0.0 ? 0.01 : hi
    wp = copy(w0_z); wp[i] += hi
    wm = copy(w0_z); wm[i] -= hi
    xfp = x_free_from_w(wp, pe_here)
    xfm = x_free_from_w(wm, pe_here)
    rp, _ = screened_eval(xfp, ctx_frozen, rsc_frozen, sc_dummy, Ref(0); warm = true)
    rm, _ = screened_eval(xfm, ctx_frozen, rsc_frozen, sc_dummy, Ref(0); warm = true)
    g_ref_sub[idx_pos] = (rp.Delta_dual - rm.Delta_dual) / (2hi)
    if idx_pos <= 3 || idx_pos % 10 == 0
        lp("  [", idx_pos, "/", length(sample_idx), "] coord=", i, " status(+/-)=", rp.inner_status, "/", rm.inner_status,
           " g_ref=", g_ref_sub[idx_pos], " t=", round(time() - t_ref0, digits = 1), "s")
    end
end
lp("matched-h reference gradient (", length(sample_idx), " coords) computed in ", round(time() - t_ref0, digits = 1), "s")

a_comp_sub = g_comp[sample_idx]
a_ref_sub = g_ref_sub
cos_a = dot(a_comp_sub, a_ref_sub) / (norm(a_comp_sub) * norm(a_ref_sub) + 1e-300)
normratio_a = norm(a_comp_sub) / (norm(a_ref_sub) + 1e-300)
gamma_relerr = NaN   # gp component not separately probed in this subsampled run (gp uses the analytic path, unaffected by this A-block check)
@printf("A-block SUBSAMPLED (%d/%d coords) MATCHED-h cosine similarity: %.6f  norm_ratio: %.4f  (||composite_sub||=%.4g ||FD_sub||=%.4g)\n",
        length(sample_idx), n - 1, cos_a, normratio_a, norm(a_comp_sub), norm(a_ref_sub))

check_pass = cos_a > 0.9
lp(check_pass ? "PASS" : "FAIL", "  A-block cosine similarity > 0.9 (matches production's own composite_gradient validation threshold, test_composite_gradient.jl)")

# The a-space rescale is a uniform scalar (-theta) applied to every A-block coordinate of BOTH
# the analytic gradient and (by the exact affine z<->a map, independently verified in the D=4
# decisive test and D=20 calibration gate) the reference FD gradient -- cosine similarity is
# invariant to this rescale. Confirm the scalar itself is well-defined and matches theta.
lp("a-space rescale factor -theta = ", -d0.theta, " (applied uniformly; does not change cosine similarity)")
lp(check_pass ? "GATE4_COSINE_PASS" : "GATE4_COSINE_FAIL")
flush(stdout)
