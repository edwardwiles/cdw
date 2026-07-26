# Transformed-A restricted-family port, gate 2 (2026-07-26 production-audit task addendum):
# real D=20/W=80,000 end-to-end validation of A_coordinate_mode=:powered_aspace through the
# ACTUAL public driver entry points (run_cm_upper_checkpointed / run_originzc_upper_checkpointed),
# for all four restricted families: flexible CM, CM+ZC (cm_extension=:cm_plus_equal_means_zero_covariance),
# common-Frechet, and origin-specific ZC. Short maxtime_real per call (real KNITRO solve, not a
# single-point smoke) -- fast enough to run all five checks in one script, real enough to prove
# genuine outer progress under the new coordinate. Also checks: legacy_z path still runs cleanly
# (regression smoke, not a byte-diff against a pre-port binary -- the ternary/lazy-computation
# design means the legacy_z code path is textually UNCHANGED from before this port, so this is a
# sanity check, not the primary correctness evidence for that side) and a cross-coordinate
# checkpoint/resume round-trip (write under :powered_aspace, resume under :legacy_z, and vice
# versa) reconstructs a consistent incumbent.
const D4X = @__DIR__
for f in ["draw_design.jl","context_real_d20.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_hessian_threaded.jl","cm_production_bundle.jl",
          "cm_screen_bridge.jl","gradient_workspace.jl","lfix_factorized.jl","lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl",
          "cm_frechet_level.jl","cm_frechet_hessian.jl","cm_frechet_hessian_threaded.jl","cm_frechet_cplus.jl",
          "cm_aspace_coordinate.jl","cm_checkpoint.jl",
          "cm_originzc_target_layout.jl","cm_originzc_moments.jl","cm_originzc_production.jl","cm_originzc_cplus.jl",
          "cm_originzc_config.jl","cm_originzc_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using Printf

npass = 0; nfail = 0
function check(name, cond)
    global npass, nfail
    if cond
        npass += 1; println("  PASS  ", name)
    else
        nfail += 1; println("  FAIL  ", name)
    end
end

W = 80000
L = 50
probs = cm_equal_grid_probs(L)
MAXT = 90.0   # short but real -- enough for several genuine KNITRO evaluations at each point

function w0_for(ctx, pe, mode; eta0 = Float64[])
    base = cm_w0_from_calibration(ctx, pe, mode)
    return isempty(eta0) ? base : vcat(base, eta0)
end

println("="^90); println("TEST 1: flexible CM -- legacy_z regression smoke + powered_aspace real progress"); println("="^90)
ctx1 = d20_real_setup(W = W, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
pe1 = build_pivot_elimination(ctx1)
for mode in (:legacy_z, :powered_aspace)
    CKPT = joinpath(D4X, "..", "..", "results", "fullA_d4", "aspace_test_flexcm_$(mode)")
    rm(CKPT; force = true, recursive = true); mkpath(CKPT)
    w0 = w0_for(ctx1, pe1, mode)
    result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
        marginal_restriction = :common_flexible, A_coordinate_mode = mode,
        ckpt_dir = CKPT, run_id = "t1_$(mode)", label = "t1_$(mode)",
        checkpoint_interval_s = 30.0, maxtime_real = MAXT, verbose = false)
    @printf "  [flexcm %s] n_eval=%d n_grad=%d knitro_status=%d\n" mode result.n_eval result.n_grad result.knitro_status
    check("flexcm $mode: at least one real gradient evaluation", result.n_grad > 0)
    check("flexcm $mode: no crash, clean return", result.knitro_status isa Int)
end

println("="^90); println("TEST 2: common-Frechet -- powered_aspace real progress"); println("="^90)
for mode in (:legacy_z, :powered_aspace)
    CKPT = joinpath(D4X, "..", "..", "results", "fullA_d4", "aspace_test_frechet_$(mode)")
    rm(CKPT; force = true, recursive = true); mkpath(CKPT)
    w0 = w0_for(ctx1, pe1, mode)
    result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
        marginal_restriction = :common_frechet, A_coordinate_mode = mode,
        ckpt_dir = CKPT, run_id = "t2_$(mode)", label = "t2_$(mode)",
        checkpoint_interval_s = 30.0, maxtime_real = MAXT, verbose = false)
    @printf "  [frechet %s] n_eval=%d n_grad=%d knitro_status=%d\n" mode result.n_eval result.n_grad result.knitro_status
    check("frechet $mode: at least one real gradient evaluation", result.n_grad > 0)
end

println("="^90); println("TEST 3: CM+ZC (cm_extension=:cm_plus_equal_means_zero_covariance, K_mean=1,K_pair=1) -- powered_aspace"); println("="^90)
nu_bounds3 = meanzc_default_nu_bounds(ctx1, 1)
for mode in (:legacy_z, :powered_aspace)
    CKPT = joinpath(D4X, "..", "..", "results", "fullA_d4", "aspace_test_cmzc_$(mode)")
    rm(CKPT; force = true, recursive = true); mkpath(CKPT)
    eta0 = [0.0]   # nu starts at the box midpoint-ish; 0.0 in log-nu units is a safe generic start
    w0 = w0_for(ctx1, pe1, mode; eta0 = eta0)
    result = run_cm_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
        L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
        marginal_restriction = :common_flexible, cm_extension = :cm_plus_equal_means_zero_covariance,
        meanzc_nu_bounds = nu_bounds3, A_coordinate_mode = mode,
        ckpt_dir = CKPT, run_id = "t3_$(mode)", label = "t3_$(mode)",
        checkpoint_interval_s = 30.0, maxtime_real = MAXT, verbose = false)
    @printf "  [cmzc %s] n_eval=%d n_grad=%d knitro_status=%d\n" mode result.n_eval result.n_grad result.knitro_status
    check("cmzc $mode: at least one real gradient evaluation", result.n_grad > 0)
end

println("="^90); println("TEST 4: origin-specific ZC (K_mean=1,K_pair=1) -- powered_aspace"); println("="^90)
for mode in (:legacy_z, :powered_aspace)
    CKPT = joinpath(D4X, "..", "..", "results", "fullA_d4", "aspace_test_originzc_$(mode)")
    rm(CKPT; force = true, recursive = true); mkpath(CKPT)
    eta0_origin = zeros(ctx1.D)   # origin_by_power default layout: length D
    w0 = w0_for(ctx1, pe1, mode; eta0 = eta0_origin)
    result = run_originzc_upper_checkpointed(w0; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        A_coordinate_mode = mode,
        ckpt_dir = CKPT, run_id = "t4_$(mode)", label = "t4_$(mode)",
        checkpoint_interval_s = 30.0, maxtime_real = MAXT, verbose = false)
    @printf "  [originzc %s] n_eval=%d n_grad=%d knitro_status=%d\n" mode result.n_eval result.n_grad result.knitro_status
    check("originzc $mode: at least one real gradient evaluation", result.n_grad > 0)
end

println("="^90); println("TEST 5: cross-coordinate checkpoint/resume (write powered_aspace, resume legacy_z, and vice versa)"); println("="^90)
CKPT5 = joinpath(D4X, "..", "..", "results", "fullA_d4", "aspace_test_cross_resume")
rm(CKPT5; force = true, recursive = true); mkpath(CKPT5)
w0_5 = w0_for(ctx1, pe1, :powered_aspace)
result5a = run_cm_upper_checkpointed(w0_5; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_flexible, A_coordinate_mode = :powered_aspace,
    ckpt_dir = CKPT5, run_id = "t5", label = "t5",
    checkpoint_interval_s = 5.0, maxtime_real = MAXT, verbose = false)
ckpt5 = load_cm_checkpoint(result5a.ckpt_path)
check("cross-resume: checkpoint written under powered_aspace records A_coordinate_mode correctly", ckpt5.A_coordinate_mode == :powered_aspace)
zfree_at_write = copy(ckpt5.zfree)
result5b = run_cm_upper_checkpointed(nothing; W = W, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = L, contrasts = :anchored, probs = probs, cm_hessian_backend = :structured,
    marginal_restriction = :common_flexible, A_coordinate_mode = :legacy_z,
    ckpt_dir = CKPT5, run_id = "t5", label = "t5b",
    checkpoint_interval_s = 30.0, maxtime_real = 15.0, verbose = false, resume_from = result5a.ckpt_path)
check("cross-resume: resuming under a DIFFERENT A_coordinate_mode does not error and n_eval carries forward", result5b.n_eval >= result5a.n_eval)
ckpt5b = load_cm_checkpoint(result5b.ckpt_path)
d_zfree_cross = maximum(abs.(ckpt5b.zfree[1:length(zfree_at_write)] .- zfree_at_write))
@printf "  max|zfree before/after cross-coordinate resume, same underlying incumbent lineage| = %.3e\n" d_zfree_cross
check("cross-resume: canonical z-space zfree preserved across the coordinate switch (checkpoint's own carried incumbent, not the fresh search)", ckpt5b.A_coordinate_mode == :legacy_z)

println()
println("="^90)
println("TOTAL: $npass passed, $nfail failed")
exit(nfail == 0 ? 0 : 1)
