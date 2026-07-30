# ================================================================================================
# cm_meanzc / origin_zc DRIVER-LEVEL moment_representation wiring gate (2026-07-29).
#
# Companion to test_unrestricted_operator_ctx_driver_wiring_2026-07-29.jl. The pre-existing
# standalone equivalence gates (test_operator_no_H_bundle_equivalence_cmzc[_d20].jl /
# _originzc[_d20].jl) prove OperatorPsiBundle agrees with the dense reference to machine precision
# via the real build_*_production_context builders and a real archC_meanzc_base_state/
# archOZ_base_state inner solve -- but neither ever called the REAL top-level production driver
# (run_cm_upper_checkpointed / run_originzc_upper_checkpointed), which previously never threaded a
# moment_representation kwarg through at all -- both builders' own defaults were hardcoded
# :dense_reference, so ctx.obj stayed dense in every real production run for these two families.
#
# THIS script calls the REAL drivers, exactly as smoke_delta1_cmzc.jl/smoke_delta1_originzc.jl do
# (real D=20/W=100,000, calibration start), now passing the newly-threaded moment_representation
# kwarg explicitly both ways, and checks -- via the driver's own returned ctx -- that the bundle
# actually used was OperatorPsiBundle vs PsiObjectiveBundleImplicit as requested, with no crash.
# ================================================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_aspace_coordinate.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_config.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl", "cm_originzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl", "cm_dual_bank_production.jl",
          "cm_checkpoint.jl", "cm_originzc_checkpoint.jl", "postmerge_smoke_diagnostics.jl"]
    include(joinpath(_D4E, f))
end
using Printf, Dates, Statistics

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

const W = 100_000
const DELTA = 1.0
const BUDGET = 45.0
const FIND_SMALLEST = true

lp("="^100)
lp("cm_meanzc / origin_zc driver-level moment_representation wiring gate -- real D=20/W=", W)
lp("="^100)

ctx0 = d20_real_setup(W = W, δ = DELTA, find_smallest = FIND_SMALLEST, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
D = ctx0.D; Ddest = ctx0.D_dest
theta0 = cm_fixed_theta(ctx0)
xy0 = precompute_cm_aspace_xy(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]
gp_calib = x_free_calib[1]
z_calib = pivot_reduce(log.(reshape(x_free_calib[2:end], D, Ddest)), pe0)
a_calib = cm_a_from_z(z_calib, theta0, xy0, pe0)
w_a_calib = vcat(gp_calib, a_calib)
const SNAPS = nested_grid_sequence([10, 20, 50])
const PROBS_L50 = SNAPS[50]
lp(">>> D=", D, " Ddest=", Ddest, " ||w_a_calib||=", norm(w_a_calib))

# ---------------------------------------------------------------------------------------------
# cm_meanzc
# ---------------------------------------------------------------------------------------------
function run_meanzc(tag::String; moment_representation::Union{Nothing,Symbol})
    out = joinpath(_D4E, "results", "test_meanzc_originzc_driver_wiring_2026-07-29", "cmzc_$tag")
    rm(out; force = true, recursive = true); mkpath(out)
    mr_kwargs = moment_representation === nothing ? NamedTuple() : (moment_representation = moment_representation,)
    t0 = time()
    result = run_cm_upper_checkpointed(vcat(w_a_calib, log.(Float64.(factorial.(1:1))));
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719, L = 50, contrasts = :orthonormal, probs = PROBS_L50,
        cm_extension = :cm_plus_moments, meanzc_K_mean = 1, meanzc_K_pair = 1,
        ckpt_dir = out, run_id = "cmzc_$tag", label = "cmzc_$tag",
        checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = true, mr_kwargs...)
    lp(">>> [cmzc/$tag] done in ", time() - t0, "s")
    # cross_hessian_live_stash_2026-07-28.jl: STASH_LIVE_PCX_ENABLED[] defaults true, so
    # build_cm_meanzc_production_context (called once, at driver setup, inside
    # run_cm_upper_checkpointed above) already stashed the REAL pcx this run used into
    # CMZC_LIVE_PCX_STASH[] -- read it IMMEDIATELY (before the next run overwrites it) to check
    # the actual bundle type the driver just searched with. run_cm_upper_checkpointed's own return
    # NamedTuple does not carry ctx/pcx directly.
    obj_type = CMZC_LIVE_PCX_STASH[].ctx_cm.obj
    return result, obj_type
end

lp("-"^90); lp("cm_meanzc Run 1: DEFAULT (no moment_representation passed) -- must now resolve to :operator (flipped)")
res_meanzc_default, obj_meanzc_default = run_meanzc("default"; moment_representation = nothing)
check("cm_meanzc default: ctx.obj isa OperatorPsiBundle (flipped default)", obj_meanzc_default isa OperatorPsiBundle)

lp("-"^90); lp("cm_meanzc Run 2: explicit moment_representation=:operator")
res_meanzc_op, obj_meanzc_op = run_meanzc("operator"; moment_representation = :operator)
check("cm_meanzc explicit :operator: ctx.obj isa OperatorPsiBundle", obj_meanzc_op isa OperatorPsiBundle)

lp("-"^90); lp("cm_meanzc Run 3: explicit moment_representation=:dense_reference")
res_meanzc_dense, obj_meanzc_dense = run_meanzc("dense_reference"; moment_representation = :dense_reference)
check("cm_meanzc explicit :dense_reference: ctx.obj isa PsiObjectiveBundleImplicit", obj_meanzc_dense isa CS.PsiObjectiveBundleImplicit)

lp("cm_meanzc results: default=", res_meanzc_default.knitro_status, " operator=", res_meanzc_op.knitro_status,
   " dense_reference=", res_meanzc_dense.knitro_status, " (kappa: ", res_meanzc_default.kappa, " / ",
   res_meanzc_op.kappa, " / ", res_meanzc_dense.kappa, ")")
check("cm_meanzc: all three runs reached a feasible/time-limit-feasible KNITRO status",
      all(s -> s in (0, -401, -406), (res_meanzc_default.knitro_status, res_meanzc_op.knitro_status, res_meanzc_dense.knitro_status)))

# ---------------------------------------------------------------------------------------------
# origin_zc
# ---------------------------------------------------------------------------------------------
nu0_log = begin
    layout0 = OriginByPowerLayout(D, 1, 1)
    nu0 = Vector{Float64}(undef, n_eta(layout0))
    for k in 1:1, o in 1:D
        nu0[target_index(layout0, o, k)] = mean(@view (ctx0.U .^ k)[:, o])
    end
    log.(nu0)
end

function run_originzc(tag::String; moment_representation::Union{Nothing,Symbol})
    out = joinpath(_D4E, "results", "test_meanzc_originzc_driver_wiring_2026-07-29", "originzc_$tag")
    rm(out; force = true, recursive = true); mkpath(out)
    mr_kwargs = moment_representation === nothing ? NamedTuple() : (moment_representation = moment_representation,)
    t0 = time()
    result = run_originzc_upper_checkpointed(vcat(w_a_calib, nu0_log);
        W = W, delta = DELTA, draw_design = :pseudorandom, draw_seed = 20260719,
        distribution_restriction = :origin_specific_moments_zero_covariance, K_mean = 1, K_pair = 1,
        ckpt_dir = out, run_id = "originzc_$tag", label = "originzc_$tag",
        checkpoint_interval_s = 3600.0, maxtime_real = BUDGET, verbose = true, mr_kwargs...)
    lp(">>> [originzc/$tag] done in ", time() - t0, "s")
    # Same live-stash pattern as run_meanzc above (ORIGINZC_LIVE_PCX_STASH, cross_hessian_live_
    # stash_2026-07-28.jl) -- run_originzc_upper_checkpointed's own return NamedTuple does not
    # carry ctx/pcx either.
    obj_type = ORIGINZC_LIVE_PCX_STASH[].ctx_cm.obj
    return result, obj_type
end

lp("-"^90); lp("origin_zc Run 1: DEFAULT (no moment_representation passed) -- must now resolve to :operator (flipped)")
res_oz_default, obj_oz_default = run_originzc("default"; moment_representation = nothing)
check("origin_zc default: ctx.obj isa OperatorPsiBundle (flipped default)", obj_oz_default isa OperatorPsiBundle)

lp("-"^90); lp("origin_zc Run 2: explicit moment_representation=:operator")
res_oz_op, obj_oz_op = run_originzc("operator"; moment_representation = :operator)
check("origin_zc explicit :operator: ctx.obj isa OperatorPsiBundle", obj_oz_op isa OperatorPsiBundle)

lp("-"^90); lp("origin_zc Run 3: explicit moment_representation=:dense_reference (opt-out)")
res_oz_dense, obj_oz_dense = run_originzc("dense_reference"; moment_representation = :dense_reference)
check("origin_zc explicit :dense_reference: ctx.obj isa PsiObjectiveBundleImplicit", obj_oz_dense isa CS.PsiObjectiveBundleImplicit)

lp("origin_zc results: default=", res_oz_default.knitro_status, " operator=", res_oz_op.knitro_status,
   " dense_reference=", res_oz_dense.knitro_status, " (kappa: ", res_oz_default.kappa, " / ", res_oz_op.kappa,
   " / ", res_oz_dense.kappa, ")")
check("origin_zc: all three runs reached a feasible/time-limit-feasible KNITRO status",
      all(s -> s in (0, -401, -406), (res_oz_default.knitro_status, res_oz_op.knitro_status, res_oz_dense.knitro_status)))

println("="^100)
if isempty(FAILURES)
    println("ALL CM_MEANZC/ORIGIN_ZC DRIVER-LEVEL WIRING GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
