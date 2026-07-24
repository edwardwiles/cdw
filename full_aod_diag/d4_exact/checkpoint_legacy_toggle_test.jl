# Gate D (exclude-ROW-destination production release, 2026-07-24): destination_sample legacy
# toggle checks. Uses the exclude_row checkpoint already written by
# checkpoint_write_exclude_row_cplus.jl (results/fullA_d4/cm_ckpt_excluderow_cplus_smoke_test),
# plus a fresh short :all_legacy write, to confirm:
#   1. resume of the :exclude_row checkpoint under :all_legacy is REFUSED
#   2. a short :all_legacy smoke runs and writes its own checkpoint
#   3. resume of THAT :all_legacy checkpoint under :exclude_row is REFUSED
const D4X = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl",
          "gravity_elimination.jl", "three_way_derivatives.jl", "lfix_incremental.jl",
          "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl",
          "cm_meanzc_production.jl", "cm_meanzc_cplus.jl", "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
println("=== includes OK ==="); flush(stdout)
using Printf

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

EXCL_CKPT = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_excluderow_cplus_smoke_test", "cm_excl_cplus_smoke_latest.jls")
@assert isfile(EXCL_CKPT) "no exclude_row checkpoint found at $EXCL_CKPT -- run checkpoint_write_exclude_row_cplus.jl first"

println("\n=== Check 1: resume of :exclude_row checkpoint under :all_legacy must be REFUSED ==="); flush(stdout)
let rejected = false
    try
        run_cm_upper_checkpointed(; ckpt_dir = dirname(EXCL_CKPT), run_id = "legacy_toggle_check1",
            label = "cm_excl_cplus_smoke", resume_from = EXCL_CKPT, maxtime_real = 5.0,
            cm_gradient_backend = :cplus, destination_sample = :all_legacy)
    catch e
        rejected = true
        println("  (expected) rejected with: ", sprint(showerror, e)[1:min(200, end)])
    end
    check("resume of :exclude_row checkpoint under :all_legacy refused", rejected)
end

println("\n=== Check 2: short explicit :all_legacy smoke (real KNITRO) ==="); flush(stdout)
LEGACY_CKPT_DIR = joinpath(D4X, "..", "..", "results", "fullA_d4", "cm_ckpt_legacy_toggle_smoke_test")
rm(LEGACY_CKPT_DIR; recursive = true, force = true)
mkpath(LEGACY_CKPT_DIR)
ctx0_leg = d20_real_setup(W = 80000, δ = 1.0, find_smallest = true, destination_sample = :all_legacy)
pe0_leg = build_pivot_elimination(ctx0_leg)
D_leg = ctx0_leg.D; Ddest_leg = ctx0_leg.D_dest
check("legacy ctx is square (D_dest == D)", Ddest_leg == D_leg)
x_free_calib_leg = ctx0_leg.θ0_up[ctx0_leg.free_idx]
w0_leg = vcat(x_free_calib_leg[1], pivot_reduce(log.(reshape(x_free_calib_leg[2:end], D_leg, Ddest_leg)), pe0_leg))
snaps10 = nested_grid_sequence([10])[10]
res_leg = run_cm_upper_checkpointed(w0_leg; W = 80000, delta = 1.0, draw_design = :pseudorandom, draw_seed = 20260719,
    L = 10, contrasts = :anchored, probs = snaps10, cm_gradient_backend = :cplus,
    destination_sample = :all_legacy,
    ckpt_dir = LEGACY_CKPT_DIR, run_id = "legacy_toggle_smoke", label = "cm_legacy_smoke",
    maxtime_real = 60.0, checkpoint_interval_s = 15.0)
@printf("legacy smoke: knitro_status=%d n_eval=%d wall=%.1fs\n", res_leg.knitro_status, res_leg.n_eval, res_leg.wall)
check("legacy smoke wrote a checkpoint", isfile(res_leg.ckpt_path))
legacy_latest = load_cm_checkpoint(res_leg.ckpt_path)
check("legacy checkpoint destination_sample == :all_legacy", legacy_latest.destination_sample == :all_legacy)
check("legacy checkpoint row_idx === nothing", legacy_latest.row_idx === nothing)
check("legacy checkpoint D_dest == D", legacy_latest.D_dest == D_leg)

println("\n=== Check 3: resume of :all_legacy checkpoint under :exclude_row must be REFUSED ==="); flush(stdout)
let rejected = false
    try
        run_cm_upper_checkpointed(; ckpt_dir = LEGACY_CKPT_DIR, run_id = "legacy_toggle_check3",
            label = "cm_legacy_smoke", resume_from = res_leg.ckpt_path, maxtime_real = 5.0,
            cm_gradient_backend = :cplus, destination_sample = :exclude_row)
    catch e
        rejected = true
        println("  (expected) rejected with: ", sprint(showerror, e)[1:min(200, end)])
    end
    check("resume of :all_legacy checkpoint under :exclude_row refused", rejected)
end

println()
if ALL_PASS[]
    println(">>> RESULT: ALL PASS")
else
    println(">>> RESULT: SOME FAILURES -- see above")
    exit(1)
end
