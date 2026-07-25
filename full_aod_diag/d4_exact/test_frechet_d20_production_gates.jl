# ============================================================================
# D=20 W=80,000 omit-ROW correctness gates (task brief §11's D=20 section).
# Real data, real calibration, destination_sample=:exclude_row, L=50,
# frechet_feature_set=:cdf_power (full paper spec).
#
# The dense (Architecture A) structured-vs-dense Hessian check is NOT run at
# the full L=50 scale here (ncm=2000, an ncore+ncm ~= 2020-wide dense
# Hessian construction is compute-prohibitive as a routine gate at this
# system's current load -- see the port-readiness report's disclosed
# scoping) -- it runs on a BOUNDED sub-problem (small L) at real D=20/W=80,000
# data instead, per the task brief's own "bounded subproblem... if feasible"
# allowance. The D=4 gates (test_frechet_power_hessian_d4_gates.jl) already
# validate the FULL L=50-shaped structured Hessian algebra against dense to
# machine precision; this script's job is to confirm the same code paths
# run correctly at real D=20/rectangular scale, not to re-derive the algebra.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
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
include(joinpath(@__DIR__, "cm_screen_bridge.jl"))
include(joinpath(@__DIR__, "knitro_status.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_checkpoint.jl"))
using Printf, LinearAlgebra

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end

const W_MAIN = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80_000
const L_MAIN = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
const L_BOUNDED = 8

println("="^100); println("SETUP: real D=20 data, W=$W_MAIN, destination_sample=:exclude_row"); println("="^100)
t_ctx = @elapsed ctx = d20_real_setup_design(W = W_MAIN, δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false,
    destination_sample = :exclude_row)
println("ctx build: $(round(t_ctx,digits=1))s  D=$(ctx.D) D_dest=$(ctx.D_dest)")
x_free_calib = ctx.θ0_up[ctx.free_idx]

println("="^100); println("GATE D1: bounded-L (L=$L_BOUNDED) structured vs dense Hessian at real D=20/W=$W_MAIN"); println("="^100)
cfg_b = CMFrechetConfig(cm = CMConfig(cm_grid_size = L_BOUNDED, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
targets_b = build_frechet_reference_targets(ctx, cfg_b; L = L_BOUNDED)
report_frechet_targets(ctx, cfg_b, targets_b)
t_aug = @elapsed aug_b = build_cm_frechet_augmented_obj_basis(ctx, CS, targets_b; basis=:cumulative, feature_set=:cdf_power, contrasts=:orthonormal)
println("dense CM construction (L=$L_BOUNDED): $(round(t_aug,digits=2))s, ncm=$(aug_b.ncm)")
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
t_dense = @elapsed (K_d, x_d, ns_d, _, _) = inner_loop_internal_archgeneric(aug_b.obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
println("dense solve: $(round(t_dense,digits=1))s  nStatus=$ns_d  category=$(decode_knitro_status(ns_d).category)")
if decode_knitro_status(ns_d).is_feasible_result
    n_inner = aug_b.ncore + aug_b.ncm
    nh = div(n_inner*(n_inner+1),2)
    h_dense = zeros(nh); aug_b.obj_cm(x_d, h = h_dense)
    aug_cdf_b = build_cm_frechet_augmented_obj_archB(ctx, CS, targets_b; contrasts = :orthonormal)
    fctx_b = build_frechet_power_bin_ctx(ctx, aug_cdf_b, targets_b)
    aug_b.obj_cm.moments!(@view(aug_b.obj_cm.H[:,1]), CS.select_G_from_H(aug_b.obj_cm, aug_b.obj_cm.H), θ_full0, aug_b.obj_cm.U, aug_b.obj_cm)
    aug_b.obj_cm.H[:,2] .= 1.0
    _archC_prep_for_hessian!(aug_b.obj_cm, x_d)
    h_struct = zeros(nh)
    t_struct = @elapsed hessian_cm_frechet_cdf_power_structured!(h_struct, aug_b.obj_cm, fctx_b)
    diff = abs.(h_dense .- h_struct)
    maxerr, kmax = findmax(diff)
    maxabs_dense = maximum(abs.(h_dense))
    relerr = maxerr / (maxabs_dense + 1e-300)
    # locate (i,j) of kmax in the packed upper-triangular layout (n=aug_b.ncore+aug_b.ncm)
    function packed_triu_index(kmax::Int, n::Int)
        k = 0
        for ii in 1:n, jj in ii:n
            k += 1
            k == kmax && return (ii, jj)
        end
        return (0, 0)
    end
    n = aug_b.ncore + aug_b.ncm
    i_hit, j_hit = packed_triu_index(kmax, n)
    ncore = aug_b.ncore; ncm_fam = div(aug_b.ncm, 2)
    blockname(idx) = idx <= ncore ? :core : (idx <= ncore+ncm_fam ? :cdf_block : :power_block)
    println("structured Hessian build: $(round(t_struct,digits=3))s")
    println("  max|dense-struct| = $maxerr  (relative to max|h_dense|=$maxabs_dense: $relerr)")
    println("  worst entry at packed-index $kmax -> (i=$i_hit [$(blockname(i_hit))], j=$j_hit [$(blockname(j_hit))])")
    println("  h_dense[worst]=$(h_dense[kmax])  h_struct[worst]=$(h_struct[kmax])")
    check("D=20 real-data bounded-L dense vs structured Hessian match (1e-6 abs OR 1e-8 relative)", maxerr < 1e-6 || relerr < 1e-8)
else
    check("D=20 bounded-L dense solve feasible (prerequisite for gate)", false)
end

println("="^100); println("GATE D2: calibrated benchmark evaluation, full L=$L_MAIN, :cdf_power (structured Hessian only -- dense not attempted at this scale)"); println("="^100)
cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L_MAIN, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
t_fpcx = @elapsed fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L_MAIN)
println("fpcx (L=$L_MAIN) construction: $(round(t_fpcx,digits=1))s  ncore=$(fpcx.aug.ncore) ncm=$(fpcx.aug.ncm)")
check("ncm == 2*D*L at full scale", fpcx.aug.ncm == 2*ctx.D*L_MAIN)

_ok = false; _base = nothing; _verify = nothing
t_solve = @elapsed try
    global _base, _verify = cm_frechet_verified_state(x_free_calib, fpcx)
    global _ok = true
catch e
    e isa CMExpectedSolveFailure || rethrow()
    println("  solve at raw calibration point failed: ", sprint(showerror, e)[1:min(300,end)])
end
println("calibrated-point solve (L=$L_MAIN, cdf_power): $(round(t_solve,digits=1))s")
if _ok
    println("  inner_status=$(_base.inner_status)  outcome=$(frechet_solve_outcome(_base.inner_status))  verified=$(is_verified_success(_verify))")
    check("calibrated-point solve feasible", frechet_solve_outcome(_base.inner_status) == :feasible)
    Delta = _verify.Delta_dual
    @printf "  Delta* = %.10f\n" Delta
    check("Delta* finite and nonnegative", isfinite(Delta) && Delta >= 0)
else
    check("calibrated-point solve at full L=$L_MAIN (see port-readiness report for disclosed diagnosis)", false)
end

println()
println("="^100)
@printf "TOTAL: %d PASS, %d FAIL\n" n_pass n_fail
println("="^100)
