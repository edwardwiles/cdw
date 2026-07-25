# ============================================================================
# D=4 correctness gates for the port-prep branch's NEW deliverable: the fast
# structured Hessian for the FULL CDF+truncated-power fixed-Fréchet block
# (cm_frechet_power_hessian_structured.jl). Also re-validates the ported
# CDF-only structured kernel (cm_frechet_hessian.jl) and checks the
# dimension-safety assertions (task brief §2/§4).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
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
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases_structured.jl"))
include(joinpath(@__DIR__, "cm_frechet_bases_q2.jl"))
include(joinpath(@__DIR__, "cm_frechet_power_hessian_structured.jl"))
using Printf, LinearAlgebra, Random, Statistics

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1; println("  PASS: ", name)
    else
        n_fail += 1; println("  FAIL: ", name)
    end
end
function checkapprox(name, a, b; atol=1e-8, rtol=1e-8)
    check(name * "  [a=$a b=$b diff=$(abs(a-b))]", isapprox(a, b; atol=atol, rtol=rtol))
end

println("="^100); println("SETUP"); println("="^100)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D; W = size(ctx.U, 1)
const L = 8
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
println("D=$D W=$W L=$L")

cfg_frec = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured),
    marginal_mode = :frechet_reference, frechet_feature_set = :cdf_power, frechet_basis = :cumulative)
targets = build_frechet_reference_targets(ctx, cfg_frec; L = L)
report_frechet_targets(ctx, cfg_frec, targets)
refIndex1 = ctx.γ.refIndex1
origins = [o for o in 1:D if o != refIndex1]
nO = length(origins)

# ============================================================================
println("="^100); println("GATE P0: dimension-safety assertions (task brief §2/§4)"); println("="^100)
# ============================================================================
check("ctx.D == 4 (origin count)", ctx.D == 4)
check("size(ctx.U,2) == ctx.D (targets built off origin count, not D_dest)", size(ctx.U, 2) == ctx.D)
check("targets.D == ctx.D", targets.D == ctx.D)
try
    build_frechet_reference_targets(merge(ctx, (U = ctx.U[:, 1:3],)), cfg_frec; L = L)
    check("build_frechet_reference_targets rejects a U with D_dest-shaped (not D-shaped) column count", false)
catch e
    check("build_frechet_reference_targets rejects a U with D_dest-shaped (not D-shaped) column count", e isa AssertionError || occursin("ctx.D", sprint(showerror, e)))
end

# ============================================================================
println("="^100); println("GATE P1: CDF-only structured (Architecture C) vs dense (Architecture A) Hessian"); println("="^100)
# ============================================================================
aug_cdf = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = :orthonormal)
n_cdf = aug_cdf.ncore + aug_cdf.ncm
nh_cdf = div(n_cdf*(n_cdf+1), 2)

K_cdf, x_cdf, ns_cdf, _, _ = inner_loop_internal_archgeneric(aug_cdf.obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
check("CDF-only dense solve feasible", ns_cdf in (0,-100,-101,-103))
h_dense_cdf = zeros(nh_cdf)
aug_cdf.obj_cm(x_cdf, h = h_dense_cdf)

fctx_cdf = build_cm_frechet_bin_ctx(ctx, aug_cdf)
aug_cdf.obj_cm.moments!(@view(aug_cdf.obj_cm.H[:,1]), CS.select_G_from_H(aug_cdf.obj_cm, aug_cdf.obj_cm.H), θ_full0, aug_cdf.obj_cm.U, aug_cdf.obj_cm)
aug_cdf.obj_cm.H[:,2] .= 1.0
_archC_prep_for_hessian!(aug_cdf.obj_cm, x_cdf)
h_struct_cdf = zeros(nh_cdf)
hessian_cm_frechet_structured!(h_struct_cdf, aug_cdf.obj_cm, fctx_cdf)

maxerr_cdf = maximum(abs.(h_dense_cdf .- h_struct_cdf))
println("max|H_dense_CDF - H_struct_CDF| = $maxerr_cdf")
check("CDF-only dense vs structured Hessian match to 1e-8 abs", maxerr_cdf < 1e-8)

# ============================================================================
println("="^100); println("GATE P2 (NEW deliverable): CDF+POWER structured (Architecture C) vs dense (Architecture A) Hessian"); println("="^100)
# ============================================================================
aug_cp = build_cm_frechet_augmented_obj_basis(ctx, CS, targets; basis = :cumulative, feature_set = :cdf_power, contrasts = :orthonormal)
n_cp = aug_cp.ncore + aug_cp.ncm
nh_cp = div(n_cp*(n_cp+1), 2)
println("ncore=$(aug_cp.ncore)  ncm=$(aug_cp.ncm)  (expect 2*D*L = $(2*D*L))")
check("ncm == 2*D*L under feature_set=:cdf_power", aug_cp.ncm == 2*D*L)

K_cp, x_cp, ns_cp, _, _ = inner_loop_internal_archgeneric(aug_cp.obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
check("CDF+POWER dense solve feasible", ns_cp in (0,-100,-101,-103))
h_dense_cp = zeros(nh_cp)
aug_cp.obj_cm(x_cp, h = h_dense_cp)

# Build the CDF-only `aug` (same origins/refIndex1/z/contrasts/ncore) that build_frechet_power_bin_ctx needs
# as its base cctx -- this is purely a layout descriptor (see cm_frechet_power_hessian_structured.jl docstring),
# NOT a second/inconsistent moment block: aug_cdf (built above) already has exactly this shape.
fctx_cp = build_frechet_power_bin_ctx(ctx, aug_cdf, targets)
aug_cp.obj_cm.moments!(@view(aug_cp.obj_cm.H[:,1]), CS.select_G_from_H(aug_cp.obj_cm, aug_cp.obj_cm.H), θ_full0, aug_cp.obj_cm.U, aug_cp.obj_cm)
aug_cp.obj_cm.H[:,2] .= 1.0
_archC_prep_for_hessian!(aug_cp.obj_cm, x_cp)
h_struct_cp = zeros(nh_cp)
hessian_cm_frechet_cdf_power_structured!(h_struct_cp, aug_cp.obj_cm, fctx_cp)

maxerr_cp = maximum(abs.(h_dense_cp .- h_struct_cp))
relerr_cp = maxerr_cp / (maximum(abs.(h_dense_cp)) + 1e-300)
println("max|H_dense_CDF+POWER - H_struct_CDF+POWER| = $maxerr_cp  (relative: $relerr_cp)")
check("CDF+POWER dense vs structured Hessian match to 1e-8 abs", maxerr_cp < 1e-8)

# End-to-end: solving via the fast structured CDF+POWER callback must reproduce the same solution as dense
K_cps, x_cps, ns_cps, _, _ = inner_loop_internal_archgeneric(aug_cp.obj_cm, θ_full0;
    hess_cb_builder = _obj -> archC_frechet_cdf_power_hess_cb_builder(fctx_cp))
check("CDF+POWER structured-solve feasible", ns_cps in (0,-100,-101,-103))
base_dense = BaseDualState(collect(x_free_calib), θ_full0, x_cp[1], collect(x_cp[2:end]), copy(aug_cp.obj_cm.arg1), ns_cp)
base_struct = BaseDualState(collect(x_free_calib), θ_full0, x_cps[1], collect(x_cps[2:end]), copy(aug_cp.obj_cm.arg1), ns_cps)
Delta_dense = delta_dual_from_base(aug_cp.obj_cm, base_dense)
Delta_struct = delta_dual_from_base(aug_cp.obj_cm, base_struct)
checkapprox("Delta*: CDF+POWER dense-solve == structured-solve", Delta_dense, Delta_struct; atol=1e-6, rtol=1e-6)

# ============================================================================
println("="^100); println("GATE P3: nested Delta* ordering  flexible <= frechet(cdf_only) <= frechet(cdf_power)"); println("="^100)
# ============================================================================
pcx_flex = build_cm_production_context(ctx, CS; L = L, contrasts = :orthonormal)
base_flex = archC_base_state(x_free_calib, pcx_flex.ctx_cm, pcx_flex.cctx)
Delta_flex = delta_dual_from_base(pcx_flex.ctx_cm.obj, base_flex)

Delta_cdf_only = delta_dual_from_base(aug_cdf.obj_cm, BaseDualState(collect(x_free_calib), θ_full0, x_cdf[1], collect(x_cdf[2:end]), copy(aug_cdf.obj_cm.arg1), ns_cdf))
@printf "Delta*_flexible           = %.10f\n" Delta_flex
@printf "Delta*_frechet(cdf_only)  = %.10f\n" Delta_cdf_only
@printf "Delta*_frechet(cdf_power) = %.10f\n" Delta_dense
check("Delta*_flexible <= Delta*_frechet(cdf_only) (+1e-8 slack)", Delta_flex <= Delta_cdf_only + 1e-8)
check("Delta*_frechet(cdf_only) <= Delta*_frechet(cdf_power) (+1e-8 slack)", Delta_cdf_only <= Delta_dense + 1e-8)

println()
println("="^100)
@printf "TOTAL: %d PASS, %d FAIL\n" n_pass n_fail
println("="^100)
exit(n_fail == 0 ? 0 : 1)
