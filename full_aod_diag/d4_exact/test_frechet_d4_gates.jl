# ============================================================================
# Fixed-Fréchet-marginals D=4 correctness gate battery (task brief §10.1).
# Mirrors cm_cplus_expanded_d4_battery.jl's include order/setup pattern.
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
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_frechet_checkpoint.jl"))
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

println("="^100)
println("SETUP")
println("="^100)
ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2; W = size(ctx.U, 1)
const L = 8
x_free_calib = ctx.θ0_up[ctx.free_idx]
println("D=$D W=$W L=$L")

cfg_flex = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured), marginal_mode = :common_flexible)
cfg_frec = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured), marginal_mode = :frechet_reference)

# ============================================================================
println("="^100); println("GATE 0: target construction"); println("="^100)
# ============================================================================
targets = build_frechet_reference_targets(ctx, cfg_frec; L = L)
targets2 = build_frechet_reference_targets(ctx, cfg_frec; L = L)
check("targets deterministic (sha256 stable across rebuild)", targets.target_sha256 == targets2.target_sha256)
check("t_l* == p_l exactly (CDF feature)", targets.targets == targets.probs)
check("thresholds strictly increasing", all(diff(targets.thresholds) .> 0))
check("thresholds == -log(1-p) analytically", all(isapprox.(targets.thresholds, -log1p.(-targets.probs); atol=1e-14)))
check("theta_star retrieved from ctx (not hardcoded), matches 1/ctx.μHat", targets.theta_star == 1/ctx.μHat)
check("sigma retrieved from ctx, matches ctx.σ", targets.sigma == ctx.σ)
println("theta_star=$(targets.theta_star)  sigma=$(targets.sigma)")
println("probs=$(targets.probs)")
println("thresholds=$(targets.thresholds)")

# independent numerical-integration check of the (dormant) power target via a fine Riemann sum
function riemann_power_target(pw, u; n=2_000_000)
    h = u / n
    s = 0.0
    @inbounds for i in 1:n
        t = (i - 0.5) * h
        s += t^pw * exp(-t) * h
    end
    return s
end
pw = (1 - targets.sigma) / targets.theta_star
for (i, u) in enumerate(targets.thresholds)
    rt = riemann_power_target(pw, u)
    checkapprox("power_target[$i] matches independent Riemann-sum integration", targets.power_targets[i], rt; atol=1e-6, rtol=1e-5)
end

# ============================================================================
println("="^100); println("GATE 1: transformed vs. naive dense equivalence (Delta_dual)"); println("="^100)
# ============================================================================
"Naive dense D*L-column construction: DIRECT per-origin pin h_l(U_o)-t_l*=0 for every o=1..D (no contrast/common split)."
function naive_frechet_cm(U::Matrix{Float64}, targets::FrechetReferenceTargets, D::Int)
    W = size(U, 1); Lc = length(targets.probs)
    z = targets.thresholds; t = targets.targets
    CM = Matrix{Float64}(undef, W, D * Lc)
    @inbounds for l in 1:Lc
        for o in 1:D
            CM[:, (l-1)*D+o] = (U[:, o] .<= z[l]) .- t[l]
        end
    end
    return CM
end

augA = build_cm_frechet_augmented_obj(ctx, CS, targets; contrasts = :orthonormal)
CM_naive = naive_frechet_cm(ctx.U, targets, D)
obj0 = ctx.obj
ncore = obj0.d
moments_naive! = wrap_moments_with_cm(obj0.moments!, ncore, CM_naive)
obj_naive = CS.PsiObjectiveBundleImplicit(δ = obj0.δ, find_smallest = obj0.find_smallest, γ = obj0.γ,
    (moments!) = moments_naive!, moments_jacobian! = error, d = ncore + size(CM_naive,2),
    outer_constr_index = obj0.outer_constr_index + size(CM_naive,2),
    inequality_index = obj0.inequality_index, complement_index = obj0.complement_index,
    l = obj0.l, U = obj0.U, N = obj0.N, lower_limit = obj0.lower_limit, use_cached_x = obj0.use_cached_x,
    outer_loop_opt = obj0.outer_loop_opt, inner_loop_opt = obj0.inner_loop_opt,
    needs_outer_moment_jacobian = obj0.needs_outer_moment_jacobian)
check("naive column count == D*L == transformed ncm", size(CM_naive,2) == augA.ncm)

θ_full0 = CS.reconstruct_full(x_free_calib, ctx.m)
K_A, xA, nStatusA, _, _ = inner_loop_internal_archgeneric(augA.obj_cm, θ_full0; hess_cb_builder = archA_hess_cb_builder)
K_N, xN, nStatusN, _, _ = inner_loop_internal_archgeneric(obj_naive, θ_full0; hess_cb_builder = archA_hess_cb_builder)
check("transformed inner solve feasible", nStatusA in (0,-100,-101,-103))
check("naive inner solve feasible", nStatusN in (0,-100,-101,-103))
baseA = BaseDualState(collect(x_free_calib), θ_full0, xA[1], collect(xA[2:end]), copy(augA.obj_cm.arg1), nStatusA)
baseN = BaseDualState(collect(x_free_calib), θ_full0, xN[1], collect(xN[2:end]), copy(obj_naive.arg1), nStatusN)
Delta_A = delta_dual_from_base(augA.obj_cm, baseA)
Delta_N = delta_dual_from_base(obj_naive, baseN)
checkapprox("Delta_dual: transformed == naive dense", Delta_A, Delta_N; atol=1e-9, rtol=1e-9)

# ============================================================================
println("="^100); println("GATE 2: dense (Architecture A) vs structured (Architecture C) Hessian"); println("="^100)
# ============================================================================
augB = build_cm_frechet_augmented_obj_archB(ctx, CS, targets; contrasts = :orthonormal)
fctx = build_cm_frechet_bin_ctx(ctx, augB)
check("Architecture A / B ncm agree", augA.ncm == augB.ncm)

n_inner = augA.ncore + augA.ncm
nh = div(n_inner*(n_inner+1), 2)
h_dense = zeros(nh)
augA.obj_cm(xA, h = h_dense)

# populate augB.obj_cm's H buffer at the SAME theta, then compute structured Hessian at the SAME x=xA
augB.obj_cm.moments!(@view(augB.obj_cm.H[:,1]), CS.select_G_from_H(augB.obj_cm, augB.obj_cm.H), θ_full0, augB.obj_cm.U, augB.obj_cm)
augB.obj_cm.H[:,2] .= 1.0
_archC_prep_for_hessian!(augB.obj_cm, xA)
h_struct = zeros(nh)
hessian_cm_frechet_structured!(h_struct, augB.obj_cm, fctx)

maxerr_h = maximum(abs.(h_dense .- h_struct))
relerr_h = maxerr_h / (maximum(abs.(h_dense)) + 1e-300)
println("max|H_dense - H_struct| = $maxerr_h   (relative to max|H_dense|=$(maximum(abs.(h_dense))): $relerr_h)")
check("dense vs structured Hessian match to 1e-8 abs", maxerr_h < 1e-8)

# ============================================================================
println("="^100); println("GATE 3: fixed-point nesting  Delta*_unrestricted <= Delta*_flexible <= Delta*_frechet"); println("="^100)
# ============================================================================
K_U, xU, nStatusU, _, _ = inner_loop_internal_archgeneric(ctx.obj, θ_full0; hess_cb_builder = archA_hess_cb_builder)
baseU = BaseDualState(collect(x_free_calib), θ_full0, xU[1], collect(xU[2:end]), copy(ctx.obj.arg1), nStatusU)
Delta_U = delta_dual_from_base(ctx.obj, baseU)

# archC_base_state needs a CMBinHessCtx; build_cm_production_context (v1) returns one directly.
pcx_flex_v1 = build_cm_production_context(ctx, CS; L = L, contrasts = :orthonormal)
base_flex = archC_base_state(x_free_calib, pcx_flex_v1.ctx_cm, pcx_flex_v1.cctx)
Delta_flex = delta_dual_from_base(pcx_flex_v1.ctx_cm.obj, base_flex)

ctx_cm_B = merge(ctx, (obj = augB.obj_cm,))
baseA_c = archC_frechet_base_state(x_free_calib, ctx_cm_B, fctx)
Delta_frec = delta_dual_from_base(augB.obj_cm, baseA_c)

@printf "Delta*_unrestricted = %.10f\n" Delta_U
@printf "Delta*_flexible     = %.10f\n" Delta_flex
@printf "Delta*_frechet      = %.10f\n" Delta_frec
check("Delta*_unrestricted <= Delta*_flexible (+1e-8 slack)", Delta_U <= Delta_flex + 1e-8)
check("Delta*_flexible <= Delta*_frechet (+1e-8 slack)", Delta_flex <= Delta_frec + 1e-8)
checkapprox("Delta*_frechet (archB/archC) == Delta*_frechet (archA dense, Gate 1)", Delta_frec, Delta_A; atol=1e-8, rtol=1e-8)

# ============================================================================
println("="^100); println("GATE 4: C+ vs Reference complete outer gradient"); println("="^100)
# ============================================================================
pool = build_grad_workspace_pool(W)
ws = build_lfix_factorized_workspace(D, W)
fpcx = build_cm_frechet_production_context(ctx, CS, cfg_frec; L = L)
check("fpcx.mode == :frechet_reference", fpcx.mode === :frechet_reference)
check("fpcx.fctx !== nothing (structured backend)", fpcx.fctx !== nothing)

for h0 in [0.05, 0.01, 0.005]
    g_ref, _ = cm_frechet_production_gradient(x_free_calib, fpcx, ctx, pe; threaded = false, h_mode = :fixed, h0 = h0)
    g_cp, _ = cm_frechet_production_gradient_cplus(x_free_calib, fpcx, ctx, pe, pool, ws; threaded = false, h_mode = :fixed, h0 = h0)
    maxerr = maximum(abs.(g_ref .- g_cp))
    cosang = dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
    @printf "  h=%.4f  max|Δg|=%.3e  cosine=%.10f\n" h0 maxerr cosang
    check("[h=$h0] Reference/C+ full-gradient max|Δg| < 1e-6", maxerr < 1e-6)
    check("[h=$h0] Reference/C+ full-gradient cosine > 1-1e-8", cosang > 1 - 1e-8)
end

# ============================================================================
println("="^100); println("GATE 5: checkpoint round-trip + config-mismatch refusal"); println("="^100)
# ============================================================================
ckpt_ctx = cm_frechet_checkpoint_context(cfg_frec, targets)
check("checkpoint context marginal_mode correct", ckpt_ctx.marginal_mode === :frechet_reference)

ckpt = CMCheckpointV5(CM_FRECHET_CHECKPOINT_SCHEMA, "test_run", "test_label", :test, true, 1.0, W, 12345,
    :test_design, "duc", "dtc", L, targets.probs, :orthonormal, :equal, :cumulative, :structured, :cplus,
    :cm_only, 0, 0, :direct, 1, 0.0, x_free_calib, Float64[], zeros(1,1), Float64[], Dict{Int,Float64}(),
    nothing, 0, 0, 0.0, 0.0, :test, "13.x",
    ckpt_ctx.marginal_mode, ckpt_ctx.theta_star, ckpt_ctx.scale, ckpt_ctx.sigma, ckpt_ctx.probs,
    ckpt_ctx.thresholds_checksum, ckpt_ctx.target_checksum, ckpt_ctx.feature_layout_version)

tmpckpt = joinpath(mktempdir(), "test_frechet.ckpt")
save_cm_frechet_checkpoint(tmpckpt, ckpt)
ckpt_loaded = load_cm_frechet_checkpoint(tmpckpt)
check("checkpoint round-trip: marginal_mode", ckpt_loaded.marginal_mode == ckpt.marginal_mode)
check("checkpoint round-trip: frechet_theta_star", ckpt_loaded.frechet_theta_star == ckpt.frechet_theta_star)
check("checkpoint round-trip: frechet_target_checksum", ckpt_loaded.frechet_target_checksum == ckpt.frechet_target_checksum)
check("checkpoint round-trip: zfree vector", ckpt_loaded.zfree == ckpt.zfree)

reason_ok = cm_frechet_checkpoint_refusal_reason(ckpt_loaded, ckpt_ctx)
check("compatible resume: no refusal reason", reason_ok === nothing)

req_mode_mismatch = (marginal_mode = :common_flexible, theta_star=NaN, scale=NaN, sigma=NaN, probs=Float64[], thresholds_checksum="", target_checksum="", feature_layout_version=0)
check("refuses on marginal_mode mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_mode_mismatch) !== nothing)

req_theta_mismatch = merge(ckpt_ctx, (theta_star = ckpt_ctx.theta_star + 1.0,))
check("refuses on theta_star mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_theta_mismatch) !== nothing)

req_sigma_mismatch = merge(ckpt_ctx, (sigma = ckpt_ctx.sigma + 0.1,))
check("refuses on sigma mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_sigma_mismatch) !== nothing)

req_probs_mismatch = merge(ckpt_ctx, (probs = ckpt_ctx.probs .+ 1e-6,))
check("refuses on grid-probability mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_probs_mismatch) !== nothing)

req_target_mismatch = merge(ckpt_ctx, (target_checksum = "deadbeef",))
check("refuses on target-checksum mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_target_mismatch) !== nothing)

req_layout_mismatch = merge(ckpt_ctx, (feature_layout_version = ckpt_ctx.feature_layout_version + 1,))
check("refuses on feature-layout-version mismatch", cm_frechet_checkpoint_refusal_reason(ckpt_loaded, req_layout_mismatch) !== nothing)

# upgrade path: a schema-4 checkpoint (marginal_mode field didn't exist) must upgrade to :common_flexible
ckpt4 = CMCheckpointV4(4, "old_run", "old", :test, true, 1.0, W, 1, :d, "u","t", L, targets.probs, :orthonormal,
    :equal, :cumulative, :structured, :cplus, :cm_only, 0, 0, :direct, 1, 0.0, x_free_calib, Float64[],
    zeros(1,1), Float64[], Dict{Int,Float64}(), nothing, 0, 0, 0.0, 0.0, :test, "13.x")
tmpckpt4 = joinpath(mktempdir(), "test_v4.ckpt")
save_cm_checkpoint(tmpckpt4, ckpt4)
upgraded = load_cm_frechet_checkpoint(tmpckpt4)
check("schema-4 upgrade: marginal_mode=:common_flexible", upgraded.marginal_mode === :common_flexible)
check("schema-4 upgrade: schema bumped to 5", upgraded.schema == CM_FRECHET_CHECKPOINT_SCHEMA)

# ============================================================================
println("="^100); println("GATE 6: flexible-CM regression (existing battery unchanged)"); println("="^100)
# ============================================================================
# build_cm_frechet_production_context(:common_flexible) must delegate byte-identically
fpcx_flex = build_cm_frechet_production_context(ctx, CS, cfg_flex; L = L)
check("delegated context mode == :common_flexible", fpcx_flex.mode === :common_flexible)
check("delegated context targets === nothing", fpcx_flex.targets === nothing)
base_deleg = archC_base_state(x_free_calib, fpcx_flex.ctx_cm, pcx_flex_v1.cctx)
Delta_deleg = delta_dual_from_base(fpcx_flex.ctx_cm.obj, base_deleg)
checkapprox(":common_flexible delegation reproduces existing flexible Delta_dual", Delta_deleg, Delta_flex; atol=1e-9, rtol=1e-9)

println()
println("="^100)
@printf "TOTAL: %d PASS, %d FAIL\n" n_pass n_fail
println("="^100)
exit(n_fail == 0 ? 0 : 1)
