# Continuation 12, section 2/12: validate the dense cumulative-CDF common-marginals reference
# at D=4, fixed outer parameters. Checks moment count, inner KNITRO dual solve status, CM-block
# KKT residuals (the quantity that should be ~0 at a converged LFD solve -- sum(m*G_j)/W, m the
# LFD weight -- not the raw unweighted benchmark_unweighted_moment_mean, which need not vanish), and timing, for
# L in {10, 20, 50} at the calibration point (A_od=1).
#
# Layout note (see common_marginals_moments.jl docstrings): the augmented G has columns
# [1 : ncore-1] = pre-gravity core moments, [ncore : ncore+ncm-1] = CM block, [end] = gravity
# (relocated). ncore == obj0.d from BEFORE augmentation (18 at D=4).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
using Printf

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
println("Base ctx: D=$(ctx.D)  ncore=$(ctx.obj.d)  outer_constr_index=$(ctx.obj.outer_constr_index)  free dims=$(length(ctx.free_idx))")

x_free_calib = ctx.θ0_up[ctx.free_idx]
println("x_free_calib (γ'_focal, then vec(A_od)) length = ", length(x_free_calib))

function run_one(L; contrasts = :anchored, x_free = x_free_calib, label = "calib")
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = contrasts)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    @assert aug.ncm == (ctx.D - 1) * L
    @assert aug.obj_cm.d == ctx.obj.d + aug.ncm
    @assert aug.obj_cm.outer_constr_index == aug.obj_cm.d

    t0 = time()
    r = evaluate_fullA(x_free, ctx_cm; use_cache = false, warm = false)
    telapsed = time() - t0

    if r.inner_status ∉ (0, -100, -101, -103)
        @printf("[L=%2d %-10s %-5s] INNER SOLVE FAILED nStatus=%d  t=%.2fs\n", L, string(contrasts), label, r.inner_status, telapsed)
        return (L = L, contrasts = contrasts, label = label, ok = false, nStatus = r.inner_status, telapsed = telapsed)
    end

    # obj.arg1 holds dPsi(arg0) = m(s) (un-normalized LFD weight) as a side effect of the
    # evaluate_fullA -> obj(inner_x, constr=...) call chain (same trick oracle.jl itself uses;
    # see its Delta_dual/m_weights comments) -- no need to recompute arg0/dPsi by hand.
    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, aug.obj_cm.d)
    aug.obj_cm.moments!(K, G, r.θ_full, ctx.U, aug.obj_cm)
    m_full = copy(aug.obj_cm.arg1)

    ncore = aug.ncore   # = obj0.d before augmentation; CM occupies [ncore : ncore+ncm-1], gravity at end
    cm_cols = ncore:(ncore + aug.ncm - 1)
    core_cols = 1:(ncore - 1)   # pre-gravity core inner moments (excludes gravity, the sole outer-only col)

    kkt(j) = abs(sum(m_full .* G[:, j]) / W)
    cm_kkt = [kkt(j) for j in cm_cols]
    core_kkt = [kkt(j) for j in core_cols]

    cm_mean = vec(sum(m_full .* G[:, cm_cols], dims = 1)) ./ sum(m_full)
    eq35 = cm_block_to_anchored_residuals(cm_mean, ctx.D, L, ctx.D - 1; contrasts = contrasts)

    @printf("[L=%2d %-10s %-5s] nStatus=%d  t=%.2fs  max|core KKT|=%.2e  max|CM KKT|=%.2e  max|anchored CM resid|=%.2e  Delta_dual=%.6f  gamma'=%.6f\n",
            L, string(contrasts), label, r.inner_status, telapsed, maximum(core_kkt), maximum(cm_kkt), maximum(abs.(eq35)), r.Delta_dual, r.gamma_focal_prime)
    return (L = L, contrasts = contrasts, label = label, ok = true, nStatus = r.inner_status, telapsed = telapsed,
            max_core_kkt = maximum(core_kkt), max_cm_kkt = maximum(cm_kkt), max_anchored_resid = maximum(abs.(eq35)),
            Delta_dual = r.Delta_dual, gamma_focal_prime = r.gamma_focal_prime)
end

println("\n=== Calibration point (A_od=1), anchored contrasts ===")
results_anchored = [run_one(L; contrasts = :anchored, x_free = x_free_calib, label = "calib") for L in (10, 20, 50)]

println("\n=== Calibration point (A_od=1), orthonormal contrasts (max anchored-resid should match anchored run to solver tol) ===")
results_ortho = [run_one(L; contrasts = :orthonormal, x_free = x_free_calib, label = "calib") for L in (10, 20, 50)]

println("\n=== Sanity: no-CM baseline (L effectively 0) still solves ===")
r0 = evaluate_fullA(x_free_calib, ctx; use_cache = false, warm = false)
@printf("baseline: nStatus=%d  Delta_dual=%.6f  gamma'=%.6f  max_abs_moment_kkt_resid=%.2e\n",
        r0.inner_status, r0.Delta_dual, r0.gamma_focal_prime, r0.max_abs_moment_kkt_resid)
