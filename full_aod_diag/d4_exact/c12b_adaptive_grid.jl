# ================================================================================================
# Continuation 12b, Part 2: adaptive quantile activation (fixed outer point).
#
# Starting from a small "seed" active set of quantile thresholds (probability levels near
# {0.10,0.25,0.50,0.75,0.90,0.95,0.99}, picked as the CLOSEST probability levels actually
# present in a dense L_full-point candidate grid -- so the active set is a genuine, exactly
# nested subset of the L_full grid's own thresholds, not a separately-quantiled coarse grid),
# solves the CC inner problem with only the active common-marginals restrictions, evaluates ALL
# L_full candidate thresholds' weighted-mean discrepancy at the solved LFD weights, activates
# every inactive threshold whose discrepancy exceeds tolerance, warm-restarts (padding the
# previous (zeta,lambda) solution with zeros for newly-added columns -- valid because new
# columns are always APPENDED after existing ones, so the old solution is an exact prefix of the
# new problem's variable vector) and repeats until every candidate discrepancy is within
# tolerance.
#
# See docs/fullA_cm_conditioning_and_adaptive_grid_report.md for the write-up.
# ================================================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "c12b_interval_common_marginals_moments.jl"))
using Printf, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
x_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(4243)
x_perturbed = copy(x_calib)
x_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_perturbed) - 1))

const TARGET_PROBS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
const TOL = 1e-4   # see report for justification: CM columns are indicator differences bounded
                    # in [-1,1], so this is simultaneously an absolute AND relative-to-moment-scale
                    # tolerance; active (satisfied) restrictions typically show KKT residuals many
                    # orders of magnitude below this (~1e-9-1e-12), so 1e-4 cleanly separates
                    # "genuinely still violated" from "converged to solver precision".

"""
    adaptive_grid(ctx, xf; L_full=50, ref=ctx.γ.refIndex1, tol=TOL, warm_restart=true, verbose=true)

Fixed-outer-point adaptive quantile activation. Returns a NamedTuple with the round-by-round
history and the final active set / obj_cm / result.
"""
function adaptive_grid(ctx, xf; L_full::Int = 50, ref::Int = ctx.γ.refIndex1, tol::Float64 = TOL,
                        warm_restart::Bool = true, verbose::Bool = true, max_rounds::Int = 30)
    CM_full, z_full, origins = precalc_common_marginals_cdf(ctx.U, ref, L_full; contrasts = :anchored)
    nO = length(origins)
    block_cols(l) = (l - 1) * nO + 1 : l * nO

    p_full = collect(range(1 / L_full, (L_full - 1) / L_full, length = L_full))
    seed = unique([argmin(abs.(p_full .- p)) for p in TARGET_PROBS])
    active = copy(seed)   # insertion-order list of threshold indices into 1:L_full

    history = NamedTuple[]
    prev_full_x = nothing
    prev_ncols = 0
    r = nothing
    obj_cm = nothing
    for round in 1:max_rounds
        cols = vcat([collect(block_cols(l)) for l in active]...)
        CM_active = CM_full[:, cols]
        obj_cm = build_cm_augmented_obj_from_CM(ctx, CS, CM_active)

        used_warm = false
        if warm_restart && prev_full_x !== nothing
            # prev_full_x = [zeta_old; lambda_old] has length 1+pregrav+prev_ncols (prev_ncols =
            # number of ACTIVE CM columns last round, NOT length(prev_full_x) itself -- the old
            # CM columns are always a strict PREFIX of `cols` because `active` only ever grows by
            # appending new threshold indices at the end, so padding with zeros for exactly the
            # newly-added CM columns (length(cols) - prev_ncols of them) reproduces the old
            # (zeta,lambda) exactly in the new, larger variable vector.
            nadded = length(cols) - prev_ncols
            @assert nadded >= 0
            obj_cm.x .= vcat(prev_full_x, zeros(nadded))
            used_warm = true
        end
        iters0 = CS.INNER_ITERS_TOTAL[]
        t0 = time()
        r = evaluate_fullA(xf, merge(ctx, (obj = obj_cm,)); use_cache = false, warm = used_warm)
        wall = time() - t0
        iters = CS.INNER_ITERS_TOTAL[] - iters0
        ok = r.inner_status in (0, -100, -101, -103)

        if !ok
            push!(history, (round = round, n_active_thresh = length(active), n_active_cols = length(cols),
                             ok = false, nStatus = r.inner_status, iters = iters, wall = wall,
                             max_inactive_violation = NaN, worst_inactive_l = -1, Delta_dual = NaN,
                             used_warm = used_warm))
            verbose && @printf("round %2d FAILED nStatus=%d (n_active_thresh=%d)\n", round, r.inner_status, length(active))
            break
        end

        m_full = copy(obj_cm.arg1)
        sm = sum(m_full)
        discrep = zeros(L_full)
        for l in 1:L_full
            colsl = block_cols(l)
            discrep[l] = maximum(abs.(vec(sum(m_full .* CM_full[:, colsl], dims = 1)) ./ sm))
        end
        inactive = setdiff(1:L_full, active)
        if isempty(inactive)
            max_inactive_violation = 0.0
            worst_l = -1
        else
            max_inactive_violation, idx = findmax(discrep[inactive])
            worst_l = inactive[idx]
        end

        push!(history, (round = round, n_active_thresh = length(active), n_active_cols = length(cols),
                         ok = true, nStatus = r.inner_status, iters = iters, wall = wall,
                         max_inactive_violation = max_inactive_violation, worst_inactive_l = worst_l,
                         Delta_dual = r.Delta_dual, used_warm = used_warm))
        verbose && @printf("round %2d  n_active_thresh=%-3d n_active_cols=%-4d  iters=%-4d  t=%.2fs  warm=%-5s  max_inactive_violation=%.3e (l=%d)  Delta_dual=%.6f\n",
                round, length(active), length(cols), iters, wall, used_warm, max_inactive_violation, worst_l, r.Delta_dual)

        if isempty(inactive) || max_inactive_violation <= tol
            break
        end

        to_add = [l for l in inactive if discrep[l] > tol]
        append!(active, to_add)
        prev_full_x = vcat(r.zeta, r.lambda)
        prev_ncols = length(cols)
    end

    return (history = history, active = active, obj_cm = obj_cm, r = r, CM_full = CM_full,
            z_full = z_full, origins = origins, L_full = L_full, ref = ref, tol = tol)
end

function verify_against_dense(ctx, xf, res; contrasts = :anchored)
    aug = build_cm_augmented_obj(ctx, CS; L = res.L_full, contrasts = contrasts, refIndex1 = res.ref)
    ctx_dense = merge(ctx, (obj = aug.obj_cm,))
    r_dense = evaluate_fullA(xf, ctx_dense; use_cache = false, warm = false)
    ok_dense = r_dense.inner_status in (0, -100, -101, -103)

    adaptive_ok = res.r.inner_status in (0, -100, -101, -103)
    println("  dense L=$(res.L_full):    nStatus=$(r_dense.inner_status) ok=$ok_dense  Delta_dual=$(r_dense.Delta_dual)")
    println("  adaptive (final round): nStatus=$(res.r.inner_status) ok=$adaptive_ok  Delta_dual=$(res.r.Delta_dual)")
    if ok_dense && adaptive_ok
        ddiff = abs(r_dense.Delta_dual - res.r.Delta_dual)
        println("  |Delta_dual diff| = ", ddiff)
        # LFD weight comparison: both keyed to the SAME W draws, compare m(s)/sum(m(s)) directly
        m_dense = copy(aug.obj_cm.arg1); p_dense = m_dense ./ sum(m_dense)
        m_adapt = copy(res.obj_cm.arg1); p_adapt = m_adapt ./ sum(m_adapt)
        wdiff = maximum(abs.(p_dense .- p_adapt))
        println("  max|p_dense - p_adaptive| (LFD weights) = ", wdiff)
        return (Delta_diff = ddiff, weight_diff = wdiff, ok_dense = ok_dense, adaptive_ok = adaptive_ok)
    end
    return (Delta_diff = NaN, weight_diff = NaN, ok_dense = ok_dense, adaptive_ok = adaptive_ok)
end

println("=" ^ 100)
println("ADAPTIVE GRID: calibration point, L_full=50, ref=", ctx.γ.refIndex1)
println("=" ^ 100)
res_calib_50 = adaptive_grid(ctx, x_calib; L_full = 50)
println("\n--- Verification vs dense L=50 (calibration) ---")
vcheck_calib = verify_against_dense(ctx, x_calib, res_calib_50)

println("\n" * "=" ^ 100)
println("ADAPTIVE GRID: perturbed-feasible point, L_full=50, ref=", ctx.γ.refIndex1)
println("=" ^ 100)
res_pert_50 = adaptive_grid(ctx, x_perturbed; L_full = 50)
println("\n--- Verification vs dense L=50 (perturbed) ---")
vcheck_pert = verify_against_dense(ctx, x_perturbed, res_pert_50)

println("\n" * "=" ^ 100)
println("ROBUSTNESS CHECK: L_full=20, calibration point")
println("=" ^ 100)
res_calib_20 = adaptive_grid(ctx, x_calib; L_full = 20)
println("\n--- Verification vs dense L=20 (calibration) ---")
vcheck_calib_20 = verify_against_dense(ctx, x_calib, res_calib_20)

println("\n" * "=" ^ 100)
println("WARM VS COLD comparison at the FINAL active set (calibration, L_full=50) -- re-solve from scratch")
println("=" ^ 100)
let active = res_calib_50.active, L_full = 50
    CM_full = res_calib_50.CM_full
    nO = length(res_calib_50.origins)
    block_cols(l) = (l - 1) * nO + 1 : l * nO
    cols = vcat([collect(block_cols(l)) for l in active]...)
    CM_active = CM_full[:, cols]
    obj_cold = build_cm_augmented_obj_from_CM(ctx, CS, CM_active)
    iters0 = CS.INNER_ITERS_TOTAL[]
    t0 = time()
    r_cold = evaluate_fullA(x_calib, merge(ctx, (obj = obj_cold,)); use_cache = false, warm = false)
    println("cold-restart at final active set: nStatus=", r_cold.inner_status, " iters=", CS.INNER_ITERS_TOTAL[] - iters0, " t=", time() - t0, "s  Delta_dual=", r_cold.Delta_dual)
    last_hist = res_calib_50.history[end]
    println("adaptive last round (warm=", last_hist.used_warm, "): iters=", last_hist.iters, " t=", last_hist.wall, "s  Delta_dual=", last_hist.Delta_dual)
end

println("\nDONE")
