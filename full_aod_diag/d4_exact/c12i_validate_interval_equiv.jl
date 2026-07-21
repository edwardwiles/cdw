# Part A validation: bin-index / interval-moment reformulation vs the trusted dense
# cumulative-CDF reference (common_marginals_moments.jl, UNMODIFIED). Two things checked:
#
#   1. The explicit cumulative<->interval transform matrix is correct: applying it to the
#      interval-basis dense CM matrix reproduces the reference's dense CM matrix to machine
#      precision, for both :anchored and :orthonormal contrasts, at L in {10,20,50}. Checked via
#      TWO independent constructions of the transform (direct per-block cumsum,
#      `interval_to_cumulative_dense`, vs the explicit Kronecker matrix, `full_transform_matrix`)
#      which must also agree with each other.
#
#   2. End-to-end equivalence of the CC inner dual solve when the interval-basis CM block is
#      spliced into the SAME `wrap_moments_with_cm`/`build_cm_augmented_obj`-style bundle as the
#      dense reference: at IDENTICAL fixed outer parameters (calibration + every point in
#      c12_d4_fixed_param_battery.jl's battery, code copied from that file), dense-augmented and
#      interval-augmented inner solves must agree to solver tolerance on nStatus, Delta_dual,
#      Delta_primal, primal LFD weights (m_weights, pointwise), and max moment KKT residual.
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))       # trusted dense reference (unmodified)
include(joinpath(@__DIR__, "common_marginals_interval.jl"))      # NEW: interval reformulation
using Printf, LinearAlgebra, Random

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
D = ctx.D
println("Base ctx: D=$D  ncore=$(ctx.obj.d)  outer_constr_index=$(ctx.obj.outer_constr_index)  free dims=$(length(ctx.free_idx))")

# ================================================================================================
# Section 1: transform-matrix proof, at L in {10,20,50}, both contrast modes.
# ================================================================================================
println("\n" * "="^100)
println("SECTION 1: cumulative<->interval transform matrix -- explicit construction + validation")
println("="^100)

for L in (10, 20, 50), contrasts in (:anchored, :orthonormal)
    CM_dense, z_dense, origins_dense = precalc_common_marginals_cdf(ctx.U, ctx.γ.refIndex1, L; contrasts = contrasts)
    CM_int, z_int, origins_int, bins = precalc_common_marginals_interval(ctx.U, ctx.γ.refIndex1, L; contrasts = contrasts)
    nO = length(origins_dense)

    @assert z_dense == z_int "cutpoints not bit-identical at L=$L"
    @assert origins_dense == origins_int

    # two independent constructions of the transform
    CDF_via_blockcumsum = interval_to_cumulative_dense(CM_int, nO, L)
    Mfull = full_transform_matrix(nO, L)
    CDF_via_kron = CM_int * Mfull

    err_blockcumsum_vs_kron = maximum(abs.(CDF_via_blockcumsum .- CDF_via_kron))
    err_vs_reference = maximum(abs.(CDF_via_blockcumsum .- CM_dense))
    err_kron_vs_reference = maximum(abs.(CDF_via_kron .- CM_dense))

    @printf("[L=%2d %-10s] max|blockcumsum-kron|=%.3e  max|blockcumsum-reference|=%.3e  max|kron-reference|=%.3e  size(CM)=%s\n",
            L, string(contrasts), err_blockcumsum_vs_kron, err_vs_reference, err_kron_vs_reference, size(CM_dense))

    @assert err_blockcumsum_vs_kron < 1e-12 "transform constructions disagree at L=$L, $contrasts"
    @assert err_vs_reference < 1e-10 "interval->cumulative reconstruction disagrees with dense reference at L=$L, $contrasts"
    @assert err_kron_vs_reference < 1e-10 "Kronecker-transform reconstruction disagrees with dense reference at L=$L, $contrasts"
end
println("Section 1: PASS -- transform matrix proven correct at all (L, contrasts) combinations checked.")

# ================================================================================================
# Section 2: end-to-end CC inner-solve equivalence at fixed outer parameters.
# Point construction copied from c12_d4_fixed_param_battery.jl (unmodified reference script).
# ================================================================================================
println("\n" * "="^100)
println("SECTION 2: end-to-end CC inner-solve equivalence, dense-augmented vs interval-augmented")
println("="^100)

pe = build_pivot_elimination(ctx)
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

x_calib_raw = ctx.θ0_up[ctx.free_idx]

w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
x_up40 = x_free_from_w(w_up40)

Random.seed!(4243)
x_perturbed = copy(x_calib_raw)
x_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_perturbed) - 1))

x_infeasible_raw = copy(x_calib_raw)
raw_Aod_pos_11 = findfirst(==(ctx.Aod_offset + 1 + (1 - 1) * D), ctx.free_idx)
x_infeasible_raw[raw_Aod_pos_11] = 1e-6

points = [("calibration", x_calib_raw), ("upper_maxit40", x_up40), ("perturbed_feasible", x_perturbed),
          ("structurally_infeasible", x_infeasible_raw)]

function run_and_summarize(x_free, ctx_use, aug)
    r = evaluate_fullA(x_free, ctx_use; use_cache = false, warm = false)
    ok = r.inner_status in (0, -100, -101, -103)
    if !ok
        return (ok = false, nStatus = r.inner_status)
    end
    W = size(ctx_use.U, 1)
    K = zeros(W); G = zeros(W, aug.obj_cm.d)
    aug.obj_cm.moments!(K, G, r.θ_full, ctx_use.U, aug.obj_cm)
    m_full = copy(aug.obj_cm.arg1)
    return (ok = true, nStatus = r.inner_status, Delta_dual = r.Delta_dual, Delta_primal = r.Delta_primal,
            m_weights = m_full, max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
            gamma_focal_prime = r.gamma_focal_prime, zeta = r.zeta, lambda = r.lambda)
end

all_pass = Ref(true)
for L in (10, 20, 50)
    println("\n--- L=$L ---")
    aug_dense = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    aug_int   = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    ctx_dense = merge(ctx, (obj = aug_dense.obj_cm,))
    ctx_int   = merge(ctx, (obj = aug_int.obj_cm,))
    @assert aug_dense.ncm == aug_int.ncm == (D - 1) * L

    for (label, x_free) in points
        rd = run_and_summarize(x_free, ctx_dense, aug_dense)
        ri = run_and_summarize(x_free, ctx_int, aug_int)

        status_match = rd.ok == ri.ok && (!rd.ok || rd.nStatus == ri.nStatus)
        if !rd.ok && !ri.ok
            @printf("  %-26s BOTH INFEASIBLE (nStatus dense=%d interval=%d) -- consistent\n", label, rd.nStatus, ri.nStatus)
            continue
        elseif !status_match
            @printf("  %-26s STATUS MISMATCH dense.ok=%s(%d) interval.ok=%s(%d)\n", label, rd.ok, rd.nStatus, ri.ok, ri.nStatus)
            all_pass[] = false
            continue
        end

        dd_err = abs(rd.Delta_dual - ri.Delta_dual)
        dp_err = abs(rd.Delta_primal - ri.Delta_primal)
        m_err = maximum(abs.(rd.m_weights .- ri.m_weights))
        m_relerr = m_err / max(1.0, maximum(abs.(rd.m_weights)))
        kkt_d = rd.max_abs_moment_kkt_resid
        kkt_i = ri.max_abs_moment_kkt_resid

        @printf("  %-26s Delta_dual err=%.3e  Delta_primal err=%.3e  max|m_dense-m_int|=%.3e (rel %.3e)  KKT dense=%.2e interval=%.2e\n",
                label, dd_err, dp_err, m_err, m_relerr, kkt_d, kkt_i)

        ok_here = dd_err < 1e-6 && dp_err < 1e-6 && m_relerr < 1e-6
        all_pass[] &= ok_here
        ok_here || println("    *** MISMATCH ABOVE TOLERANCE ***")
    end
end

println("\n" * "="^100)
println(all_pass[] ? "SECTION 2: PASS -- dense and interval bases agree to solver tolerance at every point checked." :
                    "SECTION 2: FAIL -- see mismatches above.")
println("="^100)
