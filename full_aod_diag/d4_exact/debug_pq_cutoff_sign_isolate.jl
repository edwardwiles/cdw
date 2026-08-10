# ================================================================================================
# Isolation diagnostic for the pairwise-quantile cutoff outer gradient's SIGN and ENVELOPE claim.
#
# The aggregate FD gate compares a chain-ruled, multi-cutoff, linearized analytic gradient against a
# large-step reoptimized secant -- too many moving parts to attribute a disagreement. This script
# strips it to the single load-bearing identity, one physical cutoff at a time:
#
#     Delta*(q + dq) - Delta*(q)  ?=  -1 * fixed_dual_delta_f(q -> q + dq)
#
# LHS: two INDEPENDENT full inner KNITRO solves (reoptimized).
# RHS: the exact O(k_crossed) fixed-dual formula the gradient is built on, negated per the
#      Delta_dual = -f convention (operator_verification.jl:512).
#
# Moving ONE physical cutoff (rather than one raw coordinate) is the point: the raw->physical
# transform is lower-triangular, so a single raw coordinate moves EVERY downstream cutoff of that
# origin at once. Here the perturbed raw vector is reconstructed by inverting the softplus transform
# on a hand-edited Q column, so exactly one q_{o,r} moves and the envelope claim is tested on its own.
#
# Usage: julia --project=. full_aod_diag/d4_exact/debug_pq_cutoff_sign_isolate.jl [L] [min_crossed]
# ================================================================================================
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_threaded.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl", "cm_hessian_architectures.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_originzc_target_layout.jl", "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_moments.jl", "cm_originzc_production.jl", "operator_psi_bundle.jl",
          "cm_callback_health.jl", "compressed_factual_buffer_reuse.jl",
          "shared_a_gradient.jl", "operator_verification.jl", "cm_screen_bridge.jl",
          "pairwise_quantile_cutoff_transform.jl", "pairwise_quantile_bin_context.jl",
          "pairwise_quantile_operator.jl", "pairwise_quantile_hessian.jl",
          "pairwise_quantile_cross_hessian.jl", "pairwise_quantile_verification.jl",
          "pairwise_quantile_production.jl", "pairwise_quantile_cutoff_gradient.jl",
          "pairwise_quantile_outer_production.jl"]
    include(joinpath(D4X, f))
end
using Printf, LinearAlgebra

const PQ_L = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const MIN_CROSSED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
layout = PairwiseQuantileCutoffLayout(ctx.D, PQ_L)
pcx = build_pairwise_quantile_production_context(ctx, layout; min_crossed = MIN_CROSSED)
ctx_cm = pcx.ctx_cm
op = ctx_cm.pq_op
W = op.W
nc = n_cutoffs(layout)

function quantile_naive(v::AbstractVector{Float64}, p::Float64)
    s = sort(v); n = length(s); return s[clamp(round(Int, p * n), 1, n)]
end
raw0 = zeros(n_raw(layout))
for o in 1:ctx.D
    b = raw_index(layout, o, 1)
    Uo = @view ctx.U[:, o]
    q = [quantile_naive(Uo, r / PQ_L) for r in 1:PQ_L-1]
    raw0[b] = log(q[1])
    for k in 2:PQ_L-1
        gap = log(q[k]) - log(q[k-1])
        raw0[b+k-1] = gap > 0 ? log(expm1(gap)) : -5.0
    end
end

"Invert the softplus-ordered transform: an ordered physical cutoff column -> that origin's raw block."
function raw_from_Qcol(Qcol::AbstractVector{Float64})
    n = length(Qcol)
    r = zeros(n)
    r[1] = log(Qcol[1])
    for k in 2:n
        gap = log(Qcol[k]) - log(Qcol[k-1])
        gap > 0 || error("raw_from_Qcol: non-increasing cutoffs at k=$k")
        r[k] = log(expm1(gap))
    end
    return r
end

println("=== base solve ===")
base, verify = archPQ_verified_state(x_free_calib, raw0, ctx_cm)
Δ0 = verify.Delta_dual
println("Delta_dual(base) = ", Δ0, "  inner_status = ", verify.inner_status,
        "  class = ", classify_inner_result(verify))
ncore1 = ctx_cm.obj.outer_constr_index - 1 - n_total_rows(op.D, op.L)
λ_M, λ_P = reshape_pq_duals(base.λstar, op, ncore1)
Q0 = copy(ctx_cm.pq_bin_state.Q)

"Move ONLY q_{o,r} to q_new, re-solve from scratch, return Delta_dual."
function resolve_at_single_cutoff(o::Int, r::Int, q_new::Float64)
    Qnew = copy(Q0)
    Qnew[r, o] = q_new
    raw_new = copy(raw0)
    b = raw_index(layout, o, 1)
    raw_new[b:b+nc-1] = raw_from_Qcol(@view Qnew[:, o])
    _, vnew = archPQ_verified_state(x_free_calib, raw_new, ctx_cm)
    refresh_pairwise_quantile_bins!(ctx_cm.pq_bin_state, op, ctx_cm.U, raw0, layout)
    return vnew.Delta_dual
end

# ------------------------------------------------------------------------------------------------
# EXPERIMENT 0: does `fixed_dual_delta_f` agree with a FULL independent fixed-dual recompute done in
# the PRODUCTION sign convention? No theory, no envelope argument, no reoptimization -- just: hold
# (zeta*, lambda*) exactly fixed, rebuild the bin state at the perturbed cutoff, recompute
# r = -zeta - E*lambda_E - G_R*lambda_R through the SAME operators the real solve uses, and compare
# f_new - f_base against what the O(k_crossed) shortcut claims.
#
# This is the one comparison the existing D4 oracle test does NOT make: that test builds its own
# synthetic `r_current = -pairwise_quantile_forward!(zeros, ...)` and evaluates
# `psi_scalar.(-a0)` -- i.e. it negates BOTH the reference r and the recomputed one, so a sign error
# in the shortcut's `dR` cancels against a matching sign error in its own brute-force reference and
# the check passes either way. Here `r_current` comes from the REAL solve, in the real convention.
# ------------------------------------------------------------------------------------------------
println("\n", "="^110)
println("EXPERIMENT 0: fixed_dual_delta_f vs FULL fixed-dual recompute, production sign convention")
println("="^110)
cf_base = ctx_cm.pq_core_cf_ref[]
econ_ws_base = economic_operator_workspace(cf_base)
λ_E_base = @view base.λstar[1:ncore1]

"Full, independent fixed-dual f at an arbitrary cutoff matrix, using the base solve's own duals."
function f_fixed_dual_at(Qtarget::AbstractMatrix{Float64})
    st = PairwiseQuantileBinState(W, ctx.D, PQ_L)
    st.Q .= Qtarget
    for w in 1:W, oo in 1:ctx.D
        st.bin[w, oo] = UInt8(searchsortedfirst(view(Qtarget, :, oo), ctx_cm.U[w, oo]))
    end
    r = fill(-base.ζstar, W)
    buf = zeros(W)
    economic_forward!(buf, λ_E_base, cf_base, econ_ws_base)
    r .-= buf
    pairwise_quantile_forward!(r, λ_M, λ_P, op, st)
    return sum(psi_scalar.(r)) / W + base.ζstar
end

f_base = f_fixed_dual_at(Q0)
println("f_fixed_dual(base) = ", f_base, "   -f = ", -f_base, "   Delta_dual(base) = ", Δ0,
        "   (match? ", isapprox(-f_base, Δ0; rtol = 1e-8), ")")
@printf("\n%3s %3s %8s %18s %18s %10s\n", "o", "r", "k_cross", "shortcut df", "full-recompute df", "ratio")
n_sign_ok = 0; n_sign_tot = 0
for (o, r) in ((1, 1), (1, 3), (2, 2), (3, 4), (4, 3))
    Qcol = @view Q0[:, o]
    (q_up, _) = cutoff_probe_points(op, Qcol, o, r, nc; min_crossed = MIN_CROSSED)
    q_up > Qcol[r] || continue
    (df_short, kx) = fixed_dual_delta_f(op, ctx_cm.pq_bin_state, o, r, q_up, λ_M, λ_P, verify.r_current)
    Qmod = copy(Q0); Qmod[r, o] = q_up
    df_full = f_fixed_dual_at(Qmod) - f_base
    global n_sign_tot += 1
    global n_sign_ok += isapprox(df_short, df_full; rtol = 1e-6)
    @printf("%3d %3d %8d %18.10e %18.10e %10.4f\n", o, r, kx, df_short, df_full,
            df_full == 0 ? NaN : df_short / df_full)
    flush(stdout)
end
println("\n", n_sign_ok, " / ", n_sign_tot, " shortcut values match the full fixed-dual recompute (rtol 1e-6)")
println(n_sign_ok == n_sign_tot ? "SHORTCUT AGREES WITH PRODUCTION CONVENTION" :
        "SHORTCUT DISAGREES WITH PRODUCTION CONVENTION -- see ratio column")

# ------------------------------------------------------------------------------------------------
# EXPERIMENT 1: bandwidth sweep on a few cutoffs, BOTH directions.
#
# If the fixed-dual formula and its sign are right and the aggregate disagreement is purely a
# bandwidth/curvature artifact, then as the probe shrinks the ratio (predicted / actual) must tend
# to 1, and the CENTRAL secant (up minus down) must agree far better than either one-sided probe --
# because a symmetric second-order term cancels in a central difference but not in a one-sided one.
# If instead the ratio stays wrong as the probe shrinks, the formula itself is wrong.
# ------------------------------------------------------------------------------------------------
println("\n", "="^110)
println("EXPERIMENT 1: bandwidth sweep, one-sided AND central, on 3 cutoffs")
println("="^110)
for (o, r) in ((1, 1), (2, 2), (4, 3))
    println("\n--- origin o=$o, cutoff r=$r  (q_old = ", Q0[r, o], ") ---")
    @printf("%8s %8s %14s %14s %10s | %16s %16s %10s\n",
            "min_cr", "k_up/dn", "pred_up", "act_up", "ratio_up", "pred_central", "act_central", "ratio_c")
    for mc in (1, 5, 20, 50, 200)
        Qcol = @view Q0[:, o]
        (q_up, q_dn) = cutoff_probe_points(op, Qcol, o, r, nc; min_crossed = mc)
        (q_up > Qcol[r] && q_dn < Qcol[r]) || continue
        (df_u, k_u) = fixed_dual_delta_f(op, ctx_cm.pq_bin_state, o, r, q_up, λ_M, λ_P, verify.r_current)
        (df_d, k_d) = fixed_dual_delta_f(op, ctx_cm.pq_bin_state, o, r, q_dn, λ_M, λ_P, verify.r_current)
        Δup = resolve_at_single_cutoff(o, r, q_up)
        Δdn = resolve_at_single_cutoff(o, r, q_dn)
        pred_up = -df_u
        act_up = Δup - Δ0
        # central: derivative estimates, then rescaled back to a "change over the up-step" so the two
        # columns are directly comparable in magnitude
        dq_c = q_up - q_dn
        pred_c = -(df_u - df_d) / dq_c
        act_c = (Δup - Δdn) / dq_c
        @printf("%8d %4d/%-4d %14.6e %14.6e %10.4f | %16.8e %16.8e %10.4f\n",
                mc, k_u, k_d, pred_up, act_up, act_up == 0 ? NaN : pred_up / act_up,
                pred_c, act_c, act_c == 0 ? NaN : pred_c / act_c)
        flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# EXPERIMENT 2: is Delta* at a local MINIMUM in the cutoffs at this base point?
# The base cutoffs are the EMPIRICAL quantiles of each origin's own draws, and the restriction's
# targets are the fixed constants 1/L and 1/L^2. So at the base point the restriction is close to
# already-satisfied by the unweighted draws, and moving a cutoff EITHER way should make it harder --
# i.e. Delta* should rise in both directions, and the true first derivative should be near zero.
# If so, the aggregate FD gate was measuring curvature, not a gradient error.
# ------------------------------------------------------------------------------------------------
println("\n", "="^110)
println("EXPERIMENT 2: Delta* up-probe vs down-probe (is the base point a local minimum in q?)")
println("="^110)
@printf("%3s %3s %16s %16s %16s %s\n", "o", "r", "dDelta(up)", "dDelta(down)", "Delta0", "both up?")
n_both_up = 0; n_tot2 = 0
for o in 1:ctx.D, r in 1:nc
    Qcol = @view Q0[:, o]
    (q_up, q_dn) = cutoff_probe_points(op, Qcol, o, r, nc; min_crossed = MIN_CROSSED)
    (q_up > Qcol[r] && q_dn < Qcol[r]) || continue
    du = resolve_at_single_cutoff(o, r, q_up) - Δ0
    dd = resolve_at_single_cutoff(o, r, q_dn) - Δ0
    both = du > 0 && dd > 0
    global n_tot2 += 1
    global n_both_up += both
    @printf("%3d %3d %16.8e %16.8e %16.8e %s\n", o, r, du, dd, Δ0, both ? "YES" : "no")
    flush(stdout)
end
println("\n", n_both_up, " / ", n_tot2, " cutoffs have Delta* rising in BOTH directions",
        n_both_up == n_tot2 ? "  -> base point IS a local minimum in every cutoff" : "")
