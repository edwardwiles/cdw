# Phase 7: concrete convex-hull geometry diagnosis for origin 14's infeasibility immediately
# after the first negative-direction switch (t~8.41e-3, cell (o=14,d=19)).
REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using Printf, DelimitedFiles, LinearAlgebra, Random, Serialization
melitz_thread_startup_report()
const OUTDIR = joinpath(REPO2, "docs", "key_results")
const SCRATCH = @__DIR__
CAP = 10.0
policy_cap = CappedEvaluation(CAP)

st = deserialize(joinpath(SCRATCH, "phase0_state.jls"))
theta0 = st.theta0; calib = st.calib
D = st.D; nA = st.nA; b_q = st.b_q

function build_bundle()
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=80_000, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end
obj = build_bundle()
ctx = obj.γ
theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
theta_at(sign, t) = begin
    th = copy(theta_plain0); th[1+nA+1:end] .+= sign .* t .* b_q; th
end

o = 14
t_below = 7.1507250378e-03
t_above = 8.7302936151e-03

for (label, tt) in (("below", t_below), ("above", t_above))
    th = theta_at(-1, tt)
    iv = melitz_origin_intervals(o, th, ctx, obj)
    D_ = ctx.D
    W = size(obj.U, 1)
    z = @view obj.U[:, o]
    y = z .^ (ctx.sigma - 1)
    breakpoints, rank, autarky_rank = iv.breakpoints, iv.rank, iv.autarky_rank
    K = length(breakpoints)
    interval_of = [searchsortedlast(breakpoints, zv) for zv in z]
    ymin = fill(Inf, K); ymax = fill(-Inf, K); has_draw = falses(K); ndraw = zeros(Int, K)
    for w in 1:W
        k = interval_of[w]
        k == 0 && continue
        has_draw[k] = true; ndraw[k] += 1
        y[w] < ymin[k] && (ymin[k] = y[w])
        y[w] > ymax[k] && (ymax[k] = y[w])
    end
    C = [melitz_C(ctx.w[o], ctx.tau[o, d], iv.A[o, d], ctx.sigma, ctx.expenditure[d]) for d in 1:D_]
    c_od = [C[d] / ctx.expenditure[d] for d in 1:D_]
    lambda_od = [ctx.X_data[o, d] / ctx.expenditure[d] for d in 1:D_]
    H = lambda_od ./ c_od

    println("\n" * "="^100); println("origin=$o  label=$label  t=$tt"); println("="^100)
    println("breakpoints (sorted unique cutoffs, K=$K): ", breakpoints)
    println("rank (destination -> breakpoint index): ", rank)
    println("H[d] (required tail y-moment per destination):")
    for d in 1:D_
        @printf("  d=%2d  rank=%2d  H=%.8f  interval[rank]: ndraw=%d ymin=%.6g ymax=%.6g  achievable-tail-max=%s\n",
            d, rank[d], H[d], (rank[d]<=K ? ndraw[rank[d]] : -1),
            (rank[d]<=K ? ymin[rank[d]] : NaN), (rank[d]<=K ? ymax[rank[d]] : NaN),
            "n/a")
    end
    # Report the achievable RANGE of sum_{k=rank[d]}^K T_k given box constraints ymin_k*m_k<=T_k<=ymax_k*m_k,
    # sum m_k=1, m_k>=0 -- i.e. for the tail alone (ignoring the OTHER destinations' own equality
    # constraints), the max/min feasible tail-moment is bounded by [0, ymax over the tail range]
    # per unit mass devoted entirely to the best/worst interval in the tail; report the SPECIFIC
    # d whose target destination is d=19 (the switched cell) plus its neighbors by rank.
    d19 = 19
    println("\nFocus: destination d=$d19 (the switching cell) -- rank=$(rank[d19]), H=$(H[d19])")
    lo = rank[d19]
    println("Tail intervals [rank..K] for d=$d19:")
    for k in lo:K
        @printf("  interval k=%d  ndraw=%d  ymin=%.6g  ymax=%.6g\n", k, ndraw[k], ymin[k], ymax[k])
    end
    tail_ymax_sum = sum(has_draw[k] ? ymax[k] : 0.0 for k in lo:K)  # loose upper bound if ALL mass in tail at ymax
    println("loose upper bound on achievable tail moment (all unit mass, best interval per position) is NOT simply summed -- reporting per-interval box only, full answer is the LP result already computed.")

    feas = melitz_origin_block_lp(o, th, ctx, obj)
    println("compressed LP feasible = ", feas)
end
println("\nDONE PHASE 7")
