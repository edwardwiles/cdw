# Follow-up (user question): origin 14's own trade shares -- which destinations drive the
# extreme dependence -- and whether the fragile interval (destination 16's own rank-9
# interval, 1 draw at W=80,000) gains support at larger W.
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

o_focus = 14
t_below = 7.1507250378e-03

function build_bundle(W)
    obj, _ = build_melitz_psi_bundle_from_calibration(calib; W=W, seed=1,
        inner_loop_opt=joinpath(REPO2, "melitz_inner_loop_options_capped_2026-07-24.opt"),
        forbid_dense_fallback=true, policy=policy_cap, outer_parameterization=:logcutoff)
    return obj
end

println("\n" * "="^100); println("Origin 14 -- raw trade-share table at the pre-switch anchor (W=80,000)"); println("="^100)
obj80 = build_bundle(80_000)
ctx = obj80.γ
theta_plain0 = melitz_unpower_theta_free(theta0, ctx)
th_below = copy(theta_plain0); th_below[1+nA+1:end] .-= t_below .* b_q
A_, f_, gpj_, _ = melitz_expand_theta(th_below, ctx)

countries = vec(readdlm(joinpath(REPO2, "real_data", "noah_D20", "countries.csv"), ',', String))
println("origin 14 = ", countries[o_focus])
@printf("%-4s %-8s %10s %10s %10s %10s %10s\n", "d", "ctry", "X_data", "lambda_od", "C_od", "H[d]", "zhat_od")
zhat = melitz_baseline_cutoff(A_, f_, ctx.w, ctx.tau, ctx.expenditure, ctx.sigma)
total_X = sum(ctx.X_data[o_focus, :])
for d in 1:D
    C_od = melitz_C(ctx.w[o_focus], ctx.tau[o_focus, d], A_[o_focus, d], ctx.sigma, ctx.expenditure[d])
    lam = ctx.X_data[o_focus, d] / ctx.expenditure[d]
    H = lam / (C_od / ctx.expenditure[d])
    @printf("%-4d %-8s %10.4g %10.6f %10.4g %10.6f %10.6f\n", d, countries[d], ctx.X_data[o_focus,d], lam, C_od, H, zhat[o_focus,d])
end
@printf("\ntotal export value X_data[14,:] sum = %.4g   share of destination 14->19 (the switching cell) = %.6f\n",
    total_X, ctx.X_data[o_focus,19]/total_X)
@printf("share of destination 14->16 (the fragile interval's own destination) = %.6f\n", ctx.X_data[o_focus,16]/total_X)

# Scan ALL K intervals for how many draws support each one (not just interval 9) -- is this a
# one-off or a broader pattern for origin 14?
iv = melitz_origin_intervals(o_focus, th_below, ctx, obj80)
W80 = 80_000
z80 = @view obj80.U[:, o_focus]
interval_of = [searchsortedlast(iv.breakpoints, zv) for zv in z80]
K = length(iv.breakpoints)
ndraw80 = zeros(Int, K)
for w in 1:W80
    k = interval_of[w]; k == 0 && continue
    ndraw80[k] += 1
end
println("\nAll $K intervals for origin 14 at W=80,000 -- draw counts (looking for other thin ones):")
for k in 1:K
    marker = ndraw80[k] <= 3 ? "  <<< THIN" : ""
    @printf("  interval %2d: ndraw=%6d  breakpoint_hi=%.6f%s\n", k, ndraw80[k], iv.breakpoints[k], marker)
end
thin_count = count(<=(3), ndraw80)
println("\nintervals with <=3 draws (out of $K): $thin_count")

println("\n" * "="^100); println("Interval-9 (destination d=16's own rank position) draw count vs W"); println("="^100)
for W in (80_000, 320_000, 1_280_000)
    objW = build_bundle(W)
    ctxW = objW.γ
    ivW = melitz_origin_intervals(o_focus, th_below, ctxW, objW)
    zW = @view objW.U[:, o_focus]
    rank16 = ivW.rank[16]
    interval_ofW = [searchsortedlast(ivW.breakpoints, zv) for zv in zW]
    ndraw_k = count(==(rank16), interval_ofW)
    expected_linear = W == 80_000 ? ndraw_k : NaN
    @printf("W=%9d  destination-16 rank position=%d  ndraw in that interval=%d  (naive linear-in-W scaling from W=80k would predict %.2f)\n",
        W, rank16, ndraw_k, W == 80_000 ? NaN : ndraw_k)
    if W == 80_000
        global base_ndraw = ndraw_k
    else
        @printf("   actual/naive-linear-prediction ratio = %.3f\n", ndraw_k / (base_ndraw * W/80_000))
    end
    # Also recheck feasibility at this W, same bracket, both sides.
    t_above_local = t_below * 1.22   # matches the original above-bracket ratio roughly
    th_above = copy(melitz_unpower_theta_free(theta0, ctxW)); th_above[1+nA+1:end] .-= 8.7302936151e-03 .* b_q
    feas_below = melitz_origin_block_lp(o_focus, th_below, ctxW, objW)
    feas_above = melitz_origin_block_lp(o_focus, th_above, ctxW, objW)
    @printf("   feasibility at this W: below=%s  above=%s\n", feas_below, feas_above)
end
println("\nDONE follow-up 2")
