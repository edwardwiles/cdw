include(joinpath(@__DIR__, "debug_pq_fg_check.jl"))

println("\n=== isolating H_E,R cross-block ===")
D = ctx.D; npair = aug.op.npair
n_rows = n_total_rows(D, PQ_L)
NCORE = ncore1 + 1

cf = aug.core_cf_ref[]
wctx = build_winner_pair_ctx(cf)
println("wctx.ncolI = ", wctx.ncolI, "  NCORE (should be ncolI+1) = ", NCORE, "  has_cf=", wctx.has_cf, "  Ddest=", wctx.Ddest)

obj.ddPsi!(obj.arg2, obj.arg0)
h = obj.arg2

cross_scratch = ensure_winner_zc_cross_scratch!(Ref{Union{Nothing,WinnerZCCrossScratch}}(nothing), aug.op.W, n_rows)
winner_pair_cross_hessian_zc_prep!(cross_scratch, wctx, h)
HEQ = zeros(NCORE, n_rows)
cross_hess_scratch = PairwiseQuantileCrossHessScratch(D, npair, aug.op.W, ncore1, PQ_L)
pairwise_quantile_cross_hessian_block!(HEQ, wctx, cross_scratch, aug.op, bin_state, build_pairwise_quantile_thread_scratch(D, npair, PQ_L), h, cross_hess_scratch)

# brute-force cross derivative: d(g_E[j]) / d(lambda_R[x]) via FD on the FG functor's gradient,
# for a SMALL sample of (j,x) pairs including the culprit column j=6 (Hfull col index 7 = j+1=7 -> j=6)
hstep = 1e-5
function g_full(xvec)
    gg = zeros(length(xvec))
    st(xvec, gg)
    return gg
end

# pick a handful of restriction coordinates to test against ALL economic rows (cheap: ncore1+1=18 rows x few x)
test_xs = [1, 2, 5, marginal_row(2,1,PQ_L), pair_row(D,1,1,1,PQ_L), 20, 50, n_rows]
println("testing restriction cols (Hfull col index, i.e. NCORE+x): ", test_xs)

maxerr = 0.0
for x in test_xs
    xp = copy(x0); xp[NCORE + x] += hstep
    xm = copy(x0); xm[NCORE + x] -= hstep
    gp = g_full(xp); gm = g_full(xm)
    dgE_dlamR = (gp[1:NCORE] .- gm[1:NCORE]) ./ (2hstep)   # includes row1 (zeta) + economic rows
    # HEQ's own row convention: row 1 = zeta row (matches g[1]), rows 2:NCORE = economic lambda rows (matches g[2:NCORE])
    analytic_col = HEQ[:, x]
    err = maximum(abs.(analytic_col .- dgE_dlamR))
    global maxerr = max(maxerr, err)
    println("x=", x, "  max|analytic-fd| over NCORE rows = ", err,
            "  (analytic[1:3]=", round.(analytic_col[1:3],digits=6), " fd[1:3]=", round.(dgE_dlamR[1:3],digits=6), ")")
end
println("OVERALL max err = ", maxerr)
println(maxerr < 1e-3 ? "CROSS-BLOCK ISOLATED CHECK: PASS" : "CROSS-BLOCK ISOLATED CHECK: FAIL")
