using LinearAlgebra, Statistics, Random
# Compare estimators of the coefficient b in a gravity regression of lnA_od on lnτ_od with
# ORIGIN + DESTINATION fixed effects, vs the code's cell-referenced double-difference.
# On a balanced DxD grid, OLS-two-way-FE coefficient == FWL with the two-way "within" transform.

D = 4
# two-way within (FE residual maker): z̃_od = z_od - mean_o - mean_d + grand
within(z) = (z .- mean(z, dims=2) .- mean(z, dims=1) .+ mean(z))
# cell-referenced double difference (the code): ΔΔz_od = z_od - z_1d - z_o2 + z_12
function dd(z)
    D = size(z,1); out = zeros(D,D)
    for o in 1:D, d in 1:D
        out[o,d] = z[o,d] - z[1,d] - z[o,2] + z[1,2]
    end
    return out
end
# OLS with origin+dest FE via explicit design matrix; return coef on x
function ols_fe_coef(x, y)
    D = size(x,1); n = D*D
    xv = vec(x); yv = vec(y)
    # design: intercept + (D-1) origin dummies + (D-1) dest dummies + x
    Xcols = [ones(n), xv]
    for o in 2:D; push!(Xcols, [ (((i-1)%D)+1)==o ? 1.0 : 0.0 for i in 1:n]); end
    for d in 2:D; push!(Xcols, [ (((i-1)÷D)+1)==d ? 1.0 : 0.0 for i in 1:n]); end
    X = hcat(Xcols...)
    coef = X \ yv
    return coef[2]  # coef on x
end

Random.seed!(42)
println("GC  test: regress lnA on lnτ with origin+dest FE.  Do the transforms recover the same b?")
println("GC  (repeat over random lnτ, lnA)")
maxdiff_within = 0.0; maxdiff_dd = 0.0; maxdiff_dd_demean = 0.0
for trial in 1:2000
    lnτ = randn(D,D); lnA = randn(D,D)
    b_fe = ols_fe_coef(lnτ, lnA)
    # within-transform estimator: b = <Wτ, WA>/<Wτ,Wτ>
    Wt = within(lnτ); Wa = within(lnA)
    b_within = sum(Wt .* Wa) / sum(Wt .* Wt)
    # cell double-diff estimator (all cells): b = <DDτ, DDA>/<DDτ,DDτ>
    Dt = dd(lnτ); Da = dd(lnA)
    b_dd = sum(Dt .* Da) / sum(Dt .* Dt)
    # code's demeaned double-diff: subtract mean of ΔΔτ from ΔΔτ (over the used cells), keep ΔΔA
    meanDt = mean(Dt)
    b_dd_dm = sum((Dt .- meanDt) .* Da) / sum((Dt .- meanDt) .* Dt)
    global maxdiff_within = max(maxdiff_within, abs(b_within - b_fe))
    global maxdiff_dd     = max(maxdiff_dd,     abs(b_dd     - b_fe))
    global maxdiff_dd_demean = max(maxdiff_dd_demean, abs(b_dd_dm - b_fe))
end
println("GC  max|b_within  - b_FE| = ", maxdiff_within,   "   (within == OLS two-way FE?)")
println("GC  max|b_DDcell  - b_FE| = ", maxdiff_dd,       "   (cell double-diff == FE?)")
println("GC  max|b_DDdemean- b_FE| = ", maxdiff_dd_demean,"   (code's demeaned DD == FE?)")

# Also: is the MOMENT Σ(within τ)(within A) equivalent to Σ(DD τ)(DD A)? Compare operator inner products
println("GC")
println("GC  operator check: is <Wτ,WA> ∝ <DDτ,DDA> for all A?  (need W'W ∝ DD'DD)")
lnτ = randn(D,D)
diffs = Float64[]
for k in 1:500
    a = randn(D,D)
    lhs = sum(within(lnτ).*within(a)); rhs = sum(dd(lnτ).*dd(a))
    push!(diffs, lhs)  # collect
end
# fit ratio: if proportional, lhs = c*rhs; check correlation
rhsv = [sum(dd(lnτ).*dd(randn(D,D))) for _ in 1:1]  # placeholder
println("GC  (see per-A comparison below)")
for k in 1:5
    a = randn(D,D)
    println("GC   <Wτ,WA>=", round(sum(within(lnτ).*within(a));digits=4), "   <DDτ,DDA>=", round(sum(dd(lnτ).*dd(a));digits=4))
end
println("GC DONE")
