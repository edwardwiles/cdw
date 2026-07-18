# ============================================================================
# Phase 1B equivalence test: compute_winners_fast (winners_v2.jl) must match
# compute_winners (winners.jl) EXACTLY -- same winner index, same gap, same
# tie-breaking -- across random matrices, exact ties, near ties, and
# Inf/NaN guard cases, before being trusted to replace it in oracle_fast.jl.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
using Random

println("="^78); println("TEST A: min_and_secondmin unit tests (synthetic columns)"); println("="^78)

function check_case(name, col, expected_min, expected_idx, expected_gap)
    m1, idx1, gap = min_and_secondmin(col)
    ok = (m1 == expected_min || (isnan(expected_min) && isnan(m1))) && idx1 == expected_idx &&
         (gap == expected_gap || (isinf(gap) && isinf(expected_gap)) || (isnan(gap) && isnan(expected_gap)) || isapprox(gap, expected_gap; atol=1e-14))
    println(rpad(name, 30), " col=", col, " -> (m1=$m1, idx=$idx1, gap=$gap)  expected=(", expected_min, ",", expected_idx, ",", expected_gap, ")  ", ok ? "PASS" : "FAIL")
    return ok
end

all_ok = true
all_ok &= check_case("simple ascending",      [1.0, 2.0, 3.0, 4.0], 1.0, 1, 1.0)
all_ok &= check_case("simple descending",     [4.0, 3.0, 2.0, 1.0], 1.0, 4, 1.0)
all_ok &= check_case("min in middle",         [3.0, 1.0, 4.0, 2.0], 1.0, 2, 1.0)
all_ok &= check_case("exact tie for min",     [1.0, 1.0, 2.0, 3.0], 1.0, 1, 0.0)   # findmin picks FIRST occurrence
all_ok &= check_case("exact tie, min later",  [2.0, 1.0, 1.0, 3.0], 1.0, 2, 0.0)
all_ok &= check_case("all equal",             [5.0, 5.0, 5.0], 5.0, 1, 0.0)
all_ok &= check_case("near tie",              [1.0, 1.0 + 1e-14, 2.0], 1.0, 1, 1e-14)
all_ok &= check_case("single element",        [7.0], 7.0, 1, Inf)
all_ok &= check_case("two elements",          [3.0, 1.0], 1.0, 2, 2.0)
all_ok &= check_case("with Inf, min finite",  [Inf, 1.0, 2.0], 1.0, 2, 1.0)
all_ok &= check_case("with -Inf",             [-Inf, 1.0, 2.0], -Inf, 1, Inf)
all_ok &= check_case("all NaN",               [NaN, NaN], NaN, 1, NaN)
all_ok &= check_case("NaN + finite",          [NaN, 1.0, 2.0], 1.0, 2, 1.0)
all_ok &= check_case("finite then NaN",       [1.0, NaN, 2.0], 1.0, 1, 1.0)

println("\nCross-check against Julia's own findmin (index/value) for every case above, plus 2000 random trials:")
rng = MersenneTwister(20260718)
findmin_ok = true
for trial in 1:2000
    n = rand(rng, 1:8)
    col = randn(rng, n)
    if rand(rng) < 0.3 && n >= 2   # inject exact ties in ~30% of trials
        i, j = rand(rng, 1:n, 2)
        col[j] = col[i]
    end
    m1, idx1, gap = min_and_secondmin(col)
    fm_val, fm_idx = findmin(col)
    s = sort(col)
    expected_gap = n >= 2 ? (s[2] - s[1]) : Inf
    ok = m1 == fm_val && idx1 == fm_idx && (gap == expected_gap || (isinf(gap) && isinf(expected_gap)))
    global findmin_ok &= ok
    ok || println("  MISMATCH trial=$trial col=$col -> got ($m1,$idx1,$gap) vs findmin ($fm_val,$fm_idx) sort-gap $expected_gap")
end
println("2000 random trials (30% with injected exact ties) vs findmin+sort: ", findmin_ok ? "ALL PASS" : "SOME FAILED")
all_ok &= findmin_ok

println("\n" * "="^78); println("TEST B: compute_winners_fast vs compute_winners, at real candidate points"); println("="^78)
ctx = d4_exact_setup(find_smallest = true)

function compare_full(θ_full, label)
    winner1, price1, gap1 = compute_winners(θ_full, ctx)
    winner2, price2, gap2 = compute_winners_fast(θ_full, ctx)
    price_ok = price1 == price2
    winner_ok = winner1 == winner2
    gap_ok = gap1 == gap2
    hash_ok = hash(winner1) == hash(winner2)
    println(rpad(label, 30), " price_identical=", price_ok, " winner_identical=", winner_ok,
            " gap_identical=", gap_ok, " winner_hash_identical=", hash_ok)
    return price_ok && winner_ok && gap_ok && hash_ok
end

# calibration theta
θ_cal = copy(ctx.θ0_up)
all_ok &= compare_full(θ_cal, "calibration theta")

# upper maxit40 candidate
include(joinpath(@__DIR__, "gravity_elimination.jl"))
pe = build_pivot_elimination(ctx)
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
x_free_up40 = vcat(w_up40[1], vec(exp.(pivot_expand(w_up40[2:end], pe))))
θ_up40 = CS.reconstruct_full(x_free_up40, ctx.m)
all_ok &= compare_full(θ_up40, "upper_maxit40 theta")

# random feasible perturbations
rng2 = MersenneTwister(4242)
for trial in 1:10
    w = w_up40 .+ 0.01 .* randn(rng2, length(w_up40))
    xf = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    θ = CS.reconstruct_full(xf, ctx.m)
    global all_ok &= compare_full(θ, "random perturbation $trial")
end

println("\n" * "="^78)
println(all_ok ? "ALL WINNERS_V2 EQUIVALENCE TESTS PASSED" : "SOME TESTS FAILED")
println("="^78)
all_ok || error("test_winners_v2.jl: equivalence check failed")
