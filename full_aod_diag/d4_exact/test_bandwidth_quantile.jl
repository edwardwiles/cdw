# ============================================================================
# Continuation 9, Phase 5: correctness checks for select_bandwidth_quantile
# (bandwidth_quantile.jl) and the validate_frac subsampling lever
# (composite_gradient_fast.jl), at D=4 (fast, synthetic context) before
# trusting either at D=20/W=80000.
#
# Three claims checked:
#   1. select_bandwidth_quantile's closed-form flip-threshold formula
#      correctly predicts count_winner_flips at several h values (the formula
#      itself, independent of the quantile-picking policy).
#   2. select_bandwidth_quantile's chosen h achieves a mass close to
#      select_bandwidth's target band, and the resulting A-block FD gradient
#      agrees closely (not bit-identically -- different h choice -- but
#      cosine-similar) with the :adaptive gradient.
#   3. validate_frac subsampling changes ONLY which coordinates get a
#      non-NaN slope_ratio, never the returned gradient g itself.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
using Printf, LinearAlgebra, Statistics

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

const W_CAND = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181,
    0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515,
    1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252,
    0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
xf = x_free_from_w(W_CAND)

base = solve_base_state(xf, ctx)
cache = build_lfix_base_cache(xf, ctx, base)
z0 = log.(reshape(xf[2:end], D, D))
w0 = vcat(xf[1], pivot_reduce(z0, pe))

all_pass = true

println("="^78); println("TEST 1: closed-form flip-threshold formula predicts count_winner_flips exactly"); println("="^78)
t1_pass = true
for k in 2:D2
    cells = affected_cells(pe, k)
    dests = last.(cells)
    length(unique(dests)) < length(cells) && continue   # same-dest collision, not closed-formed
    dir_lin, slope_dir, piv_lin, slope_piv = coord_cell_slopes(pe, k)
    μ = cache.μ
    (o1, d1) = cells[1]; c1 = -μ * slope_dir
    (o2, d2) = cells[2]; c2 = -μ * slope_piv
    hflips = Float64[]
    exact_flip_thresholds!(hflips, cache, μ, o1, d1, c1)
    exact_flip_thresholds!(hflips, cache, μ, o2, d2, c2)
    sort!(hflips)
    # Spot-check at 5 h values spanning the candidate range (plus h0=0.01), each
    # nudged OFF an exact hflips boundary by a relative epsilon: at h==hflips[i]
    # EXACTLY, update_winner_o1's own convention is a STRICT inequality (equality
    # means "not yet flipped" -- see composite_gradient.jl's update_winner_o1
    # docstring), so testing exactly AT a threshold is a genuine (known, harmless
    # for bandwidth SELECTION purposes) boundary-convention mismatch, not a
    # closed-form error -- avoided here by testing strictly-interior points.
    test_hs = isempty(hflips) ? [0.01, 0.05] :
        sort(unique(vcat(0.01, hflips[max(1,end÷4)]*1.0000001, hflips[max(1,end÷2)]*1.0000001,
                          hflips[end]*0.5, hflips[end]*1.5)))
    for h in test_hs
        h <= 0 && continue
        w = copy(w0); w[k] += h
        z = pivot_expand(w[2:end], pe); Aod_theta = exp.(z)
        x_free = vcat(w[1], vec(Aod_theta))
        θ_full = CS.reconstruct_full(x_free, ctx.m)
        predicted = count(x -> x <= h, hflips)
        actual = 0
        for d in unique(dests)
            origins_here = [o for (o, dd) in cells if dd == d]
            actual += count_winner_flips(cache, ctx, θ_full, d, origins_here; multi_method = :top3)
        end
        if predicted != actual
            println("  MISMATCH coord=$k h=$h predicted=$predicted actual=$actual")
            global t1_pass = false
        end
    end
end
println("  Test 1: ", t1_pass ? "PASS" : "FAIL", " (closed-form flip count matches count_winner_flips at all sampled h, all non-collision coords)")
global all_pass &= t1_pass

println("\n" * "="^78); println("TEST 2: select_bandwidth_quantile mass/h vs select_bandwidth bisection; gradient agreement"); println("="^78)
lo_frac, hi_frac = 0.003, 0.03
mass_diffs = Float64[]
h_ratios = Float64[]
g_bisect = zeros(D2); g_quant = zeros(D2)
g_bisect[1] = g_quant[1] = gamma_component_analytic(cache, base, w0[1])
for k in 2:D2
    hb, mb, _ = select_bandwidth(cache, ctx, pe, w0, k; multi_method = :top3)
    hq, mq, metaq = select_bandwidth_quantile(cache, ctx, pe, w0, k; multi_method = :top3)
    push!(mass_diffs, abs(mb - mq))
    push!(h_ratios, hq / hb)
    g_bisect[k] = a_block_fd_component(cache, ctx, pe, w0, k, hb; multi_method = :top3)
    g_quant[k] = a_block_fd_component(cache, ctx, pe, w0, k, hq; multi_method = :top3)
    @printf("  k=%2d  method=%-24s  h_bisect=%.5f mass_bisect=%.5f  |  h_quant=%.5f mass_quant=%.5f  n_cand=%s\n",
        k, get(metaq, :method, :quantile), hb, mb, hq, mq, get(metaq, :n_candidates, -1))
end
cos_sim = dot(g_bisect[2:end], g_quant[2:end]) / (norm(g_bisect[2:end]) * norm(g_quant[2:end]))
@printf("\n  median|mass_bisect - mass_quant| = %.5f   median(h_quant/h_bisect) = %.3f\n", median(mass_diffs), median(h_ratios))
@printf("  cosine(g_bisect[A-block], g_quant[A-block]) = %.8f\n", cos_sim)
t2_pass = cos_sim > 0.995
println("  Test 2: ", t2_pass ? "PASS" : "FAIL", " (A-block gradient cosine similarity > 0.995)")
global all_pass &= t2_pass

println("\n" * "="^78); println("TEST 3: validate_frac subsampling changes only slope_ratio pattern, never g"); println("="^78)
g_full, meta_full = composite_gradient_at_fast(xf, ctx, pe; base = base, h_mode = :adaptive, validate_frac = 1.0)
g_half, meta_half = composite_gradient_at_fast(xf, ctx, pe; base = base, h_mode = :adaptive, validate_frac = 0.5)
g_zero, meta_zero = composite_gradient_at_fast(xf, ctx, pe; base = base, h_mode = :adaptive, validate_frac = 0.0)
maxdiff_g = max(maximum(abs.(g_full .- g_half)), maximum(abs.(g_full .- g_zero)))
n_validated_half = count(!isnan, meta_half.slope_ratio[2:end])
n_validated_zero = count(!isnan, meta_zero.slope_ratio[2:end])
n_validated_full = count(!isnan, meta_full.slope_ratio[2:end])
@printf("  max|g diff| across validate_frac in {1.0,0.5,0.0} = %.3e\n", maxdiff_g)
println("  n_validated: frac=1.0 -> $n_validated_full/$( D2-1),  frac=0.5 -> $n_validated_half/$( D2-1),  frac=0.0 -> $n_validated_zero/$( D2-1)")
t3_pass = maxdiff_g < 1e-13 && n_validated_full == D2-1 && n_validated_zero == 0 && abs(n_validated_half - round(Int,0.5*(D2-1))) <= 1
println("  Test 3: ", t3_pass ? "PASS" : "FAIL")
global all_pass &= t3_pass

println("\n" * "="^78); println(all_pass ? "ALL TESTS PASS" : "SOME TESTS FAILED"); println("="^78)
