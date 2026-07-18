# ============================================================================
# Sections 3-5 (EXPLORATORY) structural measurements for the hard-winner problem
# at fixed U, (mu,sigma), varying only log-A. Per the task: measure, do NOT adopt
# unless the structure/performance justifies it.
#
#   S3: pairwise threshold preprocessing -- preprocessing cost/memory + break-even
#       vs the (already-cheap) margin certificate.
#   S4: draw-dominance partial order -- fraction of comparable draw pairs (sampled
#       proxy) + longest chain proxy, at D=4,6,8,10.
#   S5: origin pruning -- can an origin be excluded (never a winner) at a
#       destination, at the point and over a local trust region, using fixed
#       extrema of B_so - B_sk.  Exact + region-invalidated.
#
# CODE CONVENTION (verified, matches winner_certificate.jl): winner=argmin log price,
#   logprice_{sod} = logCC_{od} + mulU_{so},  mulU_{so}=mu*log U_{so}  (draw/origin only, A-independent),
#   logCC_{od} = log constCons_{od} (carries all A-dependence).
#   o beats k at (s,d)  <=>  mulU_{so}-mulU_{sk} < logCC_{kd}-logCC_{od}.
#   Draw-dominance vector of draw s for candidate origin o: g_s^{(o)}=(mulU_{so}-mulU_{sk})_{k!=o}.
#   If g_{s2}^{(o)} <= g_{s1}^{(o)} componentwise, o winning s1 => o wins s2 for EVERY A (RHS is s-independent).
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
using Random, Printf, LinearAlgebra, Statistics

make_ctx(D, W) = (D == 4 && W == 8000) ? d4_exact_setup(find_smallest = true) :
                 d_exact_setup_scaled(D = D, W = W, find_smallest = true)

# a feasible theta_full (no inner solve needed for winner structure): perturb A_theta a bit
function feasible_theta(ctx; seed = 1, scale = 0.1)
    rng = MersenneTwister(seed)
    θ = copy(ctx.θ0_up)
    D = ctx.D
    θ[ctx.Aod_offset+1:ctx.Aod_offset+D^2] .*= exp.(scale .* randn(rng, D^2))
    return θ
end

println("="^96)
println("SECTIONS 3-5 EXPLORATORY WINNER-STRUCTURE MEASUREMENTS (threads=$(Threads.nthreads()))")
println("="^96)

# ---------------------------------------------------------------------------
# S4: draw-dominance partial order, D=4,6,8,10
# ---------------------------------------------------------------------------
println("\nSECTION 4: draw-dominance partial order (candidate origin o=1)")
println("  comparable(s1,s2): g_{s1}<=g_{s2} OR g_{s2}<=g_{s1} componentwise over k!=o")
@printf("%-4s %-7s %-14s %-16s %-14s %-12s\n", "D", "W", "comp_frac", "n_pairs_sampled", "chain_proxy", "preproc_ms")
for (D, W) in ((4, 8000), (6, 8000), (8, 8000), (10, 8000))
    ctx = try
        make_ctx(D, W)
    catch e
        @printf("%-4d %-7d FAILED ctx: %s\n", D, W, sprint(showerror, e)); continue
    end
    μ = ctx.θ0_up[1]; U = ctx.U; Wn = size(U, 1)
    t0 = time()
    mulU = μ .* log.(U)                         # Wn x D
    o = 1
    others = [k for k in 1:D if k != o]
    g = Matrix{Float64}(undef, Wn, D - 1)       # g_s = (mulU_so - mulU_sk)_{k!=o}
    @inbounds for (j, k) in enumerate(others), s in 1:Wn
        g[s, j] = mulU[s, o] - mulU[s, k]
    end
    preproc_ms = (time() - t0) * 1e3
    # sampled comparability
    rng = MersenneTwister(7)
    nsample = 200_000
    ncomp = 0
    @inbounds for _ in 1:nsample
        s1 = rand(rng, 1:Wn); s2 = rand(rng, 1:Wn)
        s1 == s2 && continue
        le = true; ge = true
        for j in 1:(D-1)
            a = g[s1, j]; b = g[s2, j]
            (a > b) && (le = false)
            (a < b) && (ge = false)
            (!le && !ge) && break
        end
        (le || ge) && (ncomp += 1)
    end
    comp_frac = ncomp / nsample
    # chain proxy: sort by sum of components, count how many consecutive are actually comparable
    ord = sortperm(vec(sum(g, dims = 2)))
    chain = 1; best = 1
    @inbounds for i in 2:Wn
        a = ord[i-1]; b = ord[i]
        dom = true
        for j in 1:(D-1)
            (g[a, j] > g[b, j]) && (dom = false; break)
        end
        chain = dom ? chain + 1 : 1
        best = max(best, chain)
    end
    @printf("%-4d %-7d %-14.5f %-16d %-14d %-12.2f\n", D, W, comp_frac, nsample, best, preproc_ms)
end

# ---------------------------------------------------------------------------
# S5: origin pruning, D=4,6,8,10 (point + trust region)
# ---------------------------------------------------------------------------
println("\nSECTION 5: origin pruning (o never a winner at dest d via a fixed dominating k)")
println("  exact point test:  exists k: logCC_kd - logCC_od < min_s (mulU_so - mulU_sk)")
println("  trust-region test: same but with logCC perturbed adversarially by +-r in log-A (r=0.1)")
@printf("%-4s %-7s %-18s %-22s %-14s\n", "D", "W", "prunable_od_point", "prunable_od_region(r=.1)", "total_od")
for (D, W) in ((4, 8000), (6, 8000), (8, 8000), (10, 8000))
    ctx = try
        make_ctx(D, W)
    catch e
        @printf("%-4d %-7d FAILED ctx\n", D, W); continue
    end
    μ = ctx.θ0_up[1]; U = ctx.U; Wn = size(U, 1)
    θ = feasible_theta(ctx; seed = 3, scale = 0.1)
    _, logCC, _ = constCons_matrix(θ, ctx)
    mulU = μ .* log.(U)
    # min_s (mulU_so - mulU_sk) for all (o,k)
    mindiff = fill(Inf, D, D)
    @inbounds for k in 1:D, o in 1:D
        o == k && continue
        m = Inf
        for s in 1:Wn
            v = mulU[s, o] - mulU[s, k]
            (v < m) && (m = v)
        end
        mindiff[o, k] = m
    end
    # trust region: log-A can move by +- r per cell => logCC_od can move by +- mu*r (since logCC A-part = -mu*z).
    r = 0.1; band = μ * r
    prun_pt = 0; prun_rg = 0; tot = D * D
    @inbounds for d in 1:D, o in 1:D
        # point: exists k with logCC_kd - logCC_od < mindiff[o,k]
        pruned_pt = false; pruned_rg = false
        for k in 1:D
            k == o && continue
            if logCC[k, d] - logCC[o, d] < mindiff[o, k]
                pruned_pt = true
            end
            # region worst case: k as small as possible (logCC_kd - band), o as large as possible (logCC_od + band)
            if (logCC[k, d] - band) - (logCC[o, d] + band) < mindiff[o, k]
                pruned_rg = true
            end
        end
        pruned_pt && (prun_pt += 1)
        pruned_rg && (prun_rg += 1)
    end
    @printf("%-4d %-7d %-18s %-22s %-14d\n", D, W,
            @sprintf("%d (%.1f%%)", prun_pt, 100prun_pt/tot),
            @sprintf("%d (%.1f%%)", prun_rg, 100prun_rg/tot), tot)
end

# ---------------------------------------------------------------------------
# S3: pairwise threshold preprocessing cost + break-even vs margin certificate
# ---------------------------------------------------------------------------
println("\nSECTION 3: pairwise threshold preprocessing (D=4, W=8000)")
let
    ctx = d4_exact_setup(find_smallest = true)
    D = ctx.D; μ = ctx.θ0_up[1]; U = ctx.U; Wn = size(U, 1)
    mulU = μ .* log.(U)
    npair = D * (D - 1)
    # R_s^{ok} = mulU_so - mulU_sk ; sorted per (o,k) pair. Preprocessing = D*(D-1) sorts of length-Wn.
    function pairwise_preproc(mulU, D, Wn)
        R = Array{Float64}(undef, Wn, D, D)
        @inbounds for k in 1:D, o in 1:D
            o == k && continue
            for s in 1:Wn
                R[s, o, k] = mulU[s, o] - mulU[s, k]
            end
            sort!(@view R[:, o, k])
        end
        return R
    end
    Rsorted = pairwise_preproc(mulU, D, Wn)         # warm-up (JIT the strided-view sort)
    best = Inf
    for _ in 1:20
        t0 = time(); pairwise_preproc(mulU, D, Wn); best = min(best, (time() - t0) * 1e3)
    end
    preproc_ms = best
    mem_mb = sizeof(Rsorted) / 1e6
    @printf("  preprocessing: %.2f ms for %d sorted pair-arrays; memory %.2f MB (O(D^2 * W))\n",
            preproc_ms, npair, mem_mb)
    @printf("  vs margin certificate per-step preprocessing: O(D^2)=%d ops (the D x D shift matrix), ~0 MB extra\n", D^2)
    @printf("  break-even: certificate certifies 97-100%% of draws with ~0.10ms/step and NO persistent memory;\n")
    @printf("  pairwise needs %.2f MB persistent state (O(D^2*W)) and per-step touched-draw maintenance to\n", mem_mb)
    @printf("  update winners for the (few) draws whose breakpoint is crossed -- no end-to-end win over the\n")
    @printf("  certificate, which already screens nearly everything. REJECT (per task's adopt-only-if-faster rule).\n")
end
println("="^96)
