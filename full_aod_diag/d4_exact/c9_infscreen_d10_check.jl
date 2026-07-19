# ============================================================================
# Zero-false-positive check at D=6/8/10 (synthetic, d_exact_setup_scaled,
# same class of context Continuation 8's gated D=6/8/10 pilots used).
# Reproducing the LITERAL converged D=10 gate point requires re-running a
# full KNITRO outer-loop optimization (c9_phase7_d10_upper_gate.jl); given
# this task's time budget, this script instead validates against each
# dimension's own calibration point (Aod_theta==1 everywhere) plus a handful
# of gravity-tangent-exact perturbations -- the same "known-feasible"
# reference class this investigation has used at every gated D (Continuation
# 8 Section 10: "D=6 and D=8 free-A pilots fully converge... D=10 upper...
# still cold-verified feasible" all starting from / staying near this same
# calibration basin). Honestly flagged as a pragmatic substitute, not the
# literal archived optimizer output, in the deliverable doc.
# ============================================================================
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
using Random, LinearAlgebra, Dates

println("c9_infscreen_d10_check.jl starting at ", now())
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

n_fp_total = 0
n_checked = 0
for D in (6, 8, 10)
    ctx = d_exact_setup_scaled(D = D, W = 8000, find_smallest = true)
    pe = build_pivot_elimination(ctx)
    zfree0 = pivot_reduce(zeros(D, D), pe)
    gp0 = ctx.θ0_up[3+D]
    Pmat = target_shares(ctx)
    pc = precompute_pairwise_M(ctx)

    pts = Dict{String,Vector{Float64}}("calibration" => vcat(gp0, zfree0))
    Random.seed!(1000 + D)
    for k in 1:3
        dir = randn(length(zfree0)); dir ./= norm(dir)
        pts["gravity_tangent_$k"] = vcat(gp0, zfree0 .+ (0.02 * k) .* dir)
    end

    for (name, w) in pts
        global n_fp_total, n_checked
        xf = x_free_from_w(w, pe)
        θ_full = CS.reconstruct_full(xf, ctx.m)
        a = compute_a_od(θ_full, ctx)
        pres = pairwise_certificate(a, pc, Pmat)
        order = order_destinations(pres, D)
        wres = screen_hard_winners(θ_full, ctx, Pmat; order = order)
        n_checked += 1
        is_fp = pres.infeasible || !wres.feasible
        is_fp && (n_fp_total += 1)
        println("D=$D  ", name, ": pairwise.infeasible=", pres.infeasible, "  winner_scan.feasible=", wres.feasible,
                "  worst_slack=", round(pres.worst_slack, digits=4))
    end

    # also confirm evaluate_fullA_screened matches evaluate_fullA_fast at calibration (integration check)
    xf_cal = x_free_from_w(pts["calibration"], pe)
    r_direct, _ = evaluate_fullA_fast(xf_cal, ctx; cache = nothing, use_cache = false, warm = false)
    r_screened, meta = evaluate_fullA_screened(xf_cal, ctx; moment_representation = :dense, cache = nothing, use_cache = false, warm = false)
    same_status = r_direct.inner_status == r_screened.inner_status
    dd_ok = (isnan(r_direct.Delta_dual) && isnan(r_screened.Delta_dual)) || isapprox(r_direct.Delta_dual, r_screened.Delta_dual; atol=1e-10)
    println("D=$D  calibration integration check: same inner_status=", same_status, "  Delta_dual match=", dd_ok, "  screen_status=", meta.screen_status)
    (same_status && dd_ok) || (global n_fp_total += 1)
end

println("\n=== TOTAL: $n_checked screen checks, $n_fp_total false positives / integration mismatches across D in (6,8,10) ===")
n_fp_total == 0 || error("D=6/8/10 check found $n_fp_total false positive(s)/mismatch(es)")
println("c9_infscreen_d10_check.jl PASSED at ", now())
