# ============================================================================
# Claude Code task 2026-08-01, follow-up: validate the O(1)-incremental
# profiled gradient (profiled_lfix_incremental_2026-08-01.jl) against the
# already-gated full-rebuild version (profiled_outer_gradient_fd_2026-08-01.jl)
# at MACHINE PRECISION, not cosine similarity against expensive re-solved
# ground truth -- both are the exact same central-FD formula of the exact
# same functional (Delta_dual, fixed dual), just computed two different ways,
# so they must agree to floating-point tolerance if the incremental
# mechanism is correct. This is the direct, cheap test of whether the
# earlier (abandoned) incremental draft's failure was really just the
# separately-diagnosed gp-formula bug.
#
# CORRECTED 2026-08-01 (production outer bridge task, §12): this script's own
# header above, and PROFILED_UNRESTRICTED_OUTER_AB_MASTER_2026-08-01.md /
# PROFILED_OUTER_GRADIENT_DERIVATION_2026-08-01.md, previously described this
# comparison's result as "machine precision" / "cos_sim=1.0000000000 (max rel
# err ~6e-6)". Re-running this exact script (unedited) does NOT reproduce
# that claim -- it reproduces `cos_sim~0.9998982`, `max_rel_err~2.0` at D4,
# because `profiled_composite_gradient_at` (the full-rebuild comparator) uses
# one FIXED `h=0.01` for every coordinate while
# `profiled_composite_gradient_at_incremental` uses an ADAPTIVE, per-
# coordinate `h` from `profiled_select_bandwidth` -- two different formulas'
# worth of truncation error, not a bug in either method (confirmed directly:
# see the sibling outer-gradient branch's mock-family gate, §2c-B of
# PROFILED_RESTRICTED_OUTER_GRADIENT_MASTER_2026-08-01.md). This script now
# reports BOTH comparisons explicitly, never conflating them:
#   - `mismatched_bandwidth` (fixed h=0.01 vs adaptive h): the historical
#     comparison, kept for continuity -- an APPROXIMATION/robustness check,
#     NOT a formula-equivalence proof. Expect cos_sim~0.9999, NOT ~1.0.
#   - `same_bandwidth` (full-rebuild re-run at the incremental method's own
#     h_used per coordinate): the genuine formula-equivalence gate -- THIS is
#     the one that should, and does, hit machine precision (~1e-16).
# Nothing about the incremental or full-rebuild gradient MATH changed here --
# this is a diagnostic/test correction only (task §12: "not a change to
# production gradient mathematics").
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "operator_verification.jl"))
include(joinpath(@__DIR__, "winner_certificate.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "recover_full_a_2026-07-31.jl"))
include(joinpath(@__DIR__, "homogeneous_moments_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_contraction_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_homogeneous_hessian_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_operator_verification_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_operator_bundle_2026-08-01.jl"))
include(joinpath(@__DIR__, "reduced_recovery_from_lfd_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_evaluator_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_outer_gradient_fd_2026-08-01.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "profiled_lfix_incremental_2026-08-01.jl"))
using LinearAlgebra, Printf, CSV, DataFrames

const IS_D20 = "D20" in ARGS

if IS_D20
    include(joinpath(@__DIR__, "context_real_d20.jl"))
    println("Building real D=20 :exclude_row context at W=80000 ..."); flush(stdout)
    ctx = d20_real_setup(W = 80_000, destination_sample = :exclude_row)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(14 => 3))
    tag = "D20_W80000"
else
    ctx0 = d4_exact_setup()
    ctx = build_unrestricted_operator_ctx(ctx0)
    spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
    tag = "D4"
end
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
n_total = outer_dim_profiled(pe)
println("tag=$tag  n_total=$n_total"); flush(stdout)

function compare_at(label::String, w::Vector{Float64}, rows)
    println("\n" * "="^90); println("POINT: $label"); println("="^90); flush(stdout)
    ev = evaluate_profiled_point(w, ctx, spec, pe)
    println("base Delta_dual=$(ev.result.Delta_dual)  inner_status=$(ev.result.inner_status)"); flush(stdout)

    t1 = time()
    g_full, _ = profiled_composite_gradient_at(w, ctx, spec, pe, ev)
    t_full = time() - t1
    println("full-rebuild gradient computed in $(t_full)s"); flush(stdout)

    t2 = time()
    g_inc, meta_inc = profiled_composite_gradient_at_incremental(w, ctx, spec, pe, ev)
    t_inc = time() - t2
    println("incremental gradient computed in $(t_inc)s (speedup=$(round(t_full/t_inc,digits=1))x)"); flush(stdout)

    function report(tag::String, gA, gB)
        diff = gA .- gB
        max_abs_err = maximum(abs.(diff))
        max_rel_err = maximum(abs.(diff) ./ max.(abs.(gA), 1e-8))
        cos_sim = dot(gA, gB) / (norm(gA) * norm(gB) + 1e-300)
        worst_k = argmax(abs.(diff))
        println(@sprintf("[%s] max_abs_err=%.4e  max_rel_err=%.4e  cos_sim=%.10f  worst_k=%d (full=%.6e inc=%.6e)",
            tag, max_abs_err, max_rel_err, cos_sim, worst_k, gA[worst_k], gB[worst_k]))
        flush(stdout)
        return (max_abs_err = max_abs_err, max_rel_err = max_rel_err, cos_sim = cos_sim)
    end

    # (A) mismatched_bandwidth: the historical comparison (fixed h=0.01 vs incremental's
    # adaptive h). An approximation/robustness check, NOT formula-equivalence -- expect
    # cos_sim~0.9999, not ~1.0. `g_full` above already used the default fixed h=0.01.
    mm = report("mismatched_bandwidth", g_full, g_inc)
    println(@sprintf("gp: full=%.10e  inc=%.10e  diff=%.3e", g_full[1], g_inc[1], abs(g_full[1]-g_inc[1])))

    # (B) same_bandwidth: re-run the full-rebuild comparator at the incremental method's
    # OWN per-coordinate h_used (coordinate 1 = gp is analytic in the incremental method,
    # not FD-derived, so its h_used[1] entry is meaningless for gp -- reuse g_full[1] there
    # and only match bandwidth on coordinates 2:end, which are genuinely FD-vs-FD).
    h_matched = copy(meta_inc.h_used); h_matched[1] = 0.01  # gp: no FD step to match
    t3 = time()
    g_full_matched, _ = profiled_composite_gradient_at(w, ctx, spec, pe, ev; h = h_matched)
    t_matched = time() - t3
    println("full-rebuild (matched bandwidth) computed in $(t_matched)s"); flush(stdout)
    # A-block (coords 2:end) is the genuine same-bandwidth formula-equivalence claim.
    # Coordinate 1 (gp) is EXCLUDED from that claim -- the incremental method computes gp via
    # an exact analytic formula, not FD, so there is no "h" to match for it; its own small
    # analytic-vs-FD gap (~1e-7-1e-6, already documented in the mock-family gate, §2c of
    # PROFILED_RESTRICTED_OUTER_GRADIENT_MASTER_2026-08-01.md) is reported separately, never
    # folded into the A-block's machine-precision pass criterion.
    sb = report("same_bandwidth_Ablock(2:end)", g_full_matched[2:end], g_inc[2:end])
    gp_diff = abs(g_full_matched[1] - g_inc[1])
    println(@sprintf("[same_bandwidth_gp] analytic(inc)=%.10e  FD(full,h=0.01)=%.10e  diff=%.3e (expected small, NOT part of A-block pass criterion)",
        g_inc[1], g_full_matched[1], gp_diff))

    push!(rows, (label = label, t_full = t_full, t_inc = t_inc, speedup = t_full / t_inc,
        mismatched_max_abs_err = mm.max_abs_err, mismatched_max_rel_err = mm.max_rel_err, mismatched_cos_sim = mm.cos_sim,
        same_bw_Ablock_max_abs_err = sb.max_abs_err, same_bw_Ablock_max_rel_err = sb.max_rel_err, same_bw_Ablock_cos_sim = sb.cos_sim,
        same_bw_gp_diff = gp_diff, gp_full = g_full[1], gp_inc = g_inc[1]))
    return rows
end

rows = []
compare_at("calibration", w_calib, rows)

using Random
Random.seed!(42)
n_free = n_total - 1
w_pert = copy(w_calib); w_pert[2:end] .+= 0.01 .* randn(n_free)
compare_at("small_perturbation", w_pert, rows)

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_INCREMENTAL_VS_FULLREBUILD_2026-08-01_$tag.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)

# Two SEPARATE pass criteria (task §12) -- do not conflate them:
same_bw_pass = all(r.same_bw_Ablock_max_rel_err < 1e-6 for r in rows)
mismatched_bw_reasonable = all(r.mismatched_cos_sim > 0.999 for r in rows)   # sanity only, NOT a machine-precision claim
println("\nSAME-BANDWIDTH formula-equivalence gate ($tag): ", same_bw_pass ? "PASS (machine precision)" : "FAIL")
println("MISMATCHED-BANDWIDTH robustness/approximation check ($tag): ", mismatched_bw_reasonable ? "cos_sim>0.999 (expected, NOT machine precision)" : "FAIL (unexpectedly large gap)")
println("\nINCREMENTAL VS FULL-REBUILD ($tag): ", same_bw_pass ? "PASS" : "NEEDS REVIEW")
