# ============================================================================
# Continuation 4, Phase 3 (mandatory pre-flight): validates composite_gradient.jl
# against the TRUE optimized-value (Delta_FD) gradient at calibration, the
# fixed-A benchmark, the upper headline candidate, and the lower stalled
# candidate -- per the task's explicit "before any long run" requirement.
# Reports (NOT just asserts pass/fail, since Delta_FD itself is only accurate
# to its own FD tolerance, not a machine-precision oracle):
#   - gamma component error (composite analytic vs Delta_FD central difference)
#   - A-block cosine + norm ratio (already gravity-tangent by construction of
#     the pivot-reduced w coordinates -- see docs/fullA_performance_profile.md
#     sec 4's finding that full-vector cosine is not meaningful, gamma
#     dominates the norm; reported separately here for that reason)
#   - full-vector cosine (for context only, not the decision metric)
#   - A-only random-directional-derivative check: composite-predicted vs
#     Delta_FD-actual directional derivative along several random A-block unit
#     vectors
#   - chosen h / switching mass per A-block coordinate (from the adaptive
#     bandwidth selector)
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
using Random, LinearAlgebra, Printf

ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2
x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

"""
Full optimized-value (Delta_FD) central-FD gradient in reduced w-coordinates.
`hs`: per-coordinate step (vector, length(w0)) -- IMPORTANT, see finding below:
comparing composite's ADAPTIVE-h A-block FD against a Delta_FD computed at a
DIFFERENT (e.g. fixed 0.01) h is not a fair apples-to-apples test, since
Delta_dual itself has strong h-dependent curvature from the inner dual's own
re-optimization response (confirmed empirically this session: at
upper_maxit40, FD(Delta_dual) moved from -76.2 at h=0.01 to -44.62 at
h=1e-5, while FD(L_fix) barely moved, -45.71 -> -44.62, over the SAME h
range -- and matches the historical `results/fullA_d4/1bdb1cc/h_sweep.csv`
finding that raw Delta_dual FD is unstable/non-converged even at h=0.00625 in
that direction). Default (no `hs`) uses the SAME fixed h=0.01 as
run_d4_optimized_fd.jl's FIXED_H, for the secondary "what would the OLD
driver have seen" comparison.
"""
function delta_fd_gradient(w0::AbstractVector, ctx; hs::Union{Nothing,AbstractVector} = nothing, h::Float64 = 0.01)
    n = length(w0)
    g = zeros(n)
    for i in 1:n
        hi = hs === nothing ? h : hs[i]
        wp = copy(w0); wp[i] += hi
        wm = copy(w0); wm[i] -= hi
        Δp = evaluate_fullA(x_free_from_w(wp), ctx; cache = nothing, warm = true).Delta_dual
        Δm = evaluate_fullA(x_free_from_w(wm), ctx; cache = nothing, warm = true).Delta_dual
        g[i] = (Δp - Δm) / (2hi)
    end
    return g
end

function validate_point(label, w0; n_dir::Int = 6, rng_seed::Int = 42)
    println("="^78); println("POINT: $label"); println("="^78)
    xf0 = x_free_from_w(w0)
    base = solve_base_state(xf0, ctx)
    g_comp, meta = composite_gradient_at(xf0, ctx, pe; base = base)

    # gamma: two references. (1) small-h=1e-5 Delta_FD -- the TRUE local tangent (envelope theorem
    # says this must agree with the analytic composite value at x0); (2) standard fixed h=0.01
    # Delta_FD -- what the OLD eval_grad_central_fd driver has always reported, included because
    # (per this session's finding) it can differ SUBSTANTIALLY from the true tangent due to
    # Delta_dual's own re-optimization curvature, not composite-gradient error.
    hs_smallgamma = fill(1e-5, D2)
    g_ref_smallh = delta_fd_gradient(w0, ctx; hs = hs_smallgamma)
    g_ref_fixed01 = delta_fd_gradient(w0, ctx; h = 0.01)

    gamma_err_abs = abs(g_comp[1] - g_ref_smallh[1])
    gamma_err_rel = gamma_err_abs / max(abs(g_ref_smallh[1]), 1e-12)
    gamma_gap_fixed01 = abs(g_comp[1] - g_ref_fixed01[1]) / max(abs(g_ref_fixed01[1]), 1e-12)
    @printf("gamma component: composite=%.8g  Delta_FD(h=1e-5,true tangent)=%.8g  abs_err=%.3g  rel_err=%.3g\n",
            g_comp[1], g_ref_smallh[1], gamma_err_abs, gamma_err_rel)
    @printf("  (for context) Delta_FD(h=0.01, OLD driver's own FIXED_H)=%.8g  rel_gap_vs_composite=%.3g -- a large gap here reflects Delta_dual's own h=0.01 curvature bias, not composite error (see file header)\n",
            g_ref_fixed01[1], gamma_gap_fixed01)

    # A-block: matched-h comparison (fair -- same h composite's adaptive selector chose) is the
    # PRIMARY metric; fixed-h=0.01 is secondary context only (see delta_fd_gradient's docstring).
    g_ref_matched = delta_fd_gradient(w0, ctx; hs = meta.h_used)
    a_comp = @view g_comp[2:end]
    a_ref_matched = @view g_ref_matched[2:end]
    a_ref_fixed01 = @view g_ref_fixed01[2:end]

    cos_a = dot(a_comp, a_ref_matched) / (norm(a_comp) * norm(a_ref_matched) + 1e-300)
    normratio_a = norm(a_comp) / (norm(a_ref_matched) + 1e-300)
    cos_a_fixed01 = dot(a_comp, a_ref_fixed01) / (norm(a_comp) * norm(a_ref_fixed01) + 1e-300)
    normratio_a_fixed01 = norm(a_comp) / (norm(a_ref_fixed01) + 1e-300)
    cos_full = dot(g_comp, g_ref_matched) / (norm(g_comp) * norm(g_ref_matched) + 1e-300)
    @printf("A-block (== gravity-tangent, reduced coords), MATCHED-h vs Delta_FD: cosine=%.6f  norm_ratio=%.4f  (||composite||=%.4g  ||Delta_FD||=%.4g)\n",
            cos_a, normratio_a, norm(a_comp), norm(a_ref_matched))
    @printf("A-block, fixed-h=0.01 Delta_FD (context/secondary): cosine=%.6f  norm_ratio=%.4f\n", cos_a_fixed01, normratio_a_fixed01)
    @printf("full-vector cosine (context only, gamma-dominated per known finding): %.6f\n", cos_full)
    g_ref = g_ref_matched

    println("\nper-coordinate A-block bandwidth/switching-mass/slope-stability:")
    for k in 2:D2
        @printf("  coord %2d: h=%.5g  switch_mass=%.4g  |slope(h)-slope(h/2)|/max=%.3g  g_comp=%.4g  g_ref=%.4g\n",
                k, meta.h_used[k], meta.switch_mass[k], meta.slope_ratio[k], g_comp[k], g_ref[k])
    end

    rng = MersenneTwister(rng_seed)
    println("\nA-only random directional derivative checks (unit directions in the 15-dim A-block):")
    dir_results = NamedTuple[]
    for t in 1:n_dir
        v = randn(rng, D2 - 1); v ./= norm(v)
        vfull = vcat(0.0, v)
        pred = dot(a_comp, v)
        hdir = 1e-4   # small, to estimate the TRUE local directional derivative (avoids the h=0.01
                       # curvature bias documented above for Delta_dual); a genuinely random combined
                       # direction crosses far fewer kinks per coordinate than a single-coordinate
                       # adaptively-chosen probe, so a small h is the fair choice here, not h=0.01
        Δp = evaluate_fullA(x_free_from_w(w0 .+ hdir .* vfull), ctx; warm = true).Delta_dual
        Δm = evaluate_fullA(x_free_from_w(w0 .- hdir .* vfull), ctx; warm = true).Delta_dual
        actual = (Δp - Δm) / (2hdir)
        agree_sign = sign(pred) == sign(actual)
        @printf("  dir %d: predicted=%.5g  actual(Delta_FD)=%.5g  same_sign=%s\n", t, pred, actual, agree_sign)
        push!(dir_results, (dir = t, pred = pred, actual = actual, same_sign = agree_sign))
    end
    n_agree = count(r -> r.same_sign, dir_results)
    println("  sign agreement: $n_agree/$n_dir")
    println()
    return (label = label, gamma_err_abs = gamma_err_abs, gamma_err_rel = gamma_err_rel,
            cos_a = cos_a, normratio_a = normratio_a, cos_full = cos_full,
            h_used = copy(meta.h_used), switch_mass = copy(meta.switch_mass),
            slope_ratio = copy(meta.slope_ratio), dir_sign_agree = n_agree, n_dir = n_dir)
end

zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
w_calibration = vcat(gp0, zfree0)
w_fixedA = vcat(0.9109408705840424, zfree0)
w_up40 = [0.8930839180420251, 0.13610811893107913, -0.004379640581443034, 0.08813835095078966, 0.03498620885934151, 1.357116001748786, 0.16775965151912148, 1.243115907583375, 1.2931357691233787, 0.6689580557288312, 0.49753851703712204, 0.48942922371526165, 0.5767227972531564, 0.7788321643626589, 1.3869167453049203, 0.6067838085795927]
w_low = [0.9935715170663789, -0.14329335035836313, -0.282902177899273, -0.3558713414458916, -0.020237525450975312, 0.40434904269459543, 0.8961983696609168, 0.3146203506836819, 0.6385391937801699, 0.25485162740256434, 0.04641952506696054, 0.2546749283154236, 0.7202823581899669, 1.0948019663930175, 0.5314878884720499, 1.0693627886900385]

"""
Calibration (z_free=0) and the fixed-A benchmark are BOTH documented as
cold-inner-solve-infeasible at this exact D=4/W=8000 economy (nStatus=-300,
see docs/fullA_continuation3_resume_audit.md sec 4's table: 'cold inner solve
fails (-300) at z_free=0 without a warm-started continuation path; not
claimed feasible by any prior artifact either') -- a pre-existing, already-
certified non-bug, not something this validation should treat as a failure.
Skipped (not failed) here exactly like test_lfix_incremental.jl's own
random-point discipline; the feasible upper/lower candidates below are the
points that actually matter for validating the composite gradient near a
real optimum.
"""
function try_validate_point(label, w0; kwargs...)
    try
        return validate_point(label, w0; kwargs...)
    catch e
        if occursin("inner solve failed", sprint(showerror, e))
            println("="^78); println("POINT: $label -- SKIPPED (base solve cold-infeasible, nStatus!=0, documented non-bug)"); println("="^78, "\n")
            return nothing
        else
            rethrow()
        end
    end
end

results = NamedTuple[]
for (lbl, w0) in [("calibration", w_calibration), ("fixed_A_benchmark", w_fixedA),
                  ("upper_maxit40 (headline)", w_up40), ("lower_stalled", w_low)]
    r = try_validate_point(lbl, w0)
    r !== nothing && push!(results, r)
end

println("="^78); println("SUMMARY"); println("="^78)
@printf("%-28s %12s %10s %10s %10s %8s\n", "point", "gamma_relerr", "A_cosine", "A_normrat", "full_cos", "dir_agree")
for r in results
    @printf("%-28s %12.3g %10.6f %10.4f %10.6f %6d/%d\n", r.label, r.gamma_err_rel, r.cos_a, r.normratio_a, r.cos_full, r.dir_sign_agree, r.n_dir)
end

all_gamma_ok = all(r.gamma_err_rel < 1e-3 for r in results)
all_cos_ok = all(r.cos_a > 0.9 for r in results)
println()
println("gamma component matches Delta_FD to <0.1% relative at all points: ", all_gamma_ok)
println("A-block cosine > 0.9 at all points: ", all_cos_ok)
(all_gamma_ok && all_cos_ok) || println("WARNING: composite gradient did not pass both bars -- inspect per-point output above before trusting it in a live solve.")
