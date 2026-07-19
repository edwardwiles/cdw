# ============================================================================
# Continuation 10, Part 1: correctness check for hessian_chunked! (chunked_hessian.jl)
# against the production dense baseline (cc_algo/PsiObjectiveBundle.jl's
# hessian!), at D=4 (synthetic, fast) first, then D=20 (real data, W=80,000).
#
# For each chunk size tested, checks (at the SAME θ_full, SAME draws, SAME
# converged inner-dual point x*):
#   1. Hessian bit-identical (or within ~1e-10 relative) vs the baseline's
#      packed-triangular `h` output, computed by calling hessian!/hessian_chunked!
#      directly at the SAME (obj.arg0 refreshed from) x -- isolates the
#      Hessian-FORMING step alone, no KNITRO solve involved.
#   2. Full inner solve (inner_loop_internal_profiled vs inner_loop_internal_chunked)
#      gives IDENTICAL nStatus, n_fg_calls, n_hess_calls (iteration count),
#      Delta_dual, and dual solution x* (zeta, lambda) -- confirms the chunked
#      Hessian callback, wired into a real KNITRO solve, reproduces the exact
#      same converged point (not just that the Hessian FORMULA matches in
#      isolation).
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "chunked_hessian.jl"))
using Printf, Statistics

function rel_or_abs_diff(a, b)
    d = abs(a - b)
    denom = max(abs(a), abs(b), 1.0)
    return d / denom
end

function check_hessian_formula_matches(obj, θ_full, chunk_sizes::Vector{Int}; label::String = "")
    # Build obj.H once (moments!), set up a converged-ish x via a normal profiled solve,
    # then compare hessian! vs hessian_chunked! evaluated AT THE SAME x (same obj.arg0),
    # isolating the Hessian-forming step from the KNITRO iteration path itself.
    K_ref, x_ref, status_ref, nfg_ref, nhess_ref = inner_loop_internal_profiled(obj, θ_full)
    status_ref in (0, -100, -101, -103) || error("$label: reference solve did not converge, status=$status_ref")

    # Refresh obj.arg0 at x_ref (as the production callable would before hessian!)
    _prep_for_hessian!(obj, x_ref)
    n = obj.outer_constr_index
    ntri = div(n * (n + 1), 2)
    h_ref = zeros(ntri)
    CS.hessian!(h_ref, obj)

    println(label, ": baseline solve status=", status_ref, " n_fg=", nfg_ref, " n_hess=", nhess_ref)

    for cs in chunk_sizes
        _prep_for_hessian!(obj, x_ref)   # refresh arg0/arg2 identically before each comparison
        h_c = zeros(ntri)
        hessian_chunked!(h_c, obj, cs)
        maxabsdiff = maximum(abs.(h_c .- h_ref))
        maxreldiff = maximum(rel_or_abs_diff.(h_c, h_ref))
        bit_identical = h_c == h_ref
        @printf("  chunk_size=%6d  bit_identical=%s  max|diff|=%.3e  max_rel_diff=%.3e\n",
                cs, bit_identical, maxabsdiff, maxreldiff)
        maxreldiff < 1e-10 || error("$label chunk_size=$cs: Hessian mismatch too large ($maxreldiff)")
    end
    return x_ref, status_ref, nfg_ref, nhess_ref, K_ref
end

function check_full_inner_solve_matches(obj, θ_full, chunk_sizes::Vector{Int}; label::String = "")
    obj.x .= NaN
    K_ref, x_ref, status_ref, nfg_ref, nhess_ref = inner_loop_internal_profiled(obj, θ_full)
    println(label, " (full inner solve): baseline status=", status_ref, " n_fg=", nfg_ref,
            " n_hess=", nhess_ref, " K=", K_ref)

    for cs in chunk_sizes
        obj.x .= NaN
        K_c, x_c, status_c, nfg_c, nhess_c = inner_loop_internal_chunked(obj, θ_full, cs)
        dK = abs(K_c - K_ref)
        dx = maximum(abs.(x_c .- x_ref))
        @printf("  chunk_size=%6d  status=%d(ref %d)  n_fg=%d(ref %d)  n_hess=%d(ref %d)  |dK|=%.3e  max|dx|=%.3e\n",
                cs, status_c, status_ref, nfg_c, nfg_ref, nhess_c, nhess_ref, dK, dx)
        status_c == status_ref || error("$label chunk_size=$cs: status mismatch")
        nfg_c == nfg_ref || error("$label chunk_size=$cs: n_fg_calls mismatch ($nfg_c vs $nfg_ref)")
        nhess_c == nhess_ref || error("$label chunk_size=$cs: n_hess_calls mismatch ($nhess_c vs $nhess_ref)")
        dK < 1e-10 || error("$label chunk_size=$cs: K mismatch too large ($dK)")
        dx < 1e-8 || error("$label chunk_size=$cs: dual solution mismatch too large ($dx)")
    end
end

println("="^90)
println("D=4 correctness check")
println("="^90)
ctx4 = d4_exact_setup(find_smallest = true)
obj4 = ctx4.obj
xf4 = ctx4.θ0_up[ctx4.free_idx]
θ4 = CS.reconstruct_full(xf4, ctx4.m)
check_hessian_formula_matches(obj4, θ4, [1, 2, 5, 8000]; label = "D4-formula")
check_full_inner_solve_matches(obj4, θ4, [1, 2, 5, 8000]; label = "D4-fullsolve")

println("\nD=4 correctness check PASSED")
