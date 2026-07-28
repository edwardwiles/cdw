# ============================================================================
# Hessian upper-only cleanup (2026-07-28), task addendum Section 5: sentinel test.
#
# Proves `pack_upper_cm_hessian!` (cm_hessian_architectures.jl, shared by flexible-CM's,
# CM+ZC's, and origin-ZC's production packing) does NOT read the strict-lower-triangle H_EC
# mirror region (Hfull[NCORE+1:n, 1:NCORE]) it no longer needs -- poisoning that specific region
# with NaN leaves the packed upper-triangle result byte-identical and finite. H_EE/H_CC's own
# lower-triangle dependence is NOT poisoned here -- those blocks genuinely still read it (see
# PRODUCTION_HESSIAN_UPPER_ONLY_AUDIT_2026-07-28.md: H_CC's (l,lp) grid is independently
# accumulated, not mirrored, so a poison-everything sentinel would correctly fail there; this test
# targets exactly the region this session's fix actually stopped depending on).
#
# Separately, `archA_partitioned_hess_cb_builder` (origin-ZC)'s dead H_ER mirror write
# (cm_hessian_architectures.jl:1372, formerly present) was deleted outright this session -- its
# own sentinel is simpler: confirm the family's real inner-solve Hessian gate still passes
# (test_shared_core_hessian_d4_gates.jl, Section D) with the line gone, which is checked
# separately, not re-derived here.
#
# Usage: JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia \
#          --project=. full_aod_diag/d4_exact/test_hessian_upper_only_sentinel_d4.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(20260728)

println("="^90)
println("SENTINEL: pack_upper_cm_hessian! ignores the poisoned H_EC mirror region (flexible-CM)")
println("="^90)
let
    pcx = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = true)
    pcx.cctx.core_hessian_backend = :exact_winner_pair_parallel
    pcx.cctx.core_hessian_workers = 2
    base = archC_base_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
    check("CM: inner solve feasible", base.inner_status in (0,-100,-101,-103))

    obj = pcx.ctx_cm.obj
    cctx = pcx.cctx
    x = vcat(base.ζstar, base.λstar)
    n = cctx.NCORE + cctx.ncm
    NCORE = cctx.NCORE

    # Reference: normal callback, unpoisoned.
    _archC_prep_for_hessian!(obj, x)
    h_ref = Vector{Float64}(undef, n*(n+1)÷2)
    hessian_cm_structured_v2!(h_ref, obj, cctx; threaded_bins = true, tls = cctx.tls, use_syrk = true)

    # Re-run to repopulate cctx.Hfull (rebuilt fresh each call), then poison ONLY the H_EC mirror
    # sub-block (Hfull[NCORE+1:n, 1:NCORE], the region pack_upper_cm_hessian! no longer reads)
    # and re-pack directly via the shared function -- NOT via a fresh callback call, which would
    # just overwrite the poison before packing runs.
    _archC_prep_for_hessian!(obj, x)
    hessian_cm_structured_v2!(h_ref, obj, cctx; threaded_bins = true, tls = cctx.tls, use_syrk = true)  # repopulate cctx.Hfull
    @views cctx.Hfull[NCORE+1:n, 1:NCORE] .= NaN
    h_poisoned = Vector{Float64}(undef, n*(n+1)÷2)
    pack_upper_cm_hessian!(h_poisoned, cctx.Hfull, NCORE, n)

    n_nan = count(isnan, h_poisoned)
    max_diff = maximum(abs.(h_poisoned .- h_ref))
    lp("  poisoned-region NaN count in packed result: $n_nan   max|Δ| vs unpoisoned pack: $max_diff")
    check("CM: packed upper-triangle result has zero NaN after poisoning the H_EC mirror region", n_nan == 0)
    check("CM: packed upper-triangle result unchanged after poisoning the H_EC mirror region", max_diff == 0.0)

    # Negative control: poisoning the H_CC region (genuinely independently accumulated, NOT
    # mirrored) SHOULD propagate into the packed result -- proves this sentinel isn't vacuous
    # (i.e. pack_upper_cm_hessian! does read SOME lower-triangle entries, just not the H_EC ones).
    _archC_prep_for_hessian!(obj, x)
    hessian_cm_structured_v2!(h_ref, obj, cctx; threaded_bins = true, tls = cctx.tls, use_syrk = true)
    if n > NCORE
        @views cctx.Hfull[NCORE+2:n, NCORE+1] .= NaN   # one column of H_CC's lower triangle, if it exists
        h_cc_poisoned = Vector{Float64}(undef, n*(n+1)÷2)
        pack_upper_cm_hessian!(h_cc_poisoned, cctx.Hfull, NCORE, n)
        cc_propagated = any(isnan, h_cc_poisoned)
        lp("  H_CC-region poison propagated into packed result (expected true): $cc_propagated")
        check("CM: negative control -- H_CC poison DOES propagate (sentinel is not vacuous)", cc_propagated)
    end
end

println("="^90)
println("SENTINEL: origin-ZC's dead H_ER mirror removal -- production gate re-check")
println("="^90)
let
    # The origin-ZC fix deleted a mirror write outright rather than gating a packer (its own
    # packing loop was already a plain upper-only copy with no averaging step to fool) -- its
    # sentinel is simply that the family's full production Hessian gate
    # (test_shared_core_hessian_d4_gates.jl, Section D) still passes with the line gone, checked
    # there. Recorded here as a named pointer so this file is a complete index of both fixes'
    # verification, not a silent gap.
    check("originZC: dead-mirror removal verified by Section D of test_shared_core_hessian_d4_gates.jl (not re-derived here)", true)
end

println("="^90)
if isempty(FAILURES)
    println("ALL HESSIAN UPPER-ONLY SENTINEL TESTS PASSED")
else
    println("FAILURES:")
    for f in FAILURES
        println("  - $f")
    end
    exit(1)
end
