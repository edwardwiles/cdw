# Phase 12 (integration/phase12-13-runner-checkpoints-2026-08-02), item 2 gate: proves the
# reduced-basis cache/bank types (profiled_reduced_basis_cache_bank_2026-08-02.jl) are genuinely
# SEPARATE from the dense/full ones -- a Julia type-system guarantee, not just a runtime tag --
# plus the lookup/store/warm-start mechanics behave correctly in isolation. Pure unit-level: no
# real ctx/KNITRO solve needed.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl", "compressed_moments.jl",
          "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "nested_quantile_grids.jl", "cm_config.jl",
          "blas_thread_policy.jl", "knitro_outer_algorithm.jl", "production_backend_manifest.jl",
          "cm_exact_cache_production.jl", "incumbent_logic.jl", "cm_hessian_subblock_profiling.jl",
          "production_bundle_api.jl", "country_resolve.jl", "cm_checkpoint.jl",
          "dual_bank.jl", "cm_dual_bank_production.jl",
          "profiled_reduced_basis_cache_bank_2026-08-02.jl"]
    include(joinpath(D4X, f))
end
using Test, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

println("="^78); println("GROUP 1: constructor validation"); println("="^78)
ok = false
try
    ProfiledCMProductionEvalKey(:full_gamma_normalized, [1.0, 2.0], Float64[], 1.0, true, :ZC_only, "digestA", "fpA")
catch e
    global ok = e isa ErrorException
end
check("ProfiledCMProductionEvalKey rejects :full_gamma_normalized at construction", ok)

k_ok = ProfiledCMProductionEvalKey(:profiled_destination_scales, [1.0, 2.0], Float64[], 1.0, true, :ZC_only, "digestA", "fpA")
check(":profiled_destination_scales key constructs fine", k_ok.economic_parameterization == :profiled_destination_scales)

println("\n" * "="^78); println("GROUP 2: type-system separation (dense vs profiled key can NEVER collide)"); println("="^78)
dense_cache = cm_production_exact_cache()
profiled_cache = profiled_cm_production_exact_cache()
check("dense cache is SafeExactCache{CMProductionEvalKey}", dense_cache isa SafeExactCache{CMProductionEvalKey})
check("profiled cache is SafeExactCache{ProfiledCMProductionEvalKey}", profiled_cache isa SafeExactCache{ProfiledCMProductionEvalKey})
check("the two cache TYPES are different Julia types", typeof(dense_cache) !== typeof(profiled_cache))

dense_key = CMProductionEvalKey([1.0, 2.0], Float64[], 1.0, true, "cm_upper", :ZC_only, 10, :anchored, 0, 0, :legacy_z, "fpA")
mismatch_caught = false
try
    _cache_store!(profiled_cache, dense_key, (base = nothing, verify = (inner_status = 0,)))
catch e
    global mismatch_caught = e isa MethodError
end
check("storing a DENSE key into the PROFILED cache is a Julia MethodError (compile-time-enforced separation)", mismatch_caught)

mismatch_caught2 = false
try
    profiled_cm_cache_lookup_or_compute!(profiled_cache, dense_key, () -> (nothing, (inner_status = 0,)))
catch e
    global mismatch_caught2 = e isa MethodError
end
check("profiled_cm_cache_lookup_or_compute! rejects a dense key at the method-dispatch level", mismatch_caught2)

println("\n" * "="^78); println("GROUP 3: profiled lookup/store mechanics (miss -> compute -> store -> hit)"); println("="^78)
reset_profiled_cm_exact_cache_counters!()
key1 = ProfiledCMProductionEvalKey(:profiled_destination_scales, [1.0, 0.5, -0.2], [1.0], 1.0, true, :CM_plus_ZC, "digestB", "fpB")
n_compute_calls = Ref(0)
compute1() = (n_compute_calls[] += 1; (base = "base_payload", verify = (inner_status = 0, Delta_dual = 3.14)))
base1, verify1 = profiled_cm_cache_lookup_or_compute!(profiled_cache, key1, compute1)
check("first call computes (miss)", n_compute_calls[] == 1 && verify1.Delta_dual == 3.14)
base1b, verify1b = profiled_cm_cache_lookup_or_compute!(profiled_cache, key1, compute1)
check("second call at the SAME key hits cache (compute_fn NOT called again)", n_compute_calls[] == 1)
check("cached result matches original", verify1b.Delta_dual == 3.14)
c = PROFILED_CM_EXACT_CACHE_COUNTERS[]
check("counters: 2 lookups, 1 hit, 1 miss", c.lookups == 2 && c.hits == 1 && c.misses == 1)

key2 = ProfiledCMProductionEvalKey(:profiled_destination_scales, [9.0, 9.0], Float64[], 1.0, true, :ZC_only, "digestB", "fpB")
compute_infeasible() = (nothing, (inner_status = -300,))
profiled_cm_cache_lookup_or_compute!(profiled_cache, key2, compute_infeasible)
check("infeasible result is NOT stored (store_rejection incremented)", PROFILED_CM_EXACT_CACHE_COUNTERS[].store_rejections == 1)

println("\n" * "="^78); println("GROUP 4: dual-bank parameterization guard"); println("="^78)
pb = ProfiledRestrictedDualBank(8)
check("ProfiledRestrictedDualBank tagged :profiled_destination_scales", pb.economic_parameterization == :profiled_destination_scales)
ok_assert = (assert_bank_parameterization(pb.economic_parameterization, :profiled_destination_scales) === nothing)
check("assert_bank_parameterization passes for a matched expectation", ok_assert)
mismatch3 = false
try
    assert_bank_parameterization(pb.economic_parameterization, :full_gamma_normalized)
catch e
    global mismatch3 = e isa ErrorException
end
check("assert_bank_parameterization throws for a MISMATCHED expectation", mismatch3)

record_success_profiled!(pb, 1, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3, 0.4])
check("record_success_profiled! records into the underlying DualBank", length(pb.bank.bank.history) == 1)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
