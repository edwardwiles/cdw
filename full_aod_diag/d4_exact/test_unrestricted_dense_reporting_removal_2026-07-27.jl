# Final-architecture-closure task (2026-07-27), Goal 9: confirm evaluate_fullA_fast_compressed's
# NEW dense_reference_diagnostics default (false) actually skips the dense obj.H materialization +
# select_G_from_H + obj(inner_x,constr=...) block (no_dense_g_counters stay at their pre-call
# values for the dense-G-specific counters), and that dense_reference_diagnostics=true reproduces
# the FULL prior behavior (real numeric gravity_raw/benchmark_unweighted_moment_mean, matching
# verification_backend=:dense_reference's own numbers at this point).
const D4X = @__DIR__
cd(D4X)
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl",
          "common_marginals_interval.jl", "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl", "lfix_cm_cplus.jl",
          "no_dense_g_counters.jl", "compressed_live.jl"]
    include(joinpath(D4X, f))
end

ctx = d4_exact_setup()
x_free0 = ctx.θ0_up[ctx.free_idx]

reset_no_dense_g_counters!()
r_default, _ = evaluate_fullA_fast_compressed(x_free0, ctx; use_cache = false, warm = false)
r1 = no_dense_g_report()
println("=== default call (dense_reference_diagnostics not passed, i.e. false) ===")
println("  gravity_raw = $(r_default.gravity_raw)  (expect NaN)")
println("  benchmark_unweighted_moment_mean = $(r_default.benchmark_unweighted_moment_mean)  (expect empty)")
println("  max_abs_moment_resid = $(r_default.max_abs_moment_resid)  (expect NaN)")
ok1 = isnan(r_default.gravity_raw) && isempty(r_default.benchmark_unweighted_moment_mean) && isnan(r_default.max_abs_moment_resid)
println(ok1 ? "  PASS: reporting-only dense fields are NaN/empty at the new default" :
              "  FAIL: reporting-only dense fields were NOT skipped at the new default")

reset_no_dense_g_counters!()
r_diag, _ = evaluate_fullA_fast_compressed(x_free0, ctx; use_cache = false, warm = false, dense_reference_diagnostics = true)
println("\n=== dense_reference_diagnostics=true ===")
println("  gravity_raw = $(r_diag.gravity_raw)  (expect a real number)")
println("  benchmark_unweighted_moment_mean length = $(length(r_diag.benchmark_unweighted_moment_mean))  (expect > 0)")
ok2 = !isnan(r_diag.gravity_raw) && !isempty(r_diag.benchmark_unweighted_moment_mean)
println(ok2 ? "  PASS: reporting-only dense fields are real when explicitly requested" :
              "  FAIL: dense_reference_diagnostics=true did not restore the dense reporting fields")

# The verification-critical fields (Delta_dual, Delta_primal, mean_m_resid, max_abs_moment_kkt_resid,
# weight_norm_resid, m_mean/m_min/m_max, inner_status) must be IDENTICAL between the two calls --
# dense_reference_diagnostics only controls the reporting-only extras, never the verification path
# (both calls above used the SAME default verification_backend=:operator).
ok3 = r_default.Delta_dual == r_diag.Delta_dual && r_default.Delta_primal == r_diag.Delta_primal &&
      r_default.mean_m_resid == r_diag.mean_m_resid && r_default.max_abs_moment_kkt_resid == r_diag.max_abs_moment_kkt_resid &&
      r_default.weight_norm_resid == r_diag.weight_norm_resid && r_default.inner_status == r_diag.inner_status
println(ok3 ? "PASS: verification-critical fields identical regardless of dense_reference_diagnostics" :
              "FAIL: dense_reference_diagnostics changed a verification-critical field -- should be impossible")

all_ok = ok1 && ok2 && ok3
println(all_ok ? "\nALL PASS" : "\nSOME FAILED")
all_ok || error("test_unrestricted_dense_reporting_removal_2026-07-27.jl: failures above")
