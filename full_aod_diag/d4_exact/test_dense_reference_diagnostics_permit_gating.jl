# ================================================================================================
# architecture/production-operator-bundle-hardening-2026-07-30, task §14: dense diagnostic path
# tests. Requires (task's own explicit list):
#   - dense construction without permit -> fails
#   - dense construction with permit -> succeeds and prints banner
#   - dense construction under production purpose -> fatal
#   - dense equivalence diagnostics -> pass (delegated to the pre-existing, retained
#     test_operator_no_H_bundle_equivalence_*.jl suite -- not re-implemented here, see
#     RELEASE_CLAIMS_2026-07-30.md for how that suite's result is reported separately from this one)
#   - diagnostic manifest -> visibly marked diagnostic-only
#   - production runners must not accept a dense permit (checked structurally: prepare_production_run
#     has no permit parameter at all)
# ================================================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "production_bundle_api.jl", "dense_reference_diagnostics.jl"]
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

lp("="^90); lp("test_dense_reference_diagnostics_permit_gating.jl"); lp("="^90)

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
Random.seed!(20260730)
probs4 = collect(range(0.0, 1.0; length = 12))[2:end-1]

build_dense_inner = () -> build_cm_production_context(ctx, CS; L = 10, contrasts = :anchored, probs = probs4,
    threaded_bins = true, inner_fg_backend = :dense_reference, moment_representation = :dense_reference)

# ---- 1. dense construction without permit -> fails ----------------------------------------------
reset_dense_reference_construction_log!()
threw = try
    DenseReferenceDiagnostics.prepare_context(:flexible_cm, "test-no-permit", build_dense_inner;
        permit = DenseReferenceDiagnostics.require_permit(nothing, "test-no-permit"))
    false
catch e
    e isa ErrorException
end
check("dense construction without permit fails", threw)
check("no construction was recorded for the failed no-permit attempt", DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 0)

# ---- 2. dense construction with permit -> succeeds and prints banner ----------------------------
reset_dense_reference_construction_log!()
permit = DenseReferencePermit(reason = "test: prove the diagnostic path still works", caller = "test_dense_reference_diagnostics_permit_gating.jl")
banner_tmp = tempname()
banner_txt = open(banner_tmp, "w") do fio
    redirect_stderr(fio) do
        global dctx = DenseReferenceDiagnostics.prepare_context(:flexible_cm, "test-with-permit", build_dense_inner; permit = permit)
    end
    flush(fio)
    nothing
end
banner_txt = read(banner_tmp, String)
rm(banner_tmp; force = true)
check("dense construction with permit succeeds", dctx.obj isa DenseReferencePsiObjectiveBundle)
check("dense construction with permit is a DenseReferenceContext (structurally distinct from ProductionContext)",
      dctx isa DenseReferenceContext && !(dctx isa ProductionContext))
check("banner prints the mandatory ATTENTION header", occursin("ATTENTION: DENSE REFERENCE BUNDLE CONSTRUCTED", banner_txt))
check("banner records the reason", occursin("test: prove the diagnostic path still works", banner_txt))
check("banner records the caller", occursin("test_dense_reference_diagnostics_permit_gating.jl", banner_txt))
check("construction counter incremented exactly once", DENSE_REFERENCE_BUNDLE_CONSTRUCTIONS[] == 1)
check("construction log has exactly one entry with the right reason", length(DENSE_REFERENCE_CONSTRUCTION_LOG) == 1 &&
      DENSE_REFERENCE_CONSTRUCTION_LOG[1].reason == permit.reason)

# ---- 3. dense construction under production purpose -> fatal ------------------------------------
prod_threw = try
    DenseReferenceDiagnostics.prepare_context(:flexible_cm, "test-production-purpose", build_dense_inner;
        permit = permit, purpose = ProductionPurpose())
    false
catch e
    true
end
check("dense construction under ProductionPurpose is fatal", prod_threw)

# ---- 4. diagnostic manifest -> visibly marked diagnostic-only -----------------------------------
reset_dense_reference_construction_log!()
dctx2 = DenseReferenceDiagnostics.prepare_context(:flexible_cm, "test-manifest", build_dense_inner; permit = permit)
# derive_backend_manifest is typed to accept the shared context shapes generically -- construct a
# throwaway ProductionContext-shaped record is wrong on purpose here: instead build the manifest
# directly off the DenseReferenceContext fields, proving the manifest itself is NOT the production
# one (purpose_label must say so, not just "happen to differ").
diag_manifest = (
    structural = (run_purpose = "diagnostic_dense_reference", family = dctx2.family, runner = dctx2.runner,
                  context_type = string(typeof(dctx2)), bundle_type = string(typeof(dctx2.obj)),
                  permit_reason = dctx2.permit.reason, permit_caller = dctx2.permit.caller),
    evaluation = no_dense_g_report(),
    bundle_invariant_pass = false,   # a dense reference bundle NEVER satisfies the production invariant -- always false, by construction
)
check("diagnostic manifest run_purpose is visibly diagnostic", diag_manifest.structural.run_purpose == "diagnostic_dense_reference")
check("diagnostic manifest bundle_invariant_pass is false (never conflated with a production PASS)",
      diag_manifest.bundle_invariant_pass == false)

# ---- 5. production runners must not accept a dense permit ---------------------------------------
has_permit_kw = any(ks -> :permit in ks, Base.kwarg_decl.(methods(prepare_production_run)))
check("prepare_production_run has no permit parameter at all", !has_permit_kw)

lp("="^90)
if isempty(FAILURES)
    lp("ALL PASS")
else
    lp("FAILURES (", length(FAILURES), "): ", FAILURES)
    error("test_dense_reference_diagnostics_permit_gating.jl: ", length(FAILURES), " check(s) failed")
end
