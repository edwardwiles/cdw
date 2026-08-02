# ============================================================================
# Claude Code task 2026-08-01 (parallel outer-gradient workstream), §11 "Mock
# restricted layouts": construct synthetic family layouts with the same
# economic dual slice, arbitrary restriction dual slices, and arbitrary
# constant restriction contributions. Two DISTINCT, honest claims are tested
# separately here (see PROFILED_RESTRICTED_OUTER_GRADIENT_MASTER_2026-08-01.md
# section "on the restriction-invariance claim" for why they are kept
# separate rather than one blanket "any restriction value leaves the gradient
# unchanged" claim, which is NOT true in general once Psi's nonlinearity is
# accounted for):
#
#   (A) CODE-LEVEL NO-COUPLING: the shared engine's output depends on
#       `ev.result.beta` ONLY through `economic_dual_range(fctx)` and the
#       caller-supplied `restriction_contrib0(fctx,ev)` scalar-per-draw term
#       -- never through the raw restriction-labeled beta entries directly.
#       Proved by holding `restriction_contrib0` FIXED while randomizing the
#       "junk" restriction beta slice the shared engine never reads: gradient
#       must be BIT-IDENTICAL.
#   (B) NUMERICAL CORRECTNESS OF THE FIXED-RESTRICTION TREATMENT: given a
#       REALISTIC-SCALE restriction contribution (comparable magnitude to the
#       unrestricted engine's own existing const_part/cf_raw_κcf baseline
#       terms), the shared engine's O(1)-incremental gradient (which folds
#       restriction_contrib0 into q0 once) must match the INDEPENDENT
#       full-rebuild reference (profiled_restricted_full_rebuild_gradient_
#       reference_2026-08-01.jl, which also holds it fixed but recomputes
#       everything else from scratch every probe) to machine precision --
#       validating the shared cache's incremental SHORTCUT, not the
#       philosophical claim that the gradient is independent of the
#       restriction contribution's magnitude.
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
include(joinpath(@__DIR__, "profiled_outer_gradient_layout_contract_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_shared_economic_gradient_engine_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_family_adapters_2026-08-01.jl"))
include(joinpath(@__DIR__, "profiled_restricted_full_rebuild_gradient_reference_2026-08-01.jl"))
using LinearAlgebra, Printf, CSV, DataFrames, Random

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
spec, gauge, pe = build_profiled_ab_spec_pe(ctx; global_overrides = Dict(3 => 1))
w_calib = reduce_calibration_to_w_profiled(ctx, pe)
ev = evaluate_profiled_point(w_calib, ctx, spec, pe)
ufctx = build_unrestricted_family_ctx(ctx, spec, pe, ev)
W = ev.st.cf.W
println("D4 calibration: Delta_dual=$(ev.result.Delta_dual)  W=$W"); flush(stdout)

rows = NamedTuple[]
families = (:flexible_CM, :common_Frechet, :ZC_only, :CM_plus_ZC)
Random.seed!(7)

# ---- (A) code-level no-coupling: hold restriction_contrib0 fixed, randomize
#      the raw restriction beta slice the shared engine never reads directly.
println("\n" * "="^90); println("(A) CODE-LEVEL NO-COUPLING (restriction beta slice never read directly)"); println("="^90); flush(stdout)
for fam in families
    fixed_rc0 = 0.001 .* randn(W)  # same magnitude class as const_part/cf_raw_κcf terms
    mfctx = build_mock_restricted_family_ctx(ufctx, fam; n_restriction = 5, rc0_fn = (f, e) -> fixed_rc0)

    beta1 = mock_extended_beta(ev, 5; scale = 1.0)
    beta2 = mock_extended_beta(ev, 5; scale = 1e6)   # wildly different restriction beta "junk"
    ev1 = ev_with_beta(ev, beta1)
    ev2 = ev_with_beta(ev, beta2)

    g1, _ = shared_family_outer_gradient(w_calib, ctx, mfctx, ev1)
    g2, _ = shared_family_outer_gradient(w_calib, ctx, mfctx, ev2)
    bit_identical = g1 == g2
    println(rpad(string(fam), 16), " no-coupling bit_identical=", bit_identical); flush(stdout)
    push!(rows, (family = fam, test = "A_no_coupling_bit_identical", pass = bit_identical,
        detail = "beta restriction slice scale 1.0 vs 1e6, rc0 held fixed"))
end

# ---- (B) numerical correctness: shared engine vs independent full-rebuild
#      reference, realistic-scale restriction contribution, MATCHED
#      per-coordinate bandwidth (task §10's own literal steps: rebuild at
#      x+h*e_k and x-h*e_k using the SAME h the method under test used --
#      an apples-to-oranges bandwidth (e.g. this reference's default fixed
#      h=0.01 vs the shared engine's own adaptive per-coordinate h) reproduces
#      only the ALREADY-KNOWN, pre-existing incremental-vs-full-rebuild gap
#      (cos_sim ~0.9999, NOT a defect -- see this file's own header and the
#      master doc). Coordinate 1 (gp) is EXACT ANALYTIC in the shared engine
#      but FD in this reference regardless of h, so it is reported but
#      excluded from the strict machine-precision pass/fail (matches the
#      already-established ~1e-6-level gp analytic-vs-FD gap the unrestricted
#      gate has always reported).
println("\n" * "="^90); println("(B) SHARED ENGINE vs FULL-REBUILD REFERENCE, MATCHED BANDWIDTH (realistic-scale restriction_contrib0)"); println("="^90); flush(stdout)
for fam in families
    rc0 = 0.001 .* randn(W)
    mfctx = build_mock_restricted_family_ctx(ufctx, fam; n_restriction = 5, rc0_fn = (f, e) -> rc0)
    beta_ext = mock_extended_beta(ev, 5)
    ev_ext = ev_with_beta(ev, beta_ext)

    t1 = time()
    g_shared, meta_shared = shared_family_outer_gradient(w_calib, ctx, mfctx, ev_ext)
    t_shared = time() - t1

    h_matched = copy(meta_shared.h_used); h_matched[1] = 0.01  # gp: any nonzero placeholder, excluded from the gate below
    t2 = time()
    g_full, _ = diag_profiled_full_rebuild_gradient(w_calib, ctx, mfctx, ev_ext; h = h_matched)
    t_full = time() - t2

    diff = g_shared[2:end] .- g_full[2:end]
    max_abs_err = maximum(abs.(diff))
    max_rel_err = maximum(abs.(diff) ./ max.(abs.(g_full[2:end]), 1e-8))
    cos_sim = dot(g_shared, g_full) / (norm(g_shared) * norm(g_full) + 1e-300)
    gp_diff = abs(g_shared[1] - g_full[1])
    println(@sprintf("%-16s A-block(2:end): max_abs_err=%.4e max_rel_err=%.4e  full cos_sim=%.10f  gp: shared=%.6e full_FD=%.6e diff=%.3e  t_shared=%.4fs t_full=%.4fs",
        string(fam), max_abs_err, max_rel_err, cos_sim, g_shared[1], g_full[1], gp_diff, t_shared, t_full)); flush(stdout)
    push!(rows, (family = fam, test = "B_shared_vs_fullrebuild_matched_bandwidth", pass = max_rel_err < 1e-8,
        detail = @sprintf("A-block(2:end) max_abs_err=%.3e max_rel_err=%.3e; gp shared=%.6e full_FD=%.6e diff=%.3e (excluded from pass/fail, analytic-vs-FD)",
            max_abs_err, max_rel_err, g_shared[1], g_full[1], gp_diff)))
end

# ---- restriction contribution genuinely changes the BASELINE gradient
#      (documents the honest boundary of the cancellation claim: this is
#      NOT expected to be zero, and a zero result here would indicate the
#      mock's rc0 isn't actually being used).
println("\n" * "="^90); println("(sanity) restriction_contrib0 MAGNITUDE does perturb the gradient baseline (expected, documents the honest scope of task §3's theorem)"); println("="^90); flush(stdout)
mfctx0 = build_mock_restricted_family_ctx(ufctx, :flexible_CM; n_restriction = 5, rc0_fn = (f, e) -> zeros(W))
mfctx1 = build_mock_restricted_family_ctx(ufctx, :flexible_CM; n_restriction = 5, rc0_fn = (f, e) -> fill(0.05, W))
beta_ext = mock_extended_beta(ev, 5)
ev_ext = ev_with_beta(ev, beta_ext)
g0, _ = shared_family_outer_gradient(w_calib, ctx, mfctx0, ev_ext)
g1, _ = shared_family_outer_gradient(w_calib, ctx, mfctx1, ev_ext)
differs = g0 != g1
println("zero-rc0 vs nonzero-rc0 gradients differ (expected): $differs"); flush(stdout)
push!(rows, (family = :flexible_CM, test = "sanity_rc0_magnitude_perturbs_baseline", pass = differs,
    detail = "documents that gradient DOES depend on rc0's value through Psi nonlinearity -- expected, not a bug"))

df = DataFrame(rows)
outpath = joinpath(@__DIR__, "..", "..", "PROFILED_RESTRICTION_MOCK_FAMILY_GATE_2026-08-01.csv")
CSV.write(outpath, df)
println("\nWrote $outpath")
println(df)
all_pass = all(r.pass for r in rows)
println("\nMOCK FAMILY GATE: ", all_pass ? "PASS" : "FAIL")
all_pass || error("mock family gate failed")
