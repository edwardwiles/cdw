# ============================================================================
# Regression tests for the verifier-underflow-fix-2026-08-04 branch.
#
# Background: the OLD `classify_inner_result` (oracle.jl) required strict `mmin >
# tol.m_min_floor`. At extreme but mathematically valid solutions, a far-tail Monte Carlo draw's
# recovered weight (m = dPsi(r) = exp(r) for r<=1) underflows to exactly Float64 0.0 for r roughly
# below -745 (exp(-745) is already smaller than the smallest representable positive double), even
# though m is mathematically strictly positive there. The old gate rejected these as if m<=0 were
# itself evidence of a bad solve. The fix (ported from campaign branch commits 5eb8c99/6ade48d):
# relax to `mmin >= tol.m_min_floor`, and add `m_weights_all_finite` (all(isfinite, m_weights)) as
# the real safeguard against genuine NaN/Inf failures, which the strict `>` never actually caught
# (m is provably >=0 by construction from dPsi!, so `m<=0` was never a real defect signal).
#
# This file tests, in order:
#   1. dPsi!/verify_namedtuple_from_operator mechanics: r in {-700,-750,-999.4} really does
#      underflow to m=0.0 exactly, while m_weights_all_finite stays true and the new diagnostic
#      fields (underflow_zero_count, r_min, r_max, m_weights_all_nonnegative) report correctly.
#   2. classify_inner_result / verification_rejection_reasons: benign underflow (m_min=0.0, all
#      residuals passing) is ACCEPTED (VerifiedSolved, empty rejection reasons).
#   3. classify_inner_result / verification_rejection_reasons: genuine failures (NaN/Inf weight,
#      manually-supplied negative weight, nonfinite Delta, failed gap/normalization/KKT residual)
#      are each individually REJECTED with the correct itemized reason.
#   4. Real campaign-checkpoint replays are NOT run from this branch (this branch has no
#      checkpoint data -- see companion note in the production-fix report): they were run
#      directly on the campaign worktree, which already carries logically-equivalent
#      oracle.jl/operator_verification.jl code (commits 5eb8c99/6ade48d) plus this branch's
#      diagnostic-field additions ported over at merge time.
#
# Run standalone:
#   julia --project=. full_aod_diag/d4_exact/test_verifier_underflow_fix_2026-08-04.jl
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "infeasibility_screen.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))

lp(xs...) = (println(xs...); flush(stdout))

n_pass = 0; n_fail = 0
function check(name, cond)
    global n_pass, n_fail
    if cond
        n_pass += 1
        println("  PASS: ", name)
    else
        n_fail += 1
        println("  FAIL: ", name)
    end
end

# A mock `obj` exposing only the `dPsi!` field verify_namedtuple_from_operator actually reads --
# deliberately NOT a real KNITRO callback object, since this section tests dPsi!'s own underflow
# math + the verification NamedTuple's field computations, not an end-to-end inner solve.
mock_obj = (dPsi! = dPsi!,)

println("\n== 1. dPsi!/verify_namedtuple_from_operator: benign far-tail underflow mechanics ==")
# W=100,000-style draw vector: mostly well-scaled r near 0, plus 3 explicit far-tail draws at
# -700/-750/-999.4 (the exact value from the real cm_meanzc K=3 point that triggered this whole
# investigation -- see operator_verification.jl's own comment).
W_mock = 100_000
r_benign = vcat(fill(0.0, W_mock - 3), [-700.0, -750.0, -999.4])
ov_benign = (r = r_benign, f = -0.01, kkt_resid = 1e-9)
m_benign, verify_benign = verify_namedtuple_from_operator(ov_benign, mock_obj, W_mock, 0)
# exp(x) underflows to exactly Float64 0.0 only below roughly x<-744.4 (ln of the smallest
# subnormal double, ~4.94e-324) -- so of the task's literal {-700,-750,-999.4} trio, -700 is a
# real, tiny, but STILL REPRESENTABLE positive weight (exp(-700)~9.86e-305), while -750/-999.4
# genuinely underflow to 0.0. This is exactly the "some Float64 weights equal 0.0" the task spec
# asks for, not "every listed value underflows" -- asserted precisely rather than loosely.
check("r=-700 does NOT underflow (still representable, ~9.86e-305 > 0)", m_benign[end-2] > 0.0 && isfinite(m_benign[end-2]))
check("r=-750 underflows to m=0.0 exactly", m_benign[end-1] == 0.0)
check("r=-999.4 underflows to m=0.0 exactly", m_benign[end] == 0.0)
check("underflow_zero_count == 2 (only -750 and -999.4 underflow; -700 does not)", verify_benign.underflow_zero_count == 2)
check("m_min == 0.0", verify_benign.m_min == 0.0)
check("m_weights_all_finite is STILL true (underflow is not a NaN/Inf)", verify_benign.m_weights_all_finite == true)
check("m_weights_all_nonnegative is true (dPsi! is provably >=0)", verify_benign.m_weights_all_nonnegative == true)
check("r_min == -999.4", verify_benign.r_min == -999.4)
check("r_max == 0.0", verify_benign.r_max == 0.0)

println("\n== 2. dPsi!/verify_namedtuple_from_operator: NaN/Inf genuinely propagate (not underflow) ==")
r_nan = vcat(fill(0.0, W_mock - 1), [NaN])
_, verify_nan = verify_namedtuple_from_operator((r = r_nan, f = -0.01, kkt_resid = 1e-9), mock_obj, W_mock, 0)
check("NaN in r -> m_weights_all_finite == false", verify_nan.m_weights_all_finite == false)

r_inf = vcat(fill(0.0, W_mock - 1), [Inf])
_, verify_inf = verify_namedtuple_from_operator((r = r_inf, f = -0.01, kkt_resid = 1e-9), mock_obj, W_mock, 0)
check("+Inf in r -> m_weights_all_finite == false", verify_inf.m_weights_all_finite == false)
check("+Inf in r -> m_max is +Inf (propagates, not silently clipped)", isinf(verify_inf.m_max))

println("\n== 3. classify_inner_result: benign underflow is ACCEPTED ==")
# Directly constructed `result` NamedTuple mirroring what a fully-converged real solve with one
# far-tail underflowed weight looks like: m_min=0.0, m_weights_all_finite=true, all residuals
# passing tolerance (this is the exact shape of the real rejected cm_meanzc K=3 / origin_zc
# lower delta=1 points described in commit 6ade48d -- status feasible, residuals excellent, only
# the old strict m_min>0 check failed).
benign_underflow_rec = (inner_status = 0, Delta_dual = 0.0007790123, primal_dual_gap = 1e-7,
    mean_m_resid = 1e-9, max_abs_moment_kkt_resid = 1e-8,
    m_min = 0.0, m_weights_all_finite = true)
check("benign underflow (m_min=0.0) classifies VerifiedSolved", classify_inner_result(benign_underflow_rec) == VerifiedSolved)
check("benign underflow (m_min=0.0) IS cacheable", is_cacheable_result(benign_underflow_rec))
check("benign underflow (m_min=0.0) IS a verified success", is_verified_success(benign_underflow_rec))
check("benign underflow has ZERO rejection reasons", isempty(verification_rejection_reasons(benign_underflow_rec)))

for status in (0, -100, -101, -103)
    rec = merge(benign_underflow_rec, (inner_status = status,))
    check("benign underflow accepted at inner_status=$status too", classify_inner_result(rec) == VerifiedSolved)
end

println("\n== 4. classify_inner_result / verification_rejection_reasons: genuine failures REJECTED ==")
base_ok = (inner_status = 0, Delta_dual = 0.001, primal_dual_gap = 1e-7,
    mean_m_resid = 1e-9, max_abs_moment_kkt_resid = 1e-8, m_min = 0.5, m_weights_all_finite = true)
check("sanity: base_ok record itself IS VerifiedSolved", classify_inner_result(base_ok) == VerifiedSolved)

nan_weight_rec = merge(base_ok, (m_weights_all_finite = false,))
check("m_weights_all_finite=false (NaN weight) is REJECTED", classify_inner_result(nan_weight_rec) == ApproximateSolved)
check("... with reason :m_weights_not_all_finite", :m_weights_not_all_finite in verification_rejection_reasons(nan_weight_rec))

neg_weight_rec = merge(base_ok, (m_min = -0.25,))
check("manually-supplied negative m_min is REJECTED", classify_inner_result(neg_weight_rec) == ApproximateSolved)
check("... with reason :m_min_below_floor", :m_min_below_floor in verification_rejection_reasons(neg_weight_rec))

nan_delta_rec = merge(base_ok, (Delta_dual = NaN,))
check("nonfinite Delta_dual is REJECTED", classify_inner_result(nan_delta_rec) == ApproximateSolved)
check("... with reason :delta_dual_nonfinite", :delta_dual_nonfinite in verification_rejection_reasons(nan_delta_rec))

nan_mmin_rec = merge(base_ok, (m_min = NaN,))
check("nonfinite m_min is REJECTED", classify_inner_result(nan_mmin_rec) == ApproximateSolved)
check("... with reason :m_min_nonfinite", :m_min_nonfinite in verification_rejection_reasons(nan_mmin_rec))

failed_gap_rec = merge(base_ok, (primal_dual_gap = 10.0,))
check("failed primal-dual gap is REJECTED", classify_inner_result(failed_gap_rec) == ApproximateSolved)
check("... with reason :primal_dual_gap_exceeds_tol", :primal_dual_gap_exceeds_tol in verification_rejection_reasons(failed_gap_rec))

failed_norm_rec = merge(base_ok, (mean_m_resid = 1.0,))
check("failed normalization (mean_m_resid) is REJECTED", classify_inner_result(failed_norm_rec) == ApproximateSolved)
check("... with reason :mean_m_resid_exceeds_tol", :mean_m_resid_exceeds_tol in verification_rejection_reasons(failed_norm_rec))

failed_kkt_rec = merge(base_ok, (max_abs_moment_kkt_resid = 5.0,))
check("failed moment/KKT residual is REJECTED", classify_inner_result(failed_kkt_rec) == ApproximateSolved)
check("... with reason :max_abs_moment_kkt_resid_exceeds_tol", :max_abs_moment_kkt_resid_exceeds_tol in verification_rejection_reasons(failed_kkt_rec))

not_feasible_rec = merge(base_ok, (inner_status = -300,))
check("genuine -300 (unbounded) is ConfirmedNumericalNegative, not cacheable", classify_inner_result(not_feasible_rec) == ConfirmedNumericalNegative)
check("... rejection reason for -300 is :inner_status_not_feasible", verification_rejection_reasons(not_feasible_rec) == [:inner_status_not_feasible])

println("\n== 5. Cross-check: none of the genuine-failure records are cacheable ==")
for rec in (nan_weight_rec, neg_weight_rec, nan_delta_rec, nan_mmin_rec, failed_gap_rec, failed_norm_rec, failed_kkt_rec, not_feasible_rec)
    check("rejected record is NOT cacheable", !is_cacheable_result(rec))
end

println("\n============================================================")
println("TOTAL: $n_pass passed, $n_fail failed")
n_fail == 0 || error("$n_fail check(s) failed")

