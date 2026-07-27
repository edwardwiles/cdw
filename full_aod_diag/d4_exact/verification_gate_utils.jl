# ================================================================================================
# verification-defaults task (2026-07-27), Section 6.1 comparison harness. Shared by every
# test_verification_backend_default_*.jl gate script (one per family). Requires oracle.jl
# (classify_inner_result/is_cacheable_result/is_verified_success/InnerResultClass) and
# operator_verification.jl to already be included by the caller.
#
# `compare_verify_tuples` runs every Section 6.1 comparison that is genuinely a function of the
# `verify` NamedTuple alone (objective via Delta_dual, KKT residual, feasibility/moment residual
# via mean_m_resid, status classification, cache admission decision, incumbent admission
# decision). Draw-level dual index (m_star) and complete dual gradient (full g_lambda vector) are
# family-specific (need cf/op/layout/bins to call the operator verifier standalone) and are checked
# directly in each family's own gate script, not here.
# ================================================================================================

Base.@kwdef mutable struct VerifyGateTally
    n_checks::Int = 0
    n_pass::Int = 0
end
const GATE_TALLY = VerifyGateTally()

function gcheck(name::AbstractString, cond::Bool)
    GATE_TALLY.n_checks += 1
    cond && (GATE_TALLY.n_pass += 1)
    println(cond ? "PASS  " : "FAIL  ", name)
    return cond
end

"""
    compare_verify_tuples(label, verify_dense, verify_operator; kkt_tol=1e-6, obj_tol=1e-8) -> Bool

Runs the NamedTuple-level Section 6.1 comparisons (objective, KKT residual, feasibility/moment
residual, status classification, cache admission decision, incumbent admission decision) between a
dense-backend `verify` NamedTuple and an operator-backend one (SAME shape, both consumable
identically by `classify_inner_result`/`is_cacheable_result`/`is_verified_success`, oracle.jl).
Returns true iff every check passed.
"""
function compare_verify_tuples(label::AbstractString, verify_dense, verify_operator;
        kkt_tol::Float64 = 1e-6, obj_tol::Float64 = 1e-8)
    ok = true
    ok &= gcheck("$label: inner_status agrees (dense=$(verify_dense.inner_status), operator=$(verify_operator.inner_status))",
                 verify_dense.inner_status == verify_operator.inner_status)
    dobj = abs(verify_dense.Delta_dual - verify_operator.Delta_dual)
    ok &= gcheck("$label: objective (Delta_dual) agrees (dense=$(verify_dense.Delta_dual), operator=$(verify_operator.Delta_dual), |Δ|=$dobj)",
                 dobj < obj_tol)
    dkkt = abs(verify_dense.max_abs_moment_kkt_resid - verify_operator.max_abs_moment_kkt_resid)
    ok &= gcheck("$label: KKT residual agrees (dense=$(verify_dense.max_abs_moment_kkt_resid), operator=$(verify_operator.max_abs_moment_kkt_resid), |Δ|=$dkkt)",
                 dkkt < kkt_tol)
    dmmr = abs(verify_dense.mean_m_resid - verify_operator.mean_m_resid)
    ok &= gcheck("$label: feasibility/moment residual (mean_m_resid) agrees (dense=$(verify_dense.mean_m_resid), operator=$(verify_operator.mean_m_resid), |Δ|=$dmmr)",
                 dmmr < 1e-8)
    dgap = abs(verify_dense.primal_dual_gap - verify_operator.primal_dual_gap)
    ok &= gcheck("$label: primal_dual_gap agrees (dense=$(verify_dense.primal_dual_gap), operator=$(verify_operator.primal_dual_gap), |Δ|=$dgap)",
                 dgap < kkt_tol)

    class_dense = classify_inner_result(verify_dense)
    class_operator = classify_inner_result(verify_operator)
    ok &= gcheck("$label: status classification agrees (dense=$class_dense, operator=$class_operator)",
                 class_dense == class_operator)

    cache_dense = is_cacheable_result(verify_dense)
    cache_operator = is_cacheable_result(verify_operator)
    ok &= gcheck("$label: cache admission decision agrees (dense=$cache_dense, operator=$cache_operator)",
                 cache_dense == cache_operator)

    incumbent_dense = is_verified_success(verify_dense)
    incumbent_operator = is_verified_success(verify_operator)
    ok &= gcheck("$label: incumbent admission decision agrees (dense=$incumbent_dense, operator=$incumbent_operator)",
                 incumbent_dense == incumbent_operator)

    return ok
end
