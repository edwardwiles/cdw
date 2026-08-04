# quarantine.jl -- structured false-negative quarantine records for the post-verifier-fix W=100k
# rerun + K=3 campaign (task §2.5, POST_VERIFY_FIX namespace).
#
# Written whenever a point passes the OUTER feasibility test (Delta_dual <= delta budget) but
# `classify_inner_result` rejects it (ApproximateSolved/ConfirmedNumericalNegative) -- exactly the
# "feasible=true verified=false" pattern that hid the m_min underflow bug for weeks. Never silently
# logs one line and discards the point: every such event gets its own immutable, content-addressed
# record with the exact outer vector + digest, the inner dual + digest, every verification metric,
# and the itemized failed predicates (verification_rejection_reasons, oracle.jl).
#
# Requires oracle.jl (classify_inner_result/verification_rejection_reasons/is_verified_success)
# already included by the caller.

using Dates, SHA, Serialization

const QUARANTINE_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/quarantine"

"Set by the campaign launcher (Phase 4/5) to the active manifest hash for the family/direction/delta being run; quarantine records are stamped with whatever this holds at write time."
const ACTIVE_CAMPAIGN_MANIFEST_HASH = Ref{String}("unset")

"Per-(label,delta) consecutive-same-reason counter (§2.5: 3+ consecutive same-reason rejections must emit a prominent warning, not just accumulate silently)."
mutable struct QuarantineStreak
    reasons::Vector{Symbol}
    count::Int
end
const QUARANTINE_STREAKS = Dict{Tuple{String,Float64}, QuarantineStreak}()

"""
    quarantine_feasible_unverified!(label, delta, w, verify; checkpoint_source="") -> String

Write one structured quarantine record for a point that passed outer feasibility (Delta<=delta)
but failed `classify_inner_result` (feasible=true/verified=false). Idempotent/content-addressed
(sha256 of the outer vector `w`) -- never overwrites an existing record. `label` is the driver's
own run label (already encodes family_direction_stage, e.g. "flexible_cm_lower_POLISH_SQP" --
matches this codebase's own existing checkpoint-naming convention, see cm_checkpoint.jl). Returns
the written (or pre-existing) path.
"""
function quarantine_feasible_unverified!(label::String, delta::Float64, w::AbstractVector{Float64},
        verify; checkpoint_source::AbstractString = "")
    reasons = verification_rejection_reasons(verify)
    w_digest = bytes2hex(sha256(join(string.(w), ",")))
    lambda = get(verify, :lambda, Float64[])
    lambda_digest = isempty(lambda) ? "" : bytes2hex(sha256(join(string.(lambda), ",")))
    manifest_hash = ACTIVE_CAMPAIGN_MANIFEST_HASH[]

    dest_dir = joinpath(QUARANTINE_ROOT, manifest_hash, label, "delta_$(delta)")
    isdir(dest_dir) || mkpath(dest_dir)
    dest_path = joinpath(dest_dir, "$(w_digest[1:16]).jls")

    if !isfile(dest_path)
        record = (
            label = label, delta = delta, manifest_hash = manifest_hash,
            w = collect(w), w_digest = w_digest,
            lambda = collect(lambda), lambda_digest = lambda_digest,
            inner_status = get(verify, :inner_status, missing),
            Delta_dual = get(verify, :Delta_dual, missing), Delta_primal = get(verify, :Delta_primal, missing),
            primal_dual_gap = get(verify, :primal_dual_gap, missing),
            mean_m_resid = get(verify, :mean_m_resid, missing),
            max_abs_moment_kkt_resid = get(verify, :max_abs_moment_kkt_resid, missing),
            m_min = get(verify, :m_min, missing), m_max = get(verify, :m_max, missing),
            m_weights_all_finite = get(verify, :m_weights_all_finite, missing),
            m_weights_all_nonnegative = get(verify, :m_weights_all_nonnegative, missing),
            underflow_zero_count = get(verify, :underflow_zero_count, missing),
            r_min = get(verify, :r_min, missing), r_max = get(verify, :r_max, missing),
            failed_predicates = reasons,
            checkpoint_source = checkpoint_source,
            recorded_at = string(Dates.now()),
        )
        serialize(dest_path, record)
        chmod(dest_path, 0o444)
    end

    key = (label, delta)
    streak = get!(() -> QuarantineStreak(Symbol[], 0), QUARANTINE_STREAKS, key)
    if streak.reasons == reasons
        streak.count += 1
    else
        streak.reasons = reasons
        streak.count = 1
    end
    if streak.count >= 3
        println("  [QUARANTINE WARNING] ", label, " delta=", delta, ": ", streak.count,
                " CONSECUTIVE feasible=true/verified=false rejections with the SAME reason(s) ",
                reasons, " -- a pattern, not noise. All candidates preserved under ", dest_dir, ".")
        flush(stdout)
    end
    return dest_path
end

"Reset a (label,delta) cell's consecutive-failure streak counter -- call whenever a point IS accepted/verified, so a later isolated rejection doesn't inherit a stale streak count."
reset_quarantine_streak!(label::String, delta::Float64) = delete!(QUARANTINE_STREAKS, (label, delta))
