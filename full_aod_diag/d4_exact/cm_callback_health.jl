# ================================================================================================
# Callback-health / fake-success guard (cm-meanzc-frechet-outer-production-closeout-2026-08-06,
# task section 3).
#
# BACKGROUND: KNITRO.jl's own shared `_try_catch_handler` (C_wrapper.jl) wraps EVERY callback type
# (eval_fc, eval_ga, eval_hess, and the unrelated puts/output-text callback) in one try/catch. On a
# real Julia exception inside any of them (e.g. the 2026-08-06 cm_meanzc DimensionMismatch, see
# `cm_meanzc_lookup_production.jl`'s own 895b99b fix comment), it prints a generic, misleading
# "exception in puts callback" warning REGARDLESS of which callback actually threw, then returns
# KN_RC_CALLBACK_ERR to KNITRO rather than propagating the real Julia exception. KNITRO can then
# report nStatus=0 ("successful") with the inner solve never having actually run -- the untouched
# initial point, n_fg_calls=0 -- which reads exactly like a converged solve, not a crash. This file
# makes that failure mode impossible to mistake for a real result, for every family that shares the
# `inner_loop_KNITRO_*` KNITRO-wiring pattern (cm_lookup_production.jl/cm_meanzc_lookup_production.jl/
# cm_originzc_lookup_production.jl/cm_frechet_lookup_production.jl).
#
# DESIGN: `callback_health_guard` wraps a raw KNITRO callback in a try/catch that runs BEFORE
# KNITRO.jl's own C-boundary handler ever sees the exception -- it records the real Julia
# exception into a `CallbackHealthRecord`, then rethrows unchanged (KNITRO's own error-handling
# behavior, including its eventual nStatus, is not altered in any way -- only made diagnosable).
# `assert_no_fake_success!` is then called immediately after `KN_get_solution`, before any caller
# trusts nStatus/x/lambda_ for anything.
# ================================================================================================

mutable struct CallbackHealthRecord
    exception_seen::Bool
    exception_type::Any
    exception_message::String
    exception_backtrace_digest::UInt64
end
CallbackHealthRecord() = CallbackHealthRecord(false, Nothing, "", UInt64(0))

"Reset a `CallbackHealthRecord` to its clean state. Call at the start of every inner solve (each `inner_loop_KNITRO_*` call constructs a fresh record, so this is mostly for explicit reuse)."
function reset_callback_health!(h::CallbackHealthRecord)
    h.exception_seen = false
    h.exception_type = Nothing
    h.exception_message = ""
    h.exception_backtrace_digest = UInt64(0)
    return h
end

function record_callback_exception!(h::CallbackHealthRecord, e)
    h.exception_seen = true
    h.exception_type = typeof(e)
    h.exception_message = sprint(showerror, e)
    h.exception_backtrace_digest = hash(Base.catch_backtrace())
    return h
end

"""
    callback_health_guard(raw_cb, health::CallbackHealthRecord) -> Function

Wraps a raw KNITRO FG/Hessian callback (the `(kc, cb, evalRequest, evalResult, userParams) -> Int`
functions registered via `KN_add_eval_callback`/`KN_set_cb_hess`) so any Julia exception it throws
is recorded into `health` BEFORE it reaches KNITRO.jl's own callback-error swallowing. Rethrows
unchanged -- the KN_RC_CALLBACK_ERR KNITRO itself sees, and therefore every existing nStatus
code path, is completely unaffected; only the ability to tell a real crash from real infeasibility
is added.
"""
callback_health_guard(raw_cb, health::CallbackHealthRecord) =
    (kc, cb, evalRequest, evalResult, userParams) -> begin
        try
            return raw_cb(kc, cb, evalRequest, evalResult, userParams)
        catch e
            record_callback_exception!(health, e)
            rethrow()
        end
    end

"""
    assert_no_fake_success!(label, health, nStatus, n_fg, x_initial, x_solution) -> Nothing

Hard-rejects (throws a real `ErrorException`, NOT `CMExpectedSolveFailure` -- this is a
programming-bug signal, never a legitimate infeasibility outcome that a caller should silently
catch and treat as "try a different point") an inner solve result that cannot be trusted:

1. `health.exception_seen` -- a real Julia exception was thrown inside an FG/Hessian callback and
   masked by KNITRO.jl's own swallowing (see this file's header). This is the exact failure mode
   that let the 2026-08-06 missing-`Pow=` bug report `nStatus=0`/"success" for 200+ evaluations
   with the inner solve never having run at all.
2. `n_fg == 0` while `nStatus` is one of KNITRO's converged/near-converged codes -- the callback
   was never even invoked once, so nothing was actually solved.
3. `x_solution == x_initial` (element-wise) while `nStatus` claims a converged code -- the dual is
   still the untouched KNITRO starting point despite a claimed successful solve.

Do not call this from a context that is allowed to treat ordinary KNITRO infeasibility/timeout
codes (e.g. -300, -400, -401) as expected -- those are real, meaningful outcomes this function does
NOT reject; it only rejects results that masquerade as a genuine solve when no solve occurred.
"""
function assert_no_fake_success!(label::AbstractString, health::CallbackHealthRecord, nStatus::Integer,
                                  n_fg::Integer, x_initial::AbstractVector, x_solution::AbstractVector)
    if health.exception_seen
        error("$label: inner KNITRO callback threw a real Julia exception that KNITRO.jl's own " *
              "callback-error swallowing would otherwise have masked as nStatus=$nStatus -- " *
              "$(health.exception_type): $(health.exception_message)")
    end
    # n_fg==0 is rejected UNCONDITIONALLY, regardless of nStatus -- including KNITRO's genuine
    # infeasibility codes (e.g. -300, feedback-knitro-300-confirmed-infeasible-not-unbounded): a
    # real infeasibility certificate always requires at least one real FG evaluation to derive, so
    # n_fg==0 can never be a legitimate infeasibility result, only the fake-success signature
    # (KNITRO reporting SOME status without ever having called back into the objective).
    if n_fg == 0
        error("$label: n_fg_calls==0 (nStatus=$nStatus) -- the inner solve's FG callback was never " *
              "invoked, so nothing was actually solved; this is the fake-success signature (KNITRO " *
              "reporting a status without ever calling back into the objective), not a real result.")
    end
    # 2026-08-06 correction: `x_initial` (CS.inner_loop_initial_values(obj)) legitimately WARM-
    # STARTS from the previous solve's converged obj.x (dual_bank.jl/cm_dual_bank_production.jl's
    # own documented reuse contract) -- it is NOT always a fixed cold zero vector. A warm-started
    # dual that happens to already BE the optimum at a nearby new point legitimately converges in
    # zero real steps with x_solution==x_initial and n_fg>0 -- confirmed live 2026-08-06 at a real
    # common-Frechet D20/W=20,000 point (this exact false positive fired and was traced to a
    # genuine warm-started zero-step convergence, not a masked exception). The real fingerprint of
    # the original missing-Pow= bug this check exists to catch is specifically a COLD (all-zero)
    # start that never moved despite a claimed converged status -- restricting to that case keeps
    # the check's detection power for the actual failure mode without flagging legitimate warm
    # starts.
    if nStatus in (0, -100, -101, -102, -103) && length(x_initial) == length(x_solution) &&
       x_solution == x_initial && all(iszero, x_initial)
        error("$label: nStatus=$nStatus claims a converged solve, but the returned dual is still the " *
              "untouched COLD (all-zero) initial vector -- fake success.")
    end
    return nothing
end
