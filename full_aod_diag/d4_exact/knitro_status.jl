# ============================================================================
# KNITRO termination-status decoder (task §10).
#
# `decode_knitro_status` is a pure lookup table transcribed directly from the
# ACTUALLY-LOADED KNITRO 13.0.1 installation on this machine
# (/opt/shared_sw/knitro/13.0.1/include/knitro.h's KN_RC_* defines, cross-checked
# against the shipped HTML reference manual
# /opt/shared_sw/knitro/13.0.1/doc/html/3_referenceManual/returnCodes.html) --
# not the "14.2.0" the repo's own .knitro_env.sh points at, which KNITRO.jl's
# hardcoded deps.jl never actually loads (see docs/fullA_D20_production_consolidation_handoff.md
# §2). Also includes this repo's OWN exact-screen sentinel codes (<=-9000), which are
# NOT native KNITRO return codes -- they never reach KN_get_solution at all, since the
# exact screens short-circuit BEFORE the inner KNITRO dual solve is ever attempted
# (infeasibility_screen.jl / fast_range_screen.jl).
#
# `knitro_solve_diagnostics(kc)` queries the live per-solve counters/errors KNITRO.jl's
# low-level API exposes right after KN_solve. Two things this KNITRO version's C API
# does NOT expose via any getter (documented here rather than fabricated): a
# complementarity-error component separate from the bundled optimality error, and a
# final-step-norm / line-search-count getter -- both are only visible in the per-
# iteration outlev print stream, not queryable post-hoc from a finished kc. Restoration
# and MIP-node counts are likewise not applicable/exposed for this continuous NLP usage.
# ============================================================================

struct KnitroStatusInfo
    code::Int
    name::Symbol
    category::Symbol   # :optimal, :feasible_approx, :infeasible, :unbounded, :limit_feasible,
                        # :limit_infeasible, :error, :exact_screen_certificate, :unknown
    is_feasible_result::Bool   # true iff a primal-feasible point was returned alongside this code
    meaning::String
end

const _KN_RC_TABLE = Dict{Int,Tuple{Symbol,Symbol,Bool,String}}(
    0    => (:KN_RC_OPTIMAL_OR_SATISFACTORY, :optimal, true,
             "Locally optimal solution found; stopping criteria satisfied to full tolerance. " *
             "Globally optimal if the problem is convex."),
    -100 => (:KN_RC_NEAR_OPT, :feasible_approx, true,
             "Primal feasible; appears optimal but full dual-feasibility tolerance not reached " *
             "(stopping tests satisfied within a factor of 100). No further progress possible."),
    -101 => (:KN_RC_FEAS_XTOL, :feasible_approx, true,
             "Primal feasible; terminated because the relative change in the solution estimate " *
             "fell below xtol. May indicate degeneracy/ill-conditioning/bad scaling rather than a " *
             "real accuracy loss."),
    -102 => (:KN_RC_FEAS_NO_IMPROVE, :feasible_approx, true,
             "Primal feasible; the solution estimate cannot be improved further and the desired " *
             "dual-feasibility accuracy could not be achieved."),
    -103 => (:KN_RC_FEAS_FTOL, :feasible_approx, true,
             "Primal feasible; terminated because the relative objective change fell below ftol " *
             "for ftol_iters consecutive iterations."),
    -200 => (:KN_RC_INFEASIBLE, :infeasible, false,
             "Converged to an infeasible point; problem may be locally infeasible, or the point " *
             "is a bad-scaling/nonlinear-constraint artifact. Consider multistart."),
    -201 => (:KN_RC_INFEAS_XTOL, :infeasible, false,
             "Terminated at an infeasible point because the relative change in the solution " *
             "estimate fell below xtol."),
    -202 => (:KN_RC_INFEAS_NO_IMPROVE, :infeasible, false,
             "Current infeasible estimate cannot be improved further; problem may be badly scaled " *
             "or genuinely infeasible."),
    -203 => (:KN_RC_INFEAS_MULTISTART, :infeasible, false,
             "Multistart could not find any feasible point across all tried starts."),
    -204 => (:KN_RC_INFEAS_CON_BOUNDS, :infeasible, false,
             "The constraint bounds themselves were determined to be infeasible."),
    -205 => (:KN_RC_INFEAS_VAR_BOUNDS, :infeasible, false,
             "The variable bounds themselves were determined to be infeasible."),
    -300 => (:KN_RC_UNBOUNDED, :unbounded, false,
             "Problem appears unbounded: the current iterate is feasible and the objective " *
             "magnitude exceeds objrange while still decreasing. THIS is the code the production " *
             "driver's FEASIBLE_CODES excludes and treats as a genuine organic inner-solve failure " *
             "(see docs/fullA_driver_delta5_diagnostics_handoff.md §11) -- for the CC dual problem " *
             "here it means the inner dual objective diverged rather than that the outer economic " *
             "problem is truly unbounded."),
    -301 => (:KN_RC_UNBOUNDED_OR_INFEAS, :unbounded, false,
             "The Knitro presolver determined the problem is either unbounded or infeasible " *
             "(could not distinguish which)."),
    -400 => (:KN_RC_ITER_LIMIT_FEAS, :limit_feasible, true,
             "Iteration limit (maxit) reached before full convergence; a feasible point WAS found."),
    -401 => (:KN_RC_TIME_LIMIT_FEAS, :limit_feasible, true,
             "Time limit (maxtime_real/maxtime_cpu) reached before full convergence; a feasible " *
             "point WAS found. This is the code a per-stage staged-continuation budget cutoff " *
             "produces (§3/§6)."),
    -402 => (:KN_RC_FEVAL_LIMIT_FEAS, :limit_feasible, true,
             "Function-evaluation limit (maxfevals) reached; a feasible point WAS found."),
    -403 => (:KN_RC_MIP_EXH_FEAS, :limit_feasible, true, "MIP: all nodes explored, integer-feasible point found (not applicable to this repo's continuous NLP usage)."),
    -404 => (:KN_RC_MIP_TERM_FEAS, :limit_feasible, true, "MIP: terminated at first integer-feasible point (not applicable here)."),
    -405 => (:KN_RC_MIP_SOLVE_LIMIT_FEAS, :limit_feasible, true, "MIP subproblem-solve limit reached, integer-feasible point found (not applicable here)."),
    -406 => (:KN_RC_MIP_NODE_LIMIT_FEAS, :limit_feasible, true, "MIP node limit reached, integer-feasible point found (not applicable here)."),
    -410 => (:KN_RC_ITER_LIMIT_INFEAS, :limit_infeasible, false,
             "Iteration limit reached; NO feasible point was found."),
    -411 => (:KN_RC_TIME_LIMIT_INFEAS, :limit_infeasible, false,
             "Time limit reached; NO feasible point was found."),
    -412 => (:KN_RC_FEVAL_LIMIT_INFEAS, :limit_infeasible, false,
             "Function-evaluation limit reached; NO feasible point was found."),
    -413 => (:KN_RC_MIP_EXH_INFEAS, :limit_infeasible, false, "MIP: all nodes explored, no integer-feasible point found (not applicable here)."),
    -415 => (:KN_RC_MIP_SOLVE_LIMIT_INFEAS, :limit_infeasible, false, "MIP subproblem-solve limit reached, no feasible point (not applicable here)."),
    -416 => (:KN_RC_MIP_NODE_LIMIT_INFEAS, :limit_infeasible, false, "MIP node limit reached, no feasible point (not applicable here)."),
    -500 => (:KN_RC_CALLBACK_ERR, :error, false, "A callback function (cb_F!/cb_G!/cb_H!/cb_newpt!) raised an error or threw an exception KNITRO caught."),
    -501 => (:KN_RC_LP_SOLVER_ERR, :error, false, "Error in the LP subsolver."),
    -502 => (:KN_RC_EVAL_ERR, :error, false, "Evaluation error (e.g. NaN/Inf) reported by a callback, or an internal function evaluation error."),
    -503 => (:KN_RC_OUT_OF_MEMORY, :error, false, "KNITRO ran out of memory."),
    -504 => (:KN_RC_USER_TERMINATION, :error, false, "Terminated by a user callback (e.g. newpt callback returning nonzero)."),
    -505 => (:KN_RC_OPEN_FILE_ERR, :error, false, "Error opening an input/output file."),
    -506 => (:KN_RC_BAD_N_OR_F, :error, false, "Problem definition error: bad number of variables or objective."),
    -515 => (:KN_RC_ILLEGAL_CALL, :error, false, "A KNITRO API call was made out of the required sequence."),
    -518 => (:KN_RC_BAD_INIT_VALUE, :error, false, "The application-supplied initial point is invalid (e.g. NaN, outside bounds in a way KNITRO cannot reconcile)."),
    -520 => (:KN_RC_LICENSE_ERROR, :error, false, "License check failed (see docs/reference-knitro-license-demand.md memory note: this repo's license only activates on demand.mit.edu)."),
    -522 => (:KN_RC_LINEAR_SOLVER_ERR, :error, false, "Error in the internal linear solver (e.g. near-singular KKT system) -- a strong signal of Hessian/Jacobian ill-conditioning at the failing point."),
    -600 => (:KN_RC_INTERNAL_ERROR, :error, false, "Internal KNITRO error; Artelys support case."),
    # ---- This repo's own exact-screen sentinels (infeasibility_screen.jl / fast_range_screen.jl) --
    # NOT native KN_RC_* codes. These never reach KN_solve/KN_get_solution at all: the screen
    # short-circuits before an inner KNITRO context is even created for that evaluation. ----
    -9000 => (:LOCAL_EXACT_SCREEN_GENERIC, :exact_screen_certificate, false,
              "This repo's generic exact-infeasibility-screen sentinel (draw-independent certificate); no inner KNITRO solve was attempted."),
    -9001 => (:LOCAL_PAIRWISE_CERTIFIED_INFEASIBLE, :exact_screen_certificate, false,
              "infeasibility_screen.jl pairwise certificate: two destinations' implied winner sets are exactly, provably incompatible with any dual within the delta budget."),
    -9002 => (:LOCAL_WITNESS_CERTIFIED_INFEASIBLE, :exact_screen_certificate, false,
              "infeasibility_screen.jl witness certificate."),
    -9003 => (:LOCAL_WINNER_SCAN_INFEASIBLE, :exact_screen_certificate, false,
              "infeasibility_screen.jl winner-scan certificate."),
    -9004 => (:LOCAL_EXACT_INFEASIBLE_PREWINNER_ENVELOPE, :exact_screen_certificate, false,
              "fast_range_screen.jl pre-winner envelope certificate -- the cheapest/earliest of the three fast_range_screen.jl exact screens."),
    -9005 => (:LOCAL_EXACT_INFEASIBLE_WINNING_RANGE, :exact_screen_certificate, false,
              "fast_range_screen.jl fused winning-range certificate."),
    -9006 => (:LOCAL_EXACT_INFEASIBLE_MOMENT_RANGE, :exact_screen_certificate, false,
              "fast_range_screen.jl general moment-range/safety-net certificate."),
    -9999 => (:LOCAL_MICROBENCHMARK_ALL_OFFSETS_INFEASIBLE, :exact_screen_certificate, false,
              "Script-local sentinel used only by c9_w80k_microbenchmark.jl/c9_w800k_microbenchmark.jl -- not part of the production evaluation path."),
)

"""
    decode_knitro_status(code::Integer) -> KnitroStatusInfo

Pure lookup, no KNITRO call required. Unknown codes return `category=:unknown` with a
generic message rather than throwing, so instrumentation code can log a status it
doesn't recognize (e.g. a future KNITRO version adding new codes) without crashing.
"""
function decode_knitro_status(code::Integer)::KnitroStatusInfo
    c = Int(code)
    if haskey(_KN_RC_TABLE, c)
        name, category, is_feasible_result, meaning = _KN_RC_TABLE[c]
        return KnitroStatusInfo(c, name, category, is_feasible_result, meaning)
    end
    return KnitroStatusInfo(c, :UNKNOWN, :unknown, false,
        "Status code $c is not in this decoder's table (transcribed from the actually-loaded " *
        "KNITRO 13.0.1's knitro.h + returnCodes.html). Update _KN_RC_TABLE in knitro_status.jl.")
end

"""
    knitro_solve_diagnostics(kc) -> NamedTuple

Queries the live per-solve counters/errors from a `KNITRO.KN_context` immediately after
`KN_solve` returns (before `KN_free`). All fields are genuinely per-solve: KNITRO
resets these counters at the start of each `KN_solve` call on a given `kc` (verified by
`test_knitro_status.jl`'s "solve twice" regression test, task §9's own concern about
counters silently inheriting a prior solve's values).

Fields not populated (documented, not fabricated): this KNITRO C API version exposes no
separate complementarity-error getter (bundled into abs/rel_opt_error for this NLP
formulation) and no post-hoc final-step-norm or line-search/restoration-count getter --
those are only visible in the per-iteration `outlev` print stream, not queryable from a
finished `kc`.
"""
function knitro_solve_diagnostics(kc)
    iters = Ref{Cint}(0); cg_iters = Ref{Cint}(0)
    fc_evals = Ref{Cint}(0); ga_evals = Ref{Cint}(0); h_evals = Ref{Cint}(0); hv_evals = Ref{Cint}(0)
    t_real = Ref{Cdouble}(NaN); t_cpu = Ref{Cdouble}(NaN)
    abs_feas = Ref{Cdouble}(NaN); rel_feas = Ref{Cdouble}(NaN)
    abs_opt = Ref{Cdouble}(NaN); rel_opt = Ref{Cdouble}(NaN)
    KNITRO.KN_get_number_iters(kc, iters)
    KNITRO.KN_get_number_cg_iters(kc, cg_iters)
    KNITRO.KN_get_number_FC_evals(kc, fc_evals)
    KNITRO.KN_get_number_GA_evals(kc, ga_evals)
    KNITRO.KN_get_number_H_evals(kc, h_evals)
    KNITRO.KN_get_number_HV_evals(kc, hv_evals)
    KNITRO.KN_get_solve_time_real(kc, t_real)
    KNITRO.KN_get_solve_time_cpu(kc, t_cpu)
    KNITRO.KN_get_abs_feas_error(kc, abs_feas)
    KNITRO.KN_get_rel_feas_error(kc, rel_feas)
    KNITRO.KN_get_abs_opt_error(kc, abs_opt)
    KNITRO.KN_get_rel_opt_error(kc, rel_opt)
    return (n_iters = Int(iters[]), n_cg_iters = Int(cg_iters[]),
            n_fc_evals = Int(fc_evals[]), n_ga_evals = Int(ga_evals[]),
            n_h_evals = Int(h_evals[]), n_hv_evals = Int(hv_evals[]),
            solve_time_real = t_real[], solve_time_cpu = t_cpu[],
            abs_feas_error = abs_feas[], rel_feas_error = rel_feas[],
            abs_opt_error = abs_opt[], rel_opt_error = rel_opt[],
            complementarity_error = missing,   # not exposed by this KNITRO C API version, see docstring
            final_step_norm = missing)         # not exposed by this KNITRO C API version, see docstring
end

"""
    full_status_record(code::Integer, kc) -> NamedTuple

Convenience: `decode_knitro_status(code)` merged with `knitro_solve_diagnostics(kc)` into
one flat NamedTuple, suitable for direct JLD2/JSON serialization into a fixed-point
profile or an organic-failure archive (task §7, §10).
"""
function full_status_record(code::Integer, kc)
    info = decode_knitro_status(code)
    diag = knitro_solve_diagnostics(kc)
    return (status_code = info.code, status_name = info.name, status_category = info.category,
            is_feasible_result = info.is_feasible_result, status_meaning = info.meaning, diag...)
end
