# ================================================================================================
# D=20 profiling task (flexible_cm / common_frechet), 2026-07-28: opt-in, zero-overhead-when-off
# sub-block timing instrumentation for the SHARED production Hessian-callback code path
# (`hessian_cm_structured_v2!`/`hessian_cm_structured!`, cm_hessian_architectures.jl/
# cm_hessian_threaded.jl; `_fill_frechet_level_blocks!`, cm_frechet_hessian.jl;
# `_prep_dual_index_for_archC!`, cm_hessian_architectures.jl). Mirrors this codebase's own
# established pattern for exactly this kind of instrumentation (`no_dense_g_counters.jl`'s
# `NO_DENSE_G_COUNTERS`/`FAIL_FAST_ON_DENSE_G` -- a dedicated `Ref{Bool}`, default `false`, checked
# once per call site) -- NOT the same switch as this file's `PROF_ENABLED[]`/`@prof`
# (instrumentation.jl), which already defaults `true` and covers coarser labels
# ("inner_dual_hessian_callback_archC(_frechet)", "inner_knitro_dual_solve_arch") already wired at
# the KNITRO callback boundary; this file adds the INSIDE-the-callback sub-block breakdown those
# coarse labels don't provide, reusing the SAME `prof_record!`/`PROF_TIMES`/`prof_summary()`/
# `write_csv_rows` machinery (instrumentation.jl, included earlier in every caller's include list)
# so a caller sees both sets of labels in one `prof_summary()` call after a run.
#
# Also provides an opt-in "capture the dual-solve point `x` every Hessian callback" buffer
# (`CM_HESSIAN_CAPTURED_X`) and a "stash the live production context the moment it's built inside
# `run_cm_upper_checkpointed`" slot (`CM_LIVE_PCX_STASH`) -- together these let a profiling/gate
# script reach a REAL warmed `(cctx, obj)` state and REAL solver-visited dual points without ever
# calling a low-level KNITRO-invoking entry point directly (the confirmed KN_RC_CALLBACK_ERR trap
# for flexible_cm/common_frechet at real D=20, see STRUCTURED_CROSS_HESSIAN_MASTER_REPORT_2026-07-28.md
# §5) -- the ONLY KNITRO entry point exercised is the known-working public
# `run_cm_upper_checkpointed`; everything downstream of this file's hooks is plain Julia function
# calls on already-warmed, already-live state (no KN_new/KN_solve), reusing the SAME
# `hessian_cm_structured_v2!`/`_prep_dual_index_for_archC!` functions the real KNITRO callback
# itself calls -- not a second, potentially-drifting reimplementation.
# ================================================================================================

isdefined(Main, :PROF_ENABLED) || error("cm_hessian_subblock_profiling.jl requires instrumentation.jl (prof_record!/PROF_TIMES) to be included first")

"Opt-in master switch for this file's sub-block timing labels, x-point capture, and live-pcx stash. `false` (default): every guarded call site below is a single Ref check, no timing/allocation, no capture -- zero measurable overhead, matching this project's `no_dense_g_counters.jl` discipline for this exact kind of instrumentation."
const CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED = Ref{Bool}(false)

"Live handle to the most recently built production context (`pcx`, from `build_cm_production_context`/`build_cm_frechet_production_context`/`build_cm_meanzc_production_context`), stashed by `run_cm_upper_checkpointed` (cm_checkpoint.jl) the instant it is constructed -- ONLY while `CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]` is true. `pcx.cctx`/`pcx.ctx_cm.obj` give a profiling script a live handle into the SAME mutable state every real Hessian/FG callback for that run reads/writes, without a second direct low-level call. `nothing` until the first stash."
const CM_LIVE_PCX_STASH = Ref{Any}(nothing)

"Every dual-solve point `x` (copied, not aliased) passed to `_prep_dual_index_for_archC!` while `CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]` is true -- i.e. one entry per REAL Hessian callback KNITRO actually issued (across every outer eval, so naturally spans the calibration outer point through whatever later outer points the run's own outer KNITRO problem visits). Capped at `CM_HESSIAN_CAPTURE_MAX[]` entries (default generous, real ~90-120s runs fire far fewer Hessian callbacks than this) to bound memory on an accidental very-long run."
const CM_HESSIAN_CAPTURED_X = Vector{Vector{Float64}}()
const CM_HESSIAN_CAPTURE_MAX = Ref{Int}(5000)

"Clear the captured-x buffer (call before each fresh measurement run wanting a clean capture set)."
reset_cm_hessian_capture!() = (empty!(CM_HESSIAN_CAPTURED_X); nothing)

"Record one captured dual-solve point, subject to `CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]` and the cap. Not exported as public API -- called from `_prep_dual_index_for_archC!` only."
function _record_cm_hessian_capture_x!(x::AbstractVector{Float64})
    CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[] || return nothing
    length(CM_HESSIAN_CAPTURED_X) >= CM_HESSIAN_CAPTURE_MAX[] && return nothing
    push!(CM_HESSIAN_CAPTURED_X, copy(x))
    return nothing
end

"""
    @cmhess_prof "label" expr

Times `expr` (wall time only -- no allocation/GC tracking, unlike `@prof`, to keep this file's own
overhead minimal for what are, at production scale, sub-millisecond-to-low-millisecond sub-block
calls fired dozens of times per real inner solve) and records it under `"label"` into the SAME
`PROF_TIMES`/`prof_summary()` store `@prof` uses (instrumentation.jl), but gated by
`CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]` instead of `PROF_ENABLED[]` -- a single Ref check and a
direct fallthrough to `expr` when disabled (no try/finally, no `time_ns()` call at all), matching
the "zero overhead when off" requirement. Safe to call with `PROF_ENABLED[]` independently on or
off -- the two switches do not interact; a caller wanting exactly the fine-grained sub-block labels
without the pre-existing coarse ones can leave `PROF_ENABLED[]` at its own default.
"""
macro cmhess_prof(label, expr)
    quote
        if CM_HESSIAN_SUBBLOCK_PROFILING_ENABLED[]
            local _t0 = time_ns()
            local _r = $(esc(expr))
            local _t1 = time_ns()
            prof_record!($(esc(label)), (_t1 - _t0) / 1e9, 0, 0.0)
            _r
        else
            $(esc(expr))
        end
    end
end
