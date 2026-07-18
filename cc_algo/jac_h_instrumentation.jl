# ============================================================================
# Additive, purely observational instrumentation for the jac_h audit
# (diag/fullA-d4-exact-jach-audit, 2026-07-18 -- see docs/fullA_jach_audit.md).
#
# Every counter/timer here is a passive Ref side-effect wrapped around code
# that is otherwise UNCHANGED -- no return value, array contents, or control
# flow of any wrapped computation is altered by this file. This mirrors the
# existing INNER_SOLVE_COUNT/INNER_INFEAS_COUNT/INNER_ITERS_TOTAL pattern in
# cc_algo/inner_loop_functions.jl (reset by the caller before a scoped
# measurement, read after). Included BEFORE PsiObjectiveBundle.jl in
# cc_algo/include_cc_algo.jl so `_instrumented_jac_h_default`/`_skipped_jac_h_default`
# are in scope for that file's `@with_kw` field defaults.
# ============================================================================

const JAC_H_ALLOC_COUNT        = Ref(0)     # # times a REAL (nonempty) jac_h array was constructed
const JAC_H_ALLOC_BYTES        = Ref(0)     # cumulative bytes allocated for jac_h fields
const JAC_H_ALLOC_TIME         = Ref(0.0)   # cumulative wall time (s) spent allocating+zeroing jac_h
const JAC_H_SKIPPED_COUNT      = Ref(0)     # # times jac_h construction was SKIPPED (needs_outer_moment_jacobian=false)
const JAC_H_POPULATE_COUNT     = Ref(0)     # # calls to calculate_jac_θ! (the function that FILLS jac_h)
const JAC_H_POPULATE_TIME      = Ref(0.0)   # cumulative wall time (s) inside calculate_jac_θ!
const JAC_H_THETA_BRANCH_COUNT = Ref(0)     # # times a PsiObjectiveBundle callable's length(theta)>0 branch (the
                                             # legacy outer-gradient branch that reads/contracts jac_h) is entered
const JAC_H_IFT_COUNT          = Ref(0)     # # calls to ift! (Implicit/Delta variant) -- the other jac_h reader/contractor

"""
    _instrumented_jac_h_default(N, d, l) -> Array{Float64,3}

Drop-in replacement for the bare `zeros(N, d + 2, l)` default previously used for the `jac_h`
field default in every `PsiObjectiveBundle*` struct -- returns the IDENTICAL array (same
size/type/all-zero contents), only additionally counted and timed so this investigation's claims
about jac_h's allocation frequency/cost are measured directly, not assumed from static reading.
"""
function _instrumented_jac_h_default(N::Int, d::Int, l::Int)
    JAC_H_ALLOC_COUNT[] += 1
    t0 = time()
    arr = zeros(N, d + 2, l)
    JAC_H_ALLOC_TIME[] += time() - t0
    JAC_H_ALLOC_BYTES[] += sizeof(arr)
    return arr
end

"""
    _skipped_jac_h_default() -> Array{Float64,3}

The `needs_outer_moment_jacobian=false` branch of the `jac_h` default: a 0x0x0 array (type-stable,
zero allocation of consequence) instead of the full N x (d+2) x l tensor. Counted separately so a
run can report exactly how many objects were constructed in no-jac_h mode.
"""
function _skipped_jac_h_default()
    JAC_H_SKIPPED_COUNT[] += 1
    return zeros(0, 0, 0)
end

"""
    reset_jac_h_counters!()

Zero every counter/timer above -- call before a scoped measurement (mirrors
`INNER_SOLVE_COUNT[] = 0` in cc_algo/outer_loop_functions.jl::outer_loop).
"""
function reset_jac_h_counters!()
    JAC_H_ALLOC_COUNT[] = 0; JAC_H_ALLOC_BYTES[] = 0; JAC_H_ALLOC_TIME[] = 0.0
    JAC_H_SKIPPED_COUNT[] = 0
    JAC_H_POPULATE_COUNT[] = 0; JAC_H_POPULATE_TIME[] = 0.0
    JAC_H_THETA_BRANCH_COUNT[] = 0; JAC_H_IFT_COUNT[] = 0
end

"NamedTuple snapshot of every counter -- convenient for logging/CSV rows."
jac_h_counters_snapshot() = (alloc_count = JAC_H_ALLOC_COUNT[], alloc_bytes = JAC_H_ALLOC_BYTES[],
    alloc_time = JAC_H_ALLOC_TIME[], skipped_count = JAC_H_SKIPPED_COUNT[],
    populate_count = JAC_H_POPULATE_COUNT[], populate_time = JAC_H_POPULATE_TIME[],
    theta_branch_count = JAC_H_THETA_BRANCH_COUNT[], ift_count = JAC_H_IFT_COUNT[])
