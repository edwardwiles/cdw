# ============================================================================
# Allocation/Hessian port task, section 6.3: BLAS thread policy for the inner solve.
#
# Prior evidence (diag/fullA-inner-blas-threading's fullA_inner_blas_threading_report.md,
# 2026-07-21, `production/fullA-exact` merge-base 191 commits behind this branch's tip -- i.e.
# measured on a substantially older codebase, before the canonical winner engine/compressed
# screening path/CM+meanZC/C+ backend existed in their current form. Treated here as DIRECTIONAL
# prior evidence only, reconfirmed against current code -- see
# UNRESTRICTED_BLAS_HESSIAN_BENCHMARK_2026-07-25.md for the current-code numbers) found the
# unrestricted family's dense O(W*n^2) Hessian/gemm work scales well with BLAS threads (noCM_cold:
# 6.88s->3.73s, Julia=1, BLAS 1->20; diminishing returns past ~8 threads), while the CM path's
# gradient coordinate loop is flat-to-marginally-worse with more BLAS threads (already parallel
# over Julia threads instead). Because only one inner KNITRO solve is ever active per process
# (this codebase's own guard_enter_inner_solve!/guard_exit_inner_solve! discipline), it is safe to
# set BLAS threads ONCE per outer-solve process rather than per individual matrix multiplication.
# ============================================================================
using LinearAlgebra: BLAS

"""
    with_blas_threads(f, n::Union{Nothing,Int})

Runs `f()` with `BLAS.set_num_threads(n)` in effect (if `n !== nothing`), restoring the PRIOR
BLAS thread count in a `finally` block regardless of how `f` returns (normally or via exception).
`n === nothing` is a true no-op: `f()` runs with whatever BLAS thread count was already in effect
(the ambient process default, e.g. from `OPENBLAS_NUM_THREADS`) -- zero behavior change for any
caller that does not explicitly request a BLAS thread override.
"""
function with_blas_threads(f, n::Union{Nothing,Int})
    n === nothing && return f()
    prior = BLAS.get_num_threads()
    BLAS.set_num_threads(n)
    try
        return f()
    finally
        BLAS.set_num_threads(prior)
    end
end
