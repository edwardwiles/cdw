# ============================================================================
# Continuation 10, Part 1: chunked-BLAS exact Hessian for the CC inner dual
# solve's Hessian callback.
#
# BASELINE (unchanged production default -- cc_algo/PsiObjectiveBundle.jl:482,
# `hessian!(h, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta})`,
# the method actually dispatched for `d20_real_setup`'s obj, per
# context_real_d20.jl wiring `CS.PsiObjectiveBundleImplicit`): EVERY Hessian-
# callback invocation copies the FULL W x outer_constr_index slice of
# `obj.H` into `obj.H_copy`, scales it in place by `sqrt(arg2)` (the per-draw
# CC weight), then forms the exact Newton Hessian via ONE BLAS
# `gemm!('T','N', H_copy, H_copy)` call -- i.e. Z'DZ with the whole W-row Z
# materialized in `H_copy` before the single matmul. `obj.H` itself (the raw
# moment matrix, built once per inner solve by `obj.moments!`) is untouched by
# this file -- Part 2 addresses building THAT matrix faster/differently; this
# file only replaces the Hessian callback's OWN scaled-copy-plus-gemm step.
#
# THIS FILE (additive only, does not touch PsiObjectiveBundle.jl or
# oracle_fast.jl): `hessian_chunked!` computes the IDENTICAL Z'DZ product, but
# processes draws in row-chunks of a configurable `chunk_size` -- only a
# `chunk_size x outer_constr_index` buffer is ever materialized (not the full
# `W x outer_constr_index` H_copy), and the Hessian is accumulated across
# chunks via repeated BLAS `gemm!` calls with beta=1 (each a partial
# `Z_c'D_cZ_c`). Mathematically EXACT, not an approximation: summing partial
# row-block Gram matrices reproduces the full Gram matrix exactly (up to
# floating-point summation-order noise, which a single large BLAS gemm's own
# internal cache-blocking already reorders anyway -- so even the baseline is
# not associativity-exact against a naive O(W*ncol^2) triple loop; chunking
# does not introduce a NEW source of reordering beyond what BLAS already
# does internally).
#
# `inner_loop_KNITRO_chunked`/`inner_loop_internal_chunked` mirror
# `oracle_fast.jl`'s `inner_loop_KNITRO_profiled`/`inner_loop_internal_profiled`
# line-for-line, swapping ONLY the Hessian callback registration (FG callback,
# variable/bound setup, complementarity wiring, solve-status handling are all
# reused UNCHANGED from oracle_fast.jl) -- so any timing/correctness diff
# between the chunked and baseline paths is attributable ONLY to the Hessian
# callback's own internals.
#
# See docs/fullA_D20_chunked_hessian_report.md for the benchmark writeup.
# ============================================================================

using LinearAlgebra: BLAS

"""
    hessian_chunked!(h, obj, chunk_size; zbuf=nothing)

Chunked-BLAS replacement for `cc_algo/PsiObjectiveBundle.jl`'s
`hessian!(h, obj::Union{PsiObjectiveBundleImplicit,PsiObjectiveBundleDelta})`.
Requires `obj.arg0` to already reflect the CURRENT (ζ,λ) -- the caller must
run the same `BLAS.gemv!('N', 1.0, H[:,2:1+outer_constr_index], -x, 0.0, arg0)`
step the production callable method runs unconditionally before dispatching
to `hessian!` (see `_prep_for_hessian!` below, used by
`inner_loop_KNITRO_chunked`'s callback closure). Writes the packed
upper-triangular Hessian into `h`, EXACTLY as `hessian!` does (identical
packing loop, copied verbatim from PsiObjectiveBundle.jl:492-498).
"""
function hessian_chunked!(h, obj, chunk_size::Int;
                           zbuf::Union{Nothing,Matrix{Float64}} = nothing)
    @unpack H, M, arg0, arg2, ddPsi!, outer_constr_index, ∂∂f_∂∂x = obj
    ddPsi!(arg2, arg0)

    W = size(H, 1)
    ncol = outer_constr_index
    cs = min(chunk_size, W)
    buf = zbuf === nothing ? Matrix{Float64}(undef, cs, ncol) : zbuf
    (size(buf, 1) >= cs && size(buf, 2) == ncol) ||
        error("hessian_chunked!: zbuf size $(size(buf)) incompatible with cs=$cs, ncol=$ncol")

    fill!(∂∂f_∂∂x, 0.0)
    start = 1
    @inbounds while start <= W
        stop = min(start + chunk_size - 1, W)
        n = stop - start + 1
        Zc = @view buf[1:n, :]
        @views Zc .= H[start:stop, 2:1+outer_constr_index]
        @views Zc .*= sqrt.(arg2[start:stop])
        BLAS.gemm!('T', 'N', 1 / M, Zc, Zc, 1.0, ∂∂f_∂∂x)
        start = stop + 1
    end

    k = 1
    for i in 1:size(∂∂f_∂∂x, 2)
        for j in i:size(∂∂f_∂∂x, 2)
            h[k] = ∂∂f_∂∂x[i, j]
            k += 1
        end
    end
    return h
end

"""
Replicates the `arg0`/`arg1` refresh the production callable method performs
unconditionally before it would dispatch to `hessian!` (see
`(Q::PsiObjectiveBundleImplicit)(x,...)`'s first three statements in
cc_algo/PsiObjectiveBundle.jl). Needed because the chunked Hessian callback
below calls `hessian_chunked!` directly instead of going through the full
callable method -- this keeps the chunked callback's PER-CALL cost
apples-to-apples with the baseline's `obj(x, h=evalResult.hess)` (same
`Psi!` call included, even though `hessian_chunked!` itself does not read
`arg1` -- reproduced here so neither path does strictly less work).
"""
function _prep_for_hessian!(obj, x)
    @unpack H, arg0, arg1, outer_constr_index, Psi! = obj
    BLAS.gemv!('N', 1.0, @view(H[:, 2:1+outer_constr_index]), -x, 0.0, arg0)
    Psi!(arg1, arg0)
    return nothing
end

mutable struct ChunkedHessCounters
    n_calls::Int
end
const _CHUNKED_HESS_COUNTERS = Ref(ChunkedHessCounters(0))

"""
    inner_loop_KNITRO_chunked(obj, chunk_size) -> (nStatus, objSol, x, lambda_, n_fg_calls, n_hess_calls)

Faithful mirror of `oracle_fast.jl::inner_loop_KNITRO_profiled`, with ONLY the
Hessian callback swapped for `hessian_chunked!` above (via a closure capturing
`chunk_size`/a reusable `zbuf`, since KNITRO ties a single `userParams` value
per callback handle, already used to pass `obj` to the FG callback -- see
`KNITRO.KN_set_cb_user_params(kc, cb, obj)` below, unchanged from the
baseline). FG callback (`_callbackEvalFG_inner_profiled!`, defined in
`oracle_fast.jl`, included by this file's callers before this file) is reused
UNCHANGED.
"""
function inner_loop_KNITRO_chunked(obj, chunk_size::Int)
    _INNER_CALL_COUNTERS[] = InnerCallCounters(0, 0)
    _CHUNKED_HESS_COUNTERS[] = ChunkedHessCounters(0)

    kc = KNITRO.KN_new()
    KNITRO.KN_add_vars(kc, CS.inner_loop_number_variables(obj))
    KNITRO.KN_set_var_lobnds_all(kc, CS.inner_loop_lower_bounds(obj))
    KNITRO.KN_set_var_primal_init_values_all(kc, CS.inner_loop_initial_values(obj))

    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], _callbackEvalFG_inner_profiled!)
    KNITRO.KN_set_cb_user_params(kc, cb, obj)
    KNITRO.KN_load_param_file(kc, obj.inner_loop_opt)

    zbuf = Matrix{Float64}(undef, min(chunk_size, size(obj.H, 1)), obj.outer_constr_index)

    if KNITRO.KN_get_int_param(kc, "hessopt") == 1
        hess_cb = (kc2, cb2, evalRequest, evalResult, userParams) -> begin
            o = userParams   # == obj, the SAME shared userParams the FG callback receives
            xloc = evalRequest.x
            @prof "inner_dual_hessian_callback_chunked" begin
                _prep_for_hessian!(o, xloc)
                hessian_chunked!(evalResult.hess, o, chunk_size; zbuf = zbuf)
            end
            _CHUNKED_HESS_COUNTERS[].n_calls += 1
            return 0
        end
        KNITRO.KN_set_cb_hess(kc, cb, KNITRO.KN_DENSE_ROWMAJOR, hess_cb)
    end
    if obj.complement_index != [0 0]
        CS.inner_loop_complementarity_constraints(kc, obj)
    end

    @prof "inner_knitro_dual_solve_chunked" begin
        KNITRO.KN_solve(kc)
    end
    nSTatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
    CS.INNER_ITERS_TOTAL[] += CS._kn_num_iters(kc)
    KNITRO.KN_free(kc)

    return nSTatus, objSol, x, lambda_, _INNER_CALL_COUNTERS[].n_fg_calls, _CHUNKED_HESS_COUNTERS[].n_calls
end

"""
    inner_loop_internal_chunked(obj, θ, chunk_size) -> (K_hard, x, nStatus, n_fg_calls, n_hess_calls)

Faithful mirror of `oracle_fast.jl::inner_loop_internal_profiled`, dispatching
to `inner_loop_KNITRO_chunked` instead of `inner_loop_KNITRO_profiled`. The
ONE `obj.moments!` call (`inner_moment_build`) is UNCHANGED (dense, as always
-- Part 2 addresses that build itself).
"""
function inner_loop_internal_chunked(obj, θ, chunk_size::Int)
    @prof "inner_moment_build" begin
        obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ, obj.U, obj)
    end
    obj.H[:, 2] .= 1.0
    obj.H_save = obj.H[1, 1] * (-1.0)^obj.find_smallest

    nStatus, objSol, x, lambda_, n_fg, n_hess = inner_loop_KNITRO_chunked(obj, chunk_size)

    CS.INNER_SOLVE_COUNT[] += 1
    if nStatus ∉ [0, -100, -101, -103]
        CS.INNER_INFEAS_COUNT[] += 1
    end

    if nStatus ∈ [0, -100, -101, -103]
        obj.x .= x
        return obj.H_save, x, nStatus, n_fg, n_hess
    else
        obj.x .= NaN
        return -1e10, x, nStatus, n_fg, n_hess
    end
end
