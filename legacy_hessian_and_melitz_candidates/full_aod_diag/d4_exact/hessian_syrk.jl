# diag/compressed-hessian-operator-audit-2026-07-25, addendum: does the
# dense-BLAS reference actually need `BLAS.gemm!` (computes BOTH triangles of
# the W-contraction, n^2 output entries) when only the upper triangle is ever
# read? `BLAS.syrk!` computes a symmetric rank-k update and writes ONLY the
# requested triangle -- this is a drop-in candidate for the CURRENT dense
# architecture's BLAS call, independent of whether the winner-pair candidate
# is adopted at all.
#
# CORRECTED CITATION (found live while wiring this benchmark): the REAL
# production Hessian for the D=20 unrestricted/origin-ZC (Architecture A)
# path is `cc_algo/PsiObjectiveBundle.jl:598-618`
# (`hessian!(h, obj::Union{PsiObjectiveBundleImplicit,PsiObjectiveBundleDelta})`)
# -- NOT `full_aod_diag/PsiObjectiveBundleImplicitMethodB_fullA.jl:185`'s
# near-identical D=4-context-only twin, which this file's first draft (and
# this audit's D=4 validation script, correctly, since that script really
# does use a PsiObjectiveBundleImplicitMethodBFullA-typed obj) targeted.
# obj's REAL runtime type at D=20 is `PsiObjectiveBundleImplicit`, which does
# NOT subtype `PsiObjectiveBundleImplicitMethodBFullA` -- confirmed by a live
# MethodError when this file's first draft was benchmarked against the real
# D=20 context. Both functions are otherwise byte-identical in logic (same
# @unpack, same ddPsi!/sqrt-weight/gemm! sequence, same packing loop) modulo
# cc_algo's `_enter_callback!/_exit_callback!` guard, reproduced here for a
# fair apples-to-apples timing comparison.
#
# BENCHMARK/CANDIDATE CODE ONLY -- not wired into any production driver.
"""
    hessian_syrk!(h, obj)

Identical inputs/outputs/packing convention to the real production `hessian!`
(cc_algo/PsiObjectiveBundle.jl:598), except the O(W*n^2) contraction uses
`BLAS.syrk!('U','T',...)` (writes only the upper triangle of the n x n
result) instead of `BLAS.gemm!('T','N',...)` (writes the full n x n result,
including the lower triangle nothing downstream ever reads).
"""
function hessian_syrk!(h, obj::Union{PsiObjectiveBundleImplicit, PsiObjectiveBundleDelta})
    _enter_callback!(obj)
    try
        @unpack H, H_copy, M, arg0, arg2, ddPsi!, outer_constr_index, ∂∂f_∂∂x = obj
        ddPsi!(arg2, arg0)
        @views H_copy[:, 2:1+outer_constr_index] .= H[:, 2:1+outer_constr_index]
        @views H_copy[:, 2:1+outer_constr_index] .*= .√arg2
        @views BLAS.syrk!('U', 'T', 1/M, H_copy[:, 2:1+outer_constr_index], 0.0, ∂∂f_∂∂x)
        k = 1
        for i in 1:size(∂∂f_∂∂x)[2]
            for j in i:size(∂∂f_∂∂x)[2]
                h[k] = ∂∂f_∂∂x[i, j]
                k += 1
            end
        end
    finally
        _exit_callback!(obj)
    end
end
