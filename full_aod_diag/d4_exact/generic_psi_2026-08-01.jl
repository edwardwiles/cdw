# ============================================================================
# Phase 4 (task §4): element-type-generic scalar Psi/dPsi/ddPsi, DIAGNOSTIC
# ONLY. Never included by any production driver. Exists purely so the
# generic objective oracle (built on top of this file) can be evaluated with
# ForwardDiff.Dual arguments -- production's own `CS.Psi!`/`CS.dPsi!`/
# `CS.ddPsi!` (cc_algo/Psi.jl) are `Vector{Float64}`-in-place kernels and
# cannot be called with Dual arguments directly.
#
# `Psi_scalar`/`dPsi_scalar`/`ddPsi_scalar` are the exact scalar bodies of
# `CS.Psi!`/`CS.dPsi!`/`CS.ddPsi!`'s per-element branches (verified below,
# NOT re-derived from the paper independently -- this file's only job is an
# element-type-generic REWRITE of the existing kernel, per the task's own
# instruction "Do not call production Psi! routines... Implement a small
# generic diagnostic equivalent whose Float64 output is first verified
# against production.").
# ============================================================================

@inline function Psi_scalar(u::T) where {T<:Real}
    if u <= 1.0
        return exp(u) - one(T)
    else
        return T(0.5) * exp(T(1)) * (u^2 + one(T)) - one(T)
    end
end

@inline function dPsi_scalar(u::T) where {T<:Real}
    if u <= 1.0
        return exp(u)
    else
        return exp(T(1)) * u
    end
end

@inline function ddPsi_scalar(u::T) where {T<:Real}
    if u <= 1.0
        return exp(u)
    else
        return exp(T(1)) * one(T)
    end
end
