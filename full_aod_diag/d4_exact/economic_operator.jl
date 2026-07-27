# ================================================================================================
# port/shared-inner-fg-operator-and-verification-2026-07-26, Job 1 §2: ONE shared economic-core
# FG operator, `economic_forward!`/`economic_transpose!`, used by EVERY family (unrestricted and
# all four restricted families) for their common E = Q - νπ' block.
#
# NOT a new derivation. This file is a thin, explicitly-named wrapper around the ALREADY-VALIDATED
# `compressed_dual_contraction!`/`compressed_transpose_contraction!` (compressed_moments.jl /
# compressed_cc_inner.jl, Addendum Part A) -- the same math the unrestricted family's default
# production FG callback already uses. Per the addendum's explicit instruction ("Do not copy this
# algebra into family-specific files"), every family below calls THESE two functions directly
# rather than each re-deriving or re-implementing the winner-gather/scatter contraction.
#
# WHERE THE `CompressedFactual` COMES FROM FOR RESTRICTED FAMILIES: every restricted family's
# `moments!` closure (`wrap_moments_with_cm_archB`/`wrap_moments_with_cm_frechet_archB`/
# `wrap_moments_with_cm_meanzc`/`wrap_moments_with_originzc`) ALREADY builds a `CompressedFactual`
# for the economic-core block on every outer point and publishes it via `core_cf_ref[]` -- this was
# built for the shared winner-pair Hessian backend (port/shared-winner-pair-core-hessian-production
# -2026-07-25), not for FG. This file's whole contribution is: FG callbacks can and should consume
# the SAME `cf` for their own forward/backward instead of a dense `BLAS.gemv!` against `obj.H`'s
# core columns -- no new `CompressedFactual`-construction machinery is needed.
#
# SCOPE NOTE (economic core vs Hessian): the production Hessian for restricted families
# (`hessian_cm_structured!`/Architecture C, `archA_partitioned_hess_cb_builder`/Architecture A-
# partitioned) still reads `obj.H[:, 2:1+NCORE]` (dense E columns) directly for its H_EC/H_ER
# cross-terms -- confirmed by reading both callbacks, not assumed. Eliminating THAT dependency
# would be Hessian cross-block rework, explicitly out of scope for this task ("Do not expand this
# task into Hessian cross-block optimization"). So `obj.H`'s dense E columns continue to be BUILT
# (by `moments!`, for the Hessian's legitimate use) even after this port -- what changes is that
# the FG *callback* stops READING them for its own forward/backward, exactly the same "producer
# still exists, but this particular consumer stops using it" pattern already established for the CM
# block's `skip_cm_fill_ref` (RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md §2). This is
# documented in full in docs/SHARED_ECONOMIC_FG_OPERATOR_DESIGN_2026-07-26.md.
# ================================================================================================

isdefined(Main, :EconomicFGWorkspace) || include(joinpath(@__DIR__, "compressed_cc_inner.jl"))

"""
    economic_operator_workspace(cf::CompressedFactual) -> EconomicFGWorkspace

Build (or, via the caller's own cache field, reuse) the persistent scratch every
`economic_forward!`/`economic_transpose!` call needs. Named separately from `EconomicFGWorkspace`
itself only so call sites read as "get me the shared economic operator's workspace" rather than
reaching for the unrestricted-family-flavored constructor name directly.
"""
economic_operator_workspace(cf::CompressedFactual) = EconomicFGWorkspace(cf)

"""
    economic_forward!(out, lambda_E, cf::CompressedFactual, ws::EconomicFGWorkspace) -> out

`out[s] = Σ_j lambda_E[j] * E_{s,j}` for every draw `s`, `E = Q - νπ'` the winner-sparse economic
moment block, computed WITHOUT materializing dense E (O(W·D_dest) + O(D·D_dest), the same
compressed winner gather the unrestricted family's own default FG callback uses). `lambda_E` must
have length `cf.oci - 1`. Writes into caller-supplied `out` (length W) -- no allocation, no dense
`E`/`obj.H` read. This is THE shared economic operator every family's FG assembly composes with
its own restriction operator (task §3): `G = E` (unrestricted), `G = [E|C]` (flexible CM),
`G = [E|C|F]` (common Frechet), `G = [E|C|Z]` (CM+ZC), `G = [E|Z]` (ZC-only).
"""
function economic_forward!(out::AbstractVector{Float64}, lambda_E::AbstractVector{Float64},
                            cf::CompressedFactual, ws::EconomicFGWorkspace)
    compressed_dual_contraction!(out, lambda_E, cf, ws.κ, ws.C)
    return out
end

"""
    economic_transpose!(grad_E, draw_weights, cf::CompressedFactual, ws::EconomicFGWorkspace) -> grad_E

`grad_E[j] = Σ_s draw_weights[s] * E_{s,j}` (the winner-SCATTER, transpose of `economic_forward!`),
length `cf.oci - 1`, O(W·D_dest) + O(D·D_dest), no dense `E`. `draw_weights` is typically
`Ψ'(r)` (the inner-dual gradient weight, e.g. `dPsq`/`arg1` after `dPsi!`), matching every family's
own `g_λ = -(1/M)·G'·Ψ'(r)` convention -- callers apply the `-(1/M)` scale themselves (this
function returns the raw `Σ_s w_s E_{s,j}` scatter, exactly mirroring
`compressed_transpose_contraction!`'s own contract) so it composes cleanly with a restriction
operator's own transpose via simple accumulation into the same gradient buffer at different column
offsets.
"""
function economic_transpose!(grad_E::AbstractVector{Float64}, draw_weights::AbstractVector{Float64},
                              cf::CompressedFactual, ws::EconomicFGWorkspace)
    compressed_transpose_contraction!(grad_E, draw_weights, cf, ws.B)
    return grad_E
end
