# Task §6.1 (profiled-outer-ab-completion-2026-08-04): the ONE boundary-layer dispatch between
# KNITRO's own outer vector (in whichever A-coordinate mode a run selects) and REDUCED's native
# r_free space that evaluate_profiled_point/shared_family_outer_gradient/the family adapters/every
# existing decode_outer_profiled call site already operate in UNCHANGED. Keeping the boundary this
# thin means none of those ~15 existing call sites, nor evaluate_profiled_point's own internals,
# need to know a second coordinate mode exists at all -- only the true outer boundary
# (initial point, bounds, the KNITRO callback in run_profiled_upper_constrained, checkpoint
# namespace, and the final gradient) is mode-aware. ADDITIVE ONLY.
#
# :profiled_powered_relative_A is FIXED-THETA ONLY (derivation doc §3d: the `-theta` gradient
# rescale is a SCALAR because theta is constant for every REDUCED family that mode applies to). It
# does NOT extend to :unrestricted, whose theta is itself searched jointly (eta_theta=log(theta)) --
# doing so would require a genuinely new derivation (the Jacobian would need a d(theta)/d(eta_theta)
# term this task's own math foundation explicitly declined to add: "only the flexible-theta
# unrestricted FULL driver would need a per-call-varying theta, which REDUCED does not support and
# this task does not add", POWERED_PROFILED_COORDINATE_DERIVATION_2026-08-04.md §3d). Requesting
# powered mode for :unrestricted is therefore a hard error here, not a silent fallback to native.

isdefined(Main, :encode_powered_relative_A) ||
    error("profiled_coordinate_mode_dispatch_2026-08-04.jl requires profiled_powered_relative_a_2026-08-04.jl to be included first.")

const PROFILED_A_COORDINATE_MODES = (:profiled_pivot_anchor_relative, :profiled_powered_relative_A)

"Hard error (not a silent pass-through) on any mode other than the two known REDUCED A-coordinate modes."
function validate_profiled_a_coordinate_mode(mode::Symbol)
    mode in PROFILED_A_COORDINATE_MODES ||
        error("validate_profiled_a_coordinate_mode: unknown mode=:$mode, expected one of $PROFILED_A_COORDINATE_MODES")
    return mode
end

"Hard error if `mode` is powered and `family` is :unrestricted -- powered mode is fixed-theta only, see file header."
function validate_mode_family_compatibility(mode::Symbol, family::Symbol)
    validate_profiled_a_coordinate_mode(mode)
    (mode == :profiled_powered_relative_A && family == :unrestricted) &&
        error("validate_mode_family_compatibility: :profiled_powered_relative_A is fixed-theta only and does not apply to :unrestricted (theta is jointly searched there) -- use :profiled_pivot_anchor_relative for this family.")
    return nothing
end

"""
    decode_w_mode_to_r_free(w_mode_free, pe, mode, theta, xy) -> r_free

`w_mode_free` = the FREE (non-gp) block of KNITRO's own outer vector, in `mode`'s own units.
Native: identity (copy). Powered: `decode_powered_relative_A`.
"""
function decode_w_mode_to_r_free(w_mode_free::AbstractVector{Float64}, pe, mode::Symbol, theta::Float64, xy)
    validate_profiled_a_coordinate_mode(mode)
    mode == :profiled_pivot_anchor_relative && return collect(Float64, w_mode_free)
    return decode_powered_relative_A(w_mode_free, pe, theta, xy)
end

"Inverse of `decode_w_mode_to_r_free`."
function encode_r_free_to_w_mode(r_free::AbstractVector{Float64}, pe, mode::Symbol, theta::Float64, xy)
    validate_profiled_a_coordinate_mode(mode)
    mode == :profiled_pivot_anchor_relative && return collect(Float64, r_free)
    return encode_powered_relative_A(r_free, pe, theta, xy)
end

"""
    rescale_gradient_for_mode(g_native_free, mode, theta) -> g_mode_free

Chain rule from the native r_free-space analytic gradient (REDUCED's existing, unchanged
`shared_family_outer_gradient` A-block output) to `mode`'s own coordinate gradient. Native:
identity. Powered: `powered_relative_gradient_rescale` (derivation doc §3d). Operates ONLY on the
free-A block -- gp's own gradient component is coordinate-mode-independent and must be handled
separately by the caller (see `rescale_full_gradient_for_mode` below).
"""
function rescale_gradient_for_mode(g_native_free::AbstractVector{Float64}, mode::Symbol, theta::Float64)
    validate_profiled_a_coordinate_mode(mode)
    mode == :profiled_pivot_anchor_relative && return g_native_free
    return powered_relative_gradient_rescale(g_native_free, theta)
end

"Full outer gradient (gp ++ A-block) chain rule: gp component passes through unchanged, only the A-block is rescaled."
function rescale_full_gradient_for_mode(g_native::AbstractVector{Float64}, mode::Symbol, theta::Float64)
    return vcat(g_native[1], rescale_gradient_for_mode(g_native[2:end], mode, theta))
end

"""
    mode_bounds(r_lo_free, r_hi_free, pe, mode, theta, xy) -> (lo, hi)

Native: identity. Powered: `powered_relative_bounds` (order-flip handled there).
"""
function mode_bounds(r_lo_free::AbstractVector{Float64}, r_hi_free::AbstractVector{Float64}, pe, mode::Symbol, theta::Float64, xy)
    validate_profiled_a_coordinate_mode(mode)
    mode == :profiled_pivot_anchor_relative && return (r_lo_free, r_hi_free)
    return powered_relative_bounds(r_lo_free, r_hi_free, pe, theta, xy)
end

"""
    encode_w_native_to_mode(w_native, pe, mode, theta, xy) -> w_mode

Full outer vector (gp ++ A-block), native -> mode. gp passes through unchanged.
"""
function encode_w_native_to_mode(w_native::AbstractVector{Float64}, pe, mode::Symbol, theta::Float64, xy)
    return vcat(w_native[1], encode_r_free_to_w_mode(w_native[2:end], pe, mode, theta, xy))
end

"""
    decode_w_mode_to_native(w_mode, pe, mode, theta, xy) -> w_native

Inverse of `encode_w_native_to_mode`.
"""
function decode_w_mode_to_native(w_mode::AbstractVector{Float64}, pe, mode::Symbol, theta::Float64, xy)
    return vcat(w_mode[1], decode_w_mode_to_r_free(w_mode[2:end], pe, mode, theta, xy))
end
