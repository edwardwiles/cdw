# ============================================================================
# Part 5: wire gradient_method in {:pointwise_ad, :fixed_dual_fd, :boundary}
# into the ACTUAL production sequential outer loop
# (sequential_gravity/run_profiled_production.jl::make_seq_div_grad_fn!).
#
# DESIGN (chosen to minimize risk of a new sign/scale bug after several were
# already caught and fixed in Parts 1-2): rather than re-deriving
# PsiObjectiveBundleImplicitMethodB's exact sign/scale convention for a
# ground-up replacement, this computes the ADDITIVE CORRECTION on top of the
# existing, unmodified, already-correctly-wired AD gradient:
#
#   g_free_corrected = g_free_OLD_AD + CORRECTION
#
# This is valid because Part 1/2 proved an EXACT identity: the "intensive"
# (smooth, within-winner-regime) part of the corrected derivative equals the
# OLD AD derivative exactly (`intensive_part == -ad` to machine precision, see
# boundary_derivative.jl / Part 2 report) -- so the ENTIRE difference between
# a corrected method and AD is the boundary term alone. Consequences:
#   - CORRECTION is exactly zero for gamma'_focal (x_free[1]) -- that
#     coordinate has no winner/argmax dependence at all, AD is already exact
#     there (verified in Part 1/2), no correction needed or computed.
#   - CORRECTION is exactly zero for the gravity-linearized moment's own
#     contribution -- untouched, still computed by the existing
#     `_methodB_envelope_scalar` AD path (an affine surrogate with no winner
#     dependence, also already exact -- see the architecture note's Part 6
#     finding).
#   - Setting method=:pointwise_ad makes CORRECTION identically zero, so this
#     function is GUARANTEED to reproduce make_seq_div_grad_fn!'s output
#     exactly (not just approximately) -- a free, built-in regression check
#     every time this file is used, not merely at the moment it was written.
# ============================================================================

"""
    make_seq_div_grad_fn_corrected!(obj, fpmap, γobj, U, D, β, method; Acol_offset=3, h_fd=0.1)

Drop-in replacement for `run_profiled_production.jl::make_seq_div_grad_fn!`.
`method` ∈ (:pointwise_ad, :fixed_dual_fd, :boundary). Same closure signature
`(g_free, x_free, θ_full, inner_x) -> g_free`.
"""
function make_seq_div_grad_fn_corrected!(obj, fpmap, γobj, U::Matrix{Float64}, D::Int, β::Real, method::Symbol;
        Acol_offset::Int=3, h_fd::Float64=0.1)
    d = obj.d; oci = obj.outer_constr_index
    D1 = D + 1
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        # ---- 1. EXACT existing AD path, unchanged (this is g_free_OLD_AD) ----
        obj(inner_x, Float64[], Float64[]; constr=zeros(1))   # trigger dPsi!, populate obj.arg1 (same as original)
        λfull = collect(@view inner_x[2:end])                  # length D+2 (D focal shares, 1 price-index, 1 gravity-linearized)
        Usub = obj.U[1:obj.N, :]
        θ0 = reconstruct_full(x_free, fpmap)
        f_ad = x -> CS._methodB_envelope_scalar(reconstruct_full(x, fpmap), obj.moments!, obj.γ, Usub, λfull, obj.arg1, d, oci)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f_ad, x_free)
        end
        ForwardDiff.gradient!(g_free, f_ad, x_free, cfg_cache[])

        method == :pointwise_ad && return g_free   # CORRECTION ≡ 0 -- exact passthrough, see module docstring

        # ---- 2. CORRECTION for the Acol block only, via the D+1-baseline-moment machinery ----
        sign_conv = obj.find_smallest ? -1.0 : 1.0
        Acol0 = θ0[Acol_offset+1:Acol_offset+D]
        λ_baseline = λfull[1:D1]                                # the SAME D+1 baseline dual weights (focal shares + price-index)
        x_fixed_baseline = vcat(inner_x[1], λ_baseline)         # (zeta, lambda_1..D+1) -- same zeta, restricted lambda

        bundle_tmp = build_fixed_dual_bundle(γobj, U, length(θ0), D1, EK_moments_focal_norm_directgp!)
        bundle_tmp.moments!(@view(bundle_tmp.H[:, 1]), CS.select_G_from_H(bundle_tmp, bundle_tmp.H), θ0, bundle_tmp.U, bundle_tmp)
        bundle_tmp.H[:, 2] .= 1.0
        bundle_tmp(x_fixed_baseline, zeros(length(x_fixed_baseline)))   # refresh bundle_tmp.arg1 at (θ0, x_fixed_baseline)

        # ad_grad_log (baseline-only AD, in the validated dual_criterion_fixed_x sign/log convention) --
        # needed so CORRECTION = method_log - ad_log picks out ONLY the boundary piece (Part 1/2 identity).
        λ_x0 = collect(@view x_fixed_baseline[2:end])
        ad_grad_full_raw = ForwardDiff.gradient(θθ -> CS._methodB_envelope_scalar(θθ, EK_moments_focal_norm_directgp!, γobj, Usub, λ_x0, bundle_tmp.arg1, D1, D1 + 1), θ0)
        sign_conv_baseline = bundle_tmp.find_smallest ? -1.0 : 1.0
        ad_grad_log_baseline = (sign_conv_baseline .* (-ad_grad_full_raw ./ 1e10))[Acol_offset+1:Acol_offset+D] .* Acol0

        method_grad_log = if method == :fixed_dual_fd
            fixed_dual_fd_gradient(θ0, γobj, U, EK_moments_focal_norm_directgp!, D1, x_fixed_baseline, h_fd; l=length(θ0), Acol_offset=Acol_offset)
        elseif method == :boundary
            g, _ = boundary_envelope_gradient(θ0, bundle_tmp, x_fixed_baseline, γobj, U, D, β; Acol_offset=Acol_offset)
            g
        else
            error("unknown gradient_method $method")
        end

        # CORRECTION, converted from (signed, log-Acol) units into g_free's native (-1e10-scaled, level-Acol) units.
        correction = (-1e10 .* sign_conv_baseline) .* (method_grad_log .- ad_grad_log_baseline) ./ Acol0
        @views g_free[2:D+1] .+= correction
        return g_free
    end
end
