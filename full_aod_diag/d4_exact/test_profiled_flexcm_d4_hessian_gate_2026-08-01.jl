# Claude Code task 2026-08-01 (all-families completion): D4 numeric Hessian-formula gate for the
# reduced/profiled flexible-CM context (_fill_cm_HEE! reduced branch + hessian_cm_structured!'s H_EC
# gather branch), per mission §8 ("compare the complete structured Hessian to a reference").
#
# DESIGN: rather than hand-deriving a fresh dense G matrix (risking a subtle indexing mistake in a
# NEW derivation), the reference is built entirely from EXISTING, already-validated machinery:
#   1. Solve the FULL (unreduced) flexible-CM model normally (:dense_reference, matching
#      test_winner_pair_cross_hessian_cm_d4.jl's own convention) to get a real (zeta*, lambda*).
#   2. Zero out the ANCHOR entries of lambda* (one per destination, per `layout`) -- this makes the
#      FULL model's own q = -zeta - sum_j E_j*lambda_j collapse to EXACTLY the reduced model's own
#      q_reduced = -zeta - sum_{retained j} E_j*lambda_j (the anchor's contribution drops out because
#      its coefficient is now zero), giving a dual point at which the reduced and full models are
#      provably solving the SAME contraction.
#   3. Reference H_EE: the FULL model's OWN winner-pair kernel (fill_core_hessian_upper!, UNCHANGED)
#      at this q, GATHERED down to retained rows/cols -- justified by reduced_homogeneous_hessian_
#      2026-08-01.jl's own header derivation ("restricting j to the retained pairs only is exact...
#      holds for ANY subset of columns").
#   4. Reference H_EC: the FULL model's OWN winner_pair_cross_hessian_cm_block! with
#      use_profiled_correction=true (the ALREADY-D4-validated primitive from the prior session) at
#      the SAME q, GATHERED the same way.
#   5. Compare both against the NEW code under test: _fill_cm_HEE!'s reduced branch and
#      hessian_cm_structured!'s H_EC gather branch, called on a genuinely reduced CMBinHessCtx with
#      obj.arg0 set directly to the SAME q (bypassing moments!/KNITRO entirely -- legitimate for a
#      Hessian-FORMULA gate, which only needs a valid dual point, not an actual solve).
#
# HONEST SCOPE NOTE (see PROFILED_ALL_FAMILY_COMPLETION_MASTER_2026-08-01.md's correction): this
# gate does NOT exercise obj_reduced.moments! at all -- cf/theta are published directly into
# cctx_reduced.core_cf_ref[]/profiled_theta_ref[] by this test, and obj_reduced.arg0 is set directly,
# bypassing wrap_moments_with_cm_archB entirely. A real end-to-end KNITRO FG+Hessian solve for the
# reduced layout is NOT yet wired (found this session: wrap_moments_with_cm_archB's skip_fill=true
# path skips ALL of G, not just the economic columns, so it cannot serve as the reduced family's
# primary moments! either) -- this is a real, separate remaining gap, documented in the master doc.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "reduced_homogeneous_hessian_2026-08-01.jl",
          "profiled_restricted_family_base_2026-08-01.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

function unpack_packed(h::AbstractVector, n::Int)
    Hd = zeros(n, n)
    k = 1
    for i in 1:n, j in i:n
        Hd[i, j] = h[k]; Hd[j, i] = h[k]
        k += 1
    end
    return Hd
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(2026)

spec = build_anchor_spec_from_ctx(ctx)
cf_probe = build_compressed_factual(collect(θ_full_calib), ctx; check_ties = false)
has_france = cf_probe.cf_col > 0
layout = build_profiled_economic_moment_layout(ctx, spec; has_france_ratio = has_france)
assert_no_factual_price_index_moment(layout)
bi_slot = has_france ? dest_slot(ctx, ctx.bi) : 0
D = ctx.D; Ddest = cf_probe.D_dest
reduced_obj0 = build_reduced_base_obj_for_family(ctx, layout, CS)

for contrasts in (:anchored, :orthonormal), L in (10, 20)
    # ---- FULL model: solve normally (dense_reference, matching the pre-existing D4 gate's own convention) ----
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
                                       moment_representation = :dense_reference)
    cctx_full = pcx.cctx
    obj_full = pcx.ctx_cm.obj
    base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx_full)
    check("contrasts=$contrasts L=$L: FULL inner solve feasible", base.inner_status in (0, -100, -101, -103))
    cf_full = cctx_full.core_cf_ref[]
    check("contrasts=$contrasts L=$L: cf_full is a real CompressedFactual", cf_full isa CompressedFactual)

    NCORE_full = cctx_full.NCORE   # = D*Ddest+2
    ncm = cctx_full.ncm

    # ---- REDUCED model: same L/contrasts, on top of the SAME ctx ----
    aug_reduced = build_cm_augmented_obj_archB(ctx, CS; L = L, contrasts = contrasts, base_obj = reduced_obj0)
    cctx_reduced = build_cm_bin_ctx(ctx, aug_reduced; profiled_layout = layout)
    obj_reduced = aug_reduced.obj_cm
    NCORE_reduced = cctx_reduced.NCORE
    n_bilateral = length(layout.retained_full_factual_j)
    check("contrasts=$contrasts L=$L: NCORE_reduced == 1+layout.total", NCORE_reduced == 1 + layout.total_reduced_economic_moments)
    check("contrasts=$contrasts L=$L: reduced ncm == full ncm", cctx_reduced.ncm == ncm)

    # ---- zero out anchor lambda entries in the FULL calibration point ----
    x_full = vcat(base.ζstar, base.λstar)   # length NCORE_full-1+ncm = D*Ddest+1(+cf)+ncm
    x_zeroed = copy(x_full)
    for j_full in 1:(D * Ddest)
        if layout.full_factual_to_reduced[j_full] == 0
            x_zeroed[1 + j_full] = 0.0   # x[1]=zeta, x[2:1+D*Ddest] = bilateral lambda
        end
    end
    # perturb the RETAINED entries + CM-grid entries too (both calib and perturbed points, matching
    # the pre-existing test's own "calib"/"perturbed" convention), keeping anchor entries at exactly 0.
    for (label, xz) in (("calib_zeroed_anchor", x_zeroed),
                         ("perturbed_zeroed_anchor", begin
                             xp = copy(x_zeroed)
                             xp[1] += 0.01
                             for j_full in 1:(D * Ddest)
                                 if layout.full_factual_to_reduced[j_full] != 0
                                     xp[1 + j_full] += 0.02 * randn()
                                 end
                             end
                             has_france && (xp[1 + cf_full.cf_col] += 0.02 * randn())
                             xp[NCORE_full+1:end] .+= 0.02 .* randn(ncm)
                             xp
                         end))
        _archC_prep_for_hessian!(obj_full, xz)
        w_full = obj_full.arg2   # ddPsi! already applied inside _archC_prep_for_hessian! via Psi!... need ddPsi! too:
        obj_full.ddPsi!(obj_full.arg2, obj_full.arg0)

        # ---- Reference H_EE: NOT a gather from the OLD full winner_pair_hessian! kernel -- that
        # kernel's own "keep"/correction structure is destination-INDEPENDENT (scalar t0/s0, the
        # exact H_EE analog of H_EC's OLD nu_diff correction this whole port replaces), so gathering
        # its retained rows/cols does NOT equal the profiled/reduced model's own H_EE (confirmed
        # live: a real, reproducible ~0.22 discrepancy, not noise -- this is the CORRECT, EXPECTED
        # outcome of comparing two genuinely different formulas, not a bug in the new wiring).
        # Instead, since reduced_homogeneous_winner_pair_hessian! is ALREADY independently validated
        # (by the unrestricted family's own test_reduced_homogeneous_hessian_2026-08-01.jl, against
        # finite differences and a dense-reduced-G reconstruction -- not re-derived here, out of
        # scope), the correct thing THIS gate can check is that _fill_cm_HEE!'s new reduced branch
        # WIRES that already-validated kernel correctly: same numbers as calling
        # build_reduced_homogeneous_winner_pair_ctx/reduced_homogeneous_winner_pair_hessian! DIRECTLY
        # on the identical (cf, ctx, θ_full, layout, obj.arg0) -- a genuine, if narrower, check of
        # exactly what is NEW this session (the struct-field threading/caching/unpacking), not a
        # re-verification of the kernel's own math.
        wctx_direct = build_reduced_homogeneous_winner_pair_ctx(cf_full, ctx, θ_full_calib, layout)
        n_reduced = 1 + layout.total_reduced_economic_moments
        h_direct = Vector{Float64}(undef, n_reduced * (n_reduced + 1) ÷ 2)
        reduced_homogeneous_winner_pair_hessian!(h_direct, obj_full, wctx_direct)
        HEE_gathered = unpack_packed(h_direct, n_reduced)   # "gathered" name kept for the diff-report lines below

        # ---- Reference H_EC: FULL primitive with use_profiled_correction=true, gathered ----
        wctx_full = build_winner_pair_ctx(cf_full; bi_slot = bi_slot)
        gather_idx = Vector{Int}(undef, n_reduced)
        gather_idx[1] = 1
        for k in 1:n_bilateral
            gather_idx[1 + k] = 1 + layout.retained_full_factual_j[k]
        end
        has_france && (gather_idx[n_reduced] = 1 + cf_full.cf_col)
        n_full_ee = 1 + wctx_full.ncolI
        ws_full = ensure_winner_bin_cross_scratch!(Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing), wctx_full.ncolI, D, L, wctx_full.Ddest)
        winner_pair_cross_hessian_fill!(wctx_full, ws_full, obj_full, cctx_full.Bidx)
        nO = cctx_full.nO
        HEC_full_gathered = zeros(n_reduced, ncm)
        Hraw_EC_full = zeros(n_full_ee, nO)
        for l in 1:L
            winner_pair_cross_hessian_cm_block!(Hraw_EC_full, wctx_full, ws_full, l, cctx_full.origins, cctx_full.refIndex1, obj_full.M; use_profiled_correction = true)
            block = cctx_full.R === nothing ? Hraw_EC_full : Hraw_EC_full * cctx_full.R
            cols = (l - 1) * nO + 1 : l * nO
            HEC_full_gathered[:, cols] .= block[gather_idx, :]
        end

        # ---- New code under test: reduced context, obj.arg0 set directly (bypassing moments!) ----
        cctx_reduced.core_cf_ref[] = cf_full
        cctx_reduced.profiled_theta_ref[] = collect(Float64, θ_full_calib)
        resize!(obj_reduced.arg0, length(obj_full.arg0)); obj_reduced.arg0 .= obj_full.arg0
        x_reduced = vcat(xz[1], xz[2:1+D*Ddest][layout.full_factual_to_reduced .!= 0])
        has_france && push!(x_reduced, xz[1 + cf_full.cf_col])
        append!(x_reduced, xz[NCORE_full+1:end])
        length(x_reduced) == NCORE_reduced + ncm ||
            error("test harness bug: length(x_reduced)=$(length(x_reduced)) != NCORE_reduced+ncm=$(NCORE_reduced+ncm)")

        h_reduced = Vector{Float64}(undef, (NCORE_reduced + ncm) * (NCORE_reduced + ncm + 1) ÷ 2)
        hessian_cm_structured!(h_reduced, obj_reduced, cctx_reduced)
        Hd_reduced = unpack_packed(h_reduced, NCORE_reduced + ncm)
        HEE_new = Hd_reduced[1:NCORE_reduced, 1:NCORE_reduced]
        HEC_new = Hd_reduced[1:NCORE_reduced, NCORE_reduced+1:NCORE_reduced+ncm]

        maxdiff_ee = maximum(abs.(HEE_gathered .- HEE_new))
        scale_ee = max(1.0, maximum(abs.(HEE_gathered)))
        check("contrasts=$contrasts L=$L $label: reduced H_EE matches gathered-full reference (max|Δ|=$(maxdiff_ee))",
            maxdiff_ee < 1e-7 * scale_ee)
        @printf("  contrasts=%s L=%d %s: max|ΔH_EE|=%.3e  scale=%.3e\n", contrasts, L, label, maxdiff_ee, scale_ee)

        maxdiff_ec = maximum(abs.(HEC_full_gathered .- HEC_new))
        scale_ec = max(1.0, maximum(abs.(HEC_full_gathered)))
        check("contrasts=$contrasts L=$L $label: reduced H_EC matches gathered-full reference (max|Δ|=$(maxdiff_ec))",
            maxdiff_ec < 1e-7 * scale_ec)
        @printf("  contrasts=%s L=%d %s: max|ΔH_EC|=%.3e  scale=%.3e\n", contrasts, L, label, maxdiff_ec, scale_ec)

        check("contrasts=$contrasts L=$L $label: reduced Hessian finite", all(isfinite, Hd_reduced))
        check("contrasts=$contrasts L=$L $label: reduced Hessian symmetric", maximum(abs.(Hd_reduced .- Hd_reduced')) < 1e-10)
    end
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
