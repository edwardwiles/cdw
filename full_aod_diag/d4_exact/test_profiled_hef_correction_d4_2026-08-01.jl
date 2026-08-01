# Profiled economic block port (2026-08-01), commit 4b: D4 gate for the NEW
# use_profiled_correction=true path in winner_pair_cross_hessian_colsum!/_esum! (H_EF).
# Standalone-primitive style (mirrors test_profiled_hec_correction_d4_2026-08-01.jl): calls the
# primitives directly rather than through hessian_cm_frechet_structured!/_v2! (which don't yet
# thread use_profiled_correction through -- that wiring is separate, tracked follow-on work).
#
# Two independent checks per primitive, same structure as H_EC/H_EZ's gates:
#   1. Regression safety: use_profiled_correction=false unchanged (compared against a fully
#      independent brute-force re-derivation of the OLD formula from raw cf/Bidx data).
#   2. New-path correctness: use_profiled_correction=true vs an independent brute-force
#      T^F_{d,l} = sum_x sum_{w: Bidx[w,x]<=l} S_w*wval[w,d] (colsum!) / sum_x sum_w S_w*wval[w,d]
#      (esum!, un-binned) -- built directly from raw data, not reusing MCScum/MSumX/T0_slot.
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "three_way_derivatives.jl",
          "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl", "cm_lookup_kernels.jl",
          "lfix_cm_aware.jl", "threaded_cross_hessian.jl",
          "hcz_drawchunk_candidate_2026-07-29.jl", "zc_restriction_operator.jl",
          "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_hessian_threaded.jl", "cm_production_bundle.jl",
          "cm_screen_bridge.jl", "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl", "cm_outer_driver.jl", "cm_config.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_lookup_production.jl",
          "cm_checkpoint.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, LinearAlgebra, Random

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
end

"""
Independent brute-force sumQ (the unchanged 'keep' term colsum! computes) and T^F_{d,l} (the new
correction), from raw cf/Bidx data only. `sumQ` is kappa0[j]-SCALED (matches QCScum's own
convention, y[w,slot]=kappa0[j]*wval[w,slot] -- see build_winner_pair_ctx); `Tdl` is RAW/unscaled
(matches MCScum's convention, mv=Snu*wval with no kappa0 -- pi_vec[j] already carries kappa0[j] as
a factor, same scaling discipline as test_profiled_hec_correction_d4_2026-08-01.jl's own
brute_force_hec_entry).
"""
function brute_force_colsum_pieces(cf::CompressedFactual, Bidx::AbstractMatrix{<:Integer}, S::AbstractVector{Float64},
        j::Int, slot_row::Int, o_row::Int, D::Int, l::Int, kappa0_j::Float64)
    W = cf.W
    nu = cf.SW
    sumQ_raw = 0.0
    Tdl = 0.0
    for x in 1:D
        for w in 1:W
            Bidx[w, x] <= l || continue
            if cf.winner[w, slot_row] == o_row
                sumQ_raw += S[w] * nu[w] * cf.wval[w, slot_row]
            end
            Tdl += S[w] * nu[w] * cf.wval[w, slot_row]
        end
    end
    return kappa0_j * sumQ_raw, Tdl
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)

for contrasts in (:anchored, :orthonormal), L in (10, 20)
    # moment_representation=:dense_reference explicitly (default is :operator, no dense obj.H --
    # see test_profiled_hec_correction_d4_2026-08-01.jl's identical note).
    pcx = build_cm_frechet_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
        cm_hessian_backend = :structured, cm_cross_hessian_backend = :winner_bin,
        moment_representation = :dense_reference)
    cctx = pcx.cctx
    obj = pcx.ctx_cm.obj
    base = archC_frechet_base_state(x_free_calib, pcx.ctx_cm, cctx, pcx.aug.level_targets)
    check("contrasts=$contrasts L=$L: inner solve feasible", base.inner_status in (0, -100, -101, -103))

    for (label, x) in (("calib", vcat(base.ζstar, base.λstar)),
                        ("perturbed", vcat(base.ζstar, base.λstar) .+ vcat(0.01, 0.02 .* randn(length(base.λstar)))))
        _archC_prep_for_hessian!(obj, x)
        cf = cctx.core_cf_ref[]
        check("contrasts=$contrasts L=$L $label: cf is a real CompressedFactual", cf isa CompressedFactual)
        cf isa CompressedFactual || continue

        wctx = build_winner_pair_ctx(cf)
        ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
        ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, cctx.L, wctx.Ddest)
        winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)
        S = obj.arg2
        D = cctx.D

        maxdiff_colsum_old = 0.0
        maxdiff_colsum_new = 0.0
        for l in (1, cctx.L)
            colsum_old = zeros(wctx.ncolI + 1)
            colsum_new = zeros(wctx.ncolI + 1)
            winner_pair_cross_hessian_colsum!(colsum_old, wctx, ws, l; use_profiled_correction = false)
            winner_pair_cross_hessian_colsum!(colsum_new, wctx, ws, l; use_profiled_correction = true)
            for j in 1:wctx.ncolI
                (wctx.has_cf && j == cf.cf_col) && continue
                slot_row = wctx.target_slot[j]
                o_row = div(j - slot_row, wctx.Ddest) + 1
                sumQ, Tdl = brute_force_colsum_pieces(cf, cctx.Bidx, S, j, slot_row, o_row, D, l, wctx.kappa0[j])
                sumNu_bf = 0.0  # old correction: independent global sum over all D CM-grid coords, all w with Bidx<=l
                for xx in 1:D, w in 1:cf.W
                    cctx.Bidx[w, xx] <= l && (sumNu_bf += S[w] * cf.SW[w])
                end
                expected_old = sumQ - wctx.pi_vec[j] * sumNu_bf
                expected_new = sumQ - wctx.pi_vec[j] * Tdl
                maxdiff_colsum_old = max(maxdiff_colsum_old, abs(colsum_old[j+1] - expected_old))
                maxdiff_colsum_new = max(maxdiff_colsum_new, abs(colsum_new[j+1] - expected_new))
            end
        end
        check("contrasts=$contrasts L=$L $label: colsum! use_profiled_correction=false regression-safe (max|Δ|=$(maxdiff_colsum_old))",
              maxdiff_colsum_old < 1e-8)
        check("contrasts=$contrasts L=$L $label: colsum! use_profiled_correction=true matches brute-force T^F (max|Δ|=$(maxdiff_colsum_new))",
              maxdiff_colsum_new < 1e-6)

        # ---- esum! (un-binned) ----
        Wtot = sum(S)
        Esum_old = zeros(wctx.ncolI + 1)
        Esum_new = zeros(wctx.ncolI + 1)
        winner_pair_cross_hessian_esum!(Esum_old, wctx, ws, S, Wtot; use_profiled_correction = false)
        winner_pair_cross_hessian_esum!(Esum_new, wctx, ws, S, Wtot; use_profiled_correction = true)
        maxdiff_esum_old = 0.0
        maxdiff_esum_new = 0.0
        for j in 1:wctx.ncolI
            (wctx.has_cf && j == cf.cf_col) && continue
            slot_row = wctx.target_slot[j]
            o_row = div(j - slot_row, wctx.Ddest) + 1
            EsumEcon_j = 0.0
            t0_bf = 0.0
            T0_bf = 0.0
            for w in 1:cf.W
                snu = S[w] * cf.SW[w]
                t0_bf += snu
                T0_bf += snu * cf.wval[w, slot_row]
                cf.winner[w, slot_row] == o_row && (EsumEcon_j += snu * cf.wval[w, slot_row])
            end
            # EsumEcon_j is kappa0[j]-SCALED in the real ws.EsumEcon (accumulates snuy=Snu[w]*y[w,slot],
            # y already kappa0-scaled) -- same scaling discipline as brute_force_colsum_pieces above.
            EsumEcon_j *= wctx.kappa0[j]
            expected_old = EsumEcon_j - wctx.pi_vec[j] * t0_bf
            expected_new = EsumEcon_j - wctx.pi_vec[j] * T0_bf
            maxdiff_esum_old = max(maxdiff_esum_old, abs(Esum_old[j+1] - expected_old))
            maxdiff_esum_new = max(maxdiff_esum_new, abs(Esum_new[j+1] - expected_new))
        end
        check("contrasts=$contrasts L=$L $label: esum! use_profiled_correction=false regression-safe (max|Δ|=$(maxdiff_esum_old))",
              maxdiff_esum_old < 1e-6)
        check("contrasts=$contrasts L=$L $label: esum! use_profiled_correction=true matches brute-force T^F_0 (max|Δ|=$(maxdiff_esum_new))",
              maxdiff_esum_new < 1e-6)
        @printf("  contrasts=%s L=%d %s: colsum old=%.3e new=%.3e | esum old=%.3e new=%.3e\n",
                contrasts, L, label, maxdiff_colsum_old, maxdiff_colsum_new, maxdiff_esum_old, maxdiff_esum_new)
    end
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
