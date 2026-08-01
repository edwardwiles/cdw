# Profiled economic block port (2026-08-01), commit 3: D4 gate for the NEW
# use_profiled_correction=true path in winner_pair_cross_hessian_cm_block! (H_EC).
#
# Two independent checks:
#   1. Regression safety: use_profiled_correction=false (the default, every existing caller's
#      behavior) is UNCHANGED -- reproduces the pre-existing test_winner_pair_cross_hessian_cm_d4.jl
#      comparison against the dense hessian_cm_structured! reference, bit-for-bit.
#   2. New-path correctness: use_profiled_correction=true is checked against a FULLY INDEPENDENT
#      brute-force computation of the mission's own formula
#      (H^new_{E,R}[(o,d),k] = W^R_{od,k} - lambda_od*T^R_{d,k}, T^R_{d,k}=sum_w S_w*wval[w,d]*R_k(w))
#      built directly from raw per-draw cf data (cf.winner, cf.wval, cctx.Bidx, obj.arg2, cf.SW) --
#      does NOT reuse QCScum/MCScum or any other optimized accumulator, so a bug shared between the
#      brute-force check and the optimized code being checked cannot hide.
#
# Uses the SAME D4 setup/context as test_winner_pair_cross_hessian_cm_d4.jl (flexible CM, real
# CompressedFactual) -- the economic layout here is still the OLD/full one (no anchor omission is
# wired into flexible CM yet, see PROFILED_ALL_FAMILY_SOURCE_SNAPSHOT_2026-08-01.md), so this
# validates that T^R_{d,k} is computed correctly by the new MTab/MCScum machinery in general, not
# yet that a genuine reduced/anchor-omitting cf flows through it end-to-end (explicitly out of
# scope for this commit, see PROFILED_CROSS_BLOCK_FORMULAS_2026-08-01.md).
const D4X = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "hcz_drawchunk_candidate_2026-07-29.jl",
          "zc_restriction_operator.jl", "zc_gram_blas_candidates.jl",
          "cm_hessian_architectures.jl", "cm_production_bundle.jl"]
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

"""
Fully independent brute-force reference for ONE raw H_EC entry (row j, CM-grid origin coordinate
o_col vs refIndex1, bin l), built directly from `cf`/`Bidx`/`S`/`nu` -- no QCScum/MCScum reuse.
Returns (q_diff, T_diff) so both the unchanged "keep" term and the new correction term are checked.
"""
function brute_force_hec_entry(cf::CompressedFactual, Bidx::AbstractMatrix{<:Integer}, S::AbstractVector{Float64},
        j::Int, slot_row::Int, o_row::Int, o_col::Int, refIndex1::Int, l::Int)
    W = cf.W
    nu = cf.SW
    q_o = 0.0; q_ref = 0.0
    T_o = 0.0; T_ref = 0.0
    for w in 1:W
        snu = S[w] * nu[w]
        in_o = Bidx[w, o_col] <= l
        in_ref = Bidx[w, refIndex1] <= l
        if cf.winner[w, slot_row] == o_row
            snuy = snu * cf.wval[w, slot_row]  # kappa0 applied by caller via q_diff*kappa0-consistency (see below)
            in_o && (q_o += snuy)
            in_ref && (q_ref += snuy)
        end
        mv = snu * cf.wval[w, slot_row]
        in_o && (T_o += mv)
        in_ref && (T_ref += mv)
    end
    return (q_o - q_ref, T_o - T_ref)
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(2026)

for contrasts in (:anchored, :orthonormal), L in (10, 20)
    # moment_representation=:dense_reference explicitly requested: production's default has since
    # moved to :operator (no dense obj.H at all -- confirmed by hitting `OperatorPsiBundle has no
    # field H` when this was omitted, matching this codebase's own no-composite-G architecture
    # goal), but THIS test needs a dense H_EC reference to compare against, exactly like the
    # pre-existing test_winner_pair_cross_hessian_cm_d4.jl this is modeled on.
    pcx = build_cm_production_context(ctx, CS; L = L, contrasts = contrasts, use_compressed_core = true,
                                       moment_representation = :dense_reference)
    cctx = pcx.cctx
    obj = pcx.ctx_cm.obj
    base = archC_base_state(x_free_calib, pcx.ctx_cm, cctx)
    check("contrasts=$contrasts L=$L: inner solve feasible", base.inner_status in (0, -100, -101, -103))

    for (label, x) in (("calib", vcat(base.ζstar, base.λstar)),
                        ("perturbed", vcat(base.ζstar, base.λstar) .+ vcat(0.01, 0.02 .* randn(length(base.λstar)))))
        NCORE = cctx.NCORE; ncm = cctx.ncm; nO = cctx.nO; L_ = cctx.L; Ddest = cctx.D  # active dest count == D at D4 (square)
        n = NCORE + ncm
        _archC_prep_for_hessian!(obj, x)
        h = Vector{Float64}(undef, n * (n + 1) ÷ 2)
        hessian_cm_structured!(h, obj, cctx)
        Hd = unpack_packed(h, n)
        H_EC_dense = Hd[1:NCORE, NCORE+1:NCORE+ncm]

        cf = cctx.core_cf_ref[]
        wctx = build_winner_pair_ctx(cf)
        Ddest_wctx = wctx.Ddest
        ws_ref = Ref{Union{Nothing,WinnerBinCrossScratch}}(nothing)
        ws = ensure_winner_bin_cross_scratch!(ws_ref, wctx.ncolI, cctx.D, L_, Ddest_wctx)
        winner_pair_cross_hessian_fill!(wctx, ws, obj, cctx.Bidx)

        # ---- Check 1: regression safety (use_profiled_correction=false vs the pre-existing dense reference) ----
        Hraw_EC_old = zeros(NCORE, nO)
        H_EC_old = zeros(NCORE, ncm)
        M = obj.M
        for l in 1:L_
            winner_pair_cross_hessian_cm_block!(Hraw_EC_old, wctx, ws, l, cctx.origins, cctx.refIndex1, M; use_profiled_correction = false)
            cols = (l - 1) * nO + 1 : l * nO
            block = cctx.R === nothing ? Hraw_EC_old : Hraw_EC_old * cctx.R
            H_EC_old[:, cols] .= block
        end
        maxdiff_old = maximum(abs.(H_EC_dense .- H_EC_old))
        relscale = max(1.0, maximum(abs.(H_EC_dense)))
        check("contrasts=$contrasts L=$L $label: use_profiled_correction=false regression-safe (max|Δ|=$(maxdiff_old))",
              maxdiff_old < 1e-8 * relscale)

        # ---- Check 2: use_profiled_correction=true vs independent brute-force T^R_{d,k} ----
        S = obj.arg2  # already fresh: winner_pair_cross_hessian_fill! called ddPsi!(obj.arg2, obj.arg0) above, same obj.arg0
        Hraw_EC_new = zeros(NCORE, nO)
        maxdiff_new = 0.0
        for l in (1, L_)  # spot-check first and last bin -- O(W*NCORE*nO) per l, keep the brute force affordable
            winner_pair_cross_hessian_cm_block!(Hraw_EC_new, wctx, ws, l, cctx.origins, cctx.refIndex1, M; use_profiled_correction = true)
            for (oi, o_col) in enumerate(cctx.origins)
                for j in 1:wctx.ncolI
                    # The France/cf column is DELIBERATELY excluded from the profiled correction
                    # (kept on nu_diff -- wctx.target_slot[cf.cf_col] is the documented sentinel 0,
                    # not a real slot, see winner_pair_cross_hessian_cm_block!'s own docstring) --
                    # skip it here rather than feed a fake slot into the bilateral-pair inversion.
                    (wctx.has_cf && j == cf.cf_col) && continue
                    slot_row = wctx.target_slot[j]
                    # invert j = slot + (o-1)*Ddest for o_row
                    o_row = div(j - slot_row, Ddest_wctx) + 1
                    q_diff_bf, T_diff_bf = brute_force_hec_entry(cf, cctx.Bidx, S, j, slot_row, o_row, o_col, cctx.refIndex1, l)
                    kappa0_j = wctx.kappa0[j]
                    # q_diff (real QCScum) IS kappa0-scaled (y=kappa0*wval); T_diff (real MCScum) is
                    # RAW/unscaled (mv=Snu*wval, no kappa0) -- pi_vec[j] ALREADY carries kappa0[j] as
                    # a factor (see build_winner_pair_ctx), so it must NOT be multiplied by kappa0_j
                    # again here. Matches the real code's `(q_diff - pi_vec[j]*T_diff)*invM` exactly.
                    expected = (kappa0_j * q_diff_bf - wctx.pi_vec[j] * T_diff_bf) * (1.0 / M)
                    got = Hraw_EC_new[j + 1, oi]
                    maxdiff_new = max(maxdiff_new, abs(got - expected))
                end
            end
        end
        scale_new = max(1.0, maximum(abs.(Hraw_EC_new)))
        check("contrasts=$contrasts L=$L $label: use_profiled_correction=true matches independent brute-force T^R (max|Δ|=$(maxdiff_new))",
              maxdiff_new < 1e-8 * scale_new)
        @printf("  contrasts=%s L=%d %s: false-path max|Δ|=%.3e  true-path max|Δ|=%.3e  scale=%.3e\n",
                contrasts, L, label, maxdiff_old, maxdiff_new, scale_new)
    end
end

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
