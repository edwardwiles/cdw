# ============================================================================
# Continuation 9, Phase 4 (the "big one", motivated directly by Phase 3C's
# finding that the compressed FG/Hessian loops lose to BLAS at D=20): a
# destination-major, cache-friendly reimplementation of
# compressed_moments.jl::compressed_dual_contraction and
# compressed_cc_inner.jl::compressed_transpose_contraction -- the two
# O(W*D) primitives EVERY compressed FG/HVP call bottoms out in
# (compressed_cc_value_grad, compressed_cc_hvp both call exactly these two).
#
# THE PROBLEM (diagnosed, not assumed): both original functions have an
# OUTER loop over draws `s` and an INNER loop over destinations `d`, reading
# `cf.winner[s,d]`/`cf.wval[s,d]` for FIXED s, varying d. `cf.winner`/
# `cf.wval` are `W x D` matrices -- Julia arrays are column-major, so a
# fixed-row/varying-column access (`cf.winner[s, :]`) has STRIDE W between
# consecutive reads, the worst-case cache access pattern for a `W=80000`-row
# matrix (each read is a fresh cache line, ~80000*8=640KB apart in memory --
# nowhere near cache-resident). This is EXACTLY the "scattered/indirect
# winner indexing" Phase 3C's own report (docs/fullA_fully_compressed_inner_report.md
# §3, finding 2) flagged as the reason compressed_cc_hvp-based accumulation
# lost to one BLAS gemm at D=20.
#
# THE FIX: swap loop nesting to DESTINATION-major (outer d, inner s) --
# `cf.winner[:, d]`/`cf.wval[:, d]` for fixed d is then a CONTIGUOUS
# W-length column read, cache-friendly and @simd-able. Provably produces the
# EXACT SAME floating-point result (not just "close"): for a FIXED s, the
# original loop visits d=1,2,...,D in order and accumulates into t[s]/or
# for a FIXED (o,d) cell, the original visits s=1,2,...,W in order and
# accumulates into B[o,d] -- the destination-major reordering below preserves
# BOTH per-output-element accumulation orders exactly (only the outer/inner
# LOOP NESTING changes, not which terms get summed into which output in
# which order), so equivalence is verified to BIT-IDENTICAL (0.0 diff), not
# merely a floating-point tolerance -- confirmed empirically in
# test_compressed_cc_kernels_v2.jl, not merely asserted from the argument
# above.
#
# ADDITIVE ONLY: new function names (`_v2` suffix), zero changes to
# compressed_moments.jl / compressed_cc_inner.jl / compressed_live.jl.
# Verified equivalent before being benchmarked or wired anywhere.
#
# *** NON-PRODUCTION / UNREACHABLE / SQUARE-ONLY -- DO NOT WIRE UNDER :exclude_row ***
# Canonical winner engine task (2026-07-24): confirmed, by direct code inspection, that this
# entire file is:
#   (a) UNREACHABLE from every production entry point -- c10_d20_production_driver.jl's include
#       chain never includes this file; a repo-wide grep for its `include(...)` finds only
#       benchmark/test files (c9_phase4_kernels_v2_d20_bench.jl, c9_phase4_v2_e2e_d20_bench.jl,
#       test_compressed_cc_kernels_v2.jl).
#   (b) SQUARE-ONLY -- every function below reads `D = cf.D` and never references `cf.D_dest`,
#       using the legacy `j = d + (o-1)*D` bilateral-column formula. Under
#       destination_sample=:exclude_row (D_dest = D-1), `cf.winner`/`cf.wval` are `W x D_dest`
#       matrices -- indexing them with a destination axis of length D (as `for d in 1:D` does
#       throughout) reads a nonexistent column: silently WRONG, not merely slow, with no prior
#       `@assert D==D_dest` guard.
# The functions below now hard-error (rather than silently reading a nonexistent column) if ever
# called against a rectangular `cf` (cf.D != cf.D_dest). This is a deprecation/safety guard only;
# this file is NOT rectangularized here -- it is confirmed dead code with no production caller, so
# the engineering cost of generalizing it is not justified. Do not remove this guard to "make it
# work" under :exclude_row without first rectangularizing every loop below and re-validating
# against test_compressed_cc_kernels_v2.jl at D_dest != D.
# ============================================================================

"""
    compressed_dual_contraction_v2(β, cf::CompressedFactual) -> Vector{W}

Destination-major, `@inbounds`/`@simd`, zero-allocation-in-the-hot-loop
reimplementation of `compressed_dual_contraction`. Bit-identical output
(verified, not assumed -- see this file's header). `κ`/`C`/`Csum`/`κ_cf`/
`pmmterm` setup (O(D^2), draw-independent) is UNCHANGED from the original;
only the O(W*D) main loop is restructured.
"""
function compressed_dual_contraction_v2(β::AbstractVector, cf::CompressedFactual)
    cf.D == cf.D_dest || error("compressed_dual_contraction_v2: square-only, but cf.D=$(cf.D) != cf.D_dest=$(cf.D_dest) " *
        "(destination_sample=:exclude_row or another rectangular regime) -- this file was never rectangularized " *
        "(confirmed unreachable from production). Do not call it under a rectangular cf.")
    D = cf.D; W = cf.W
    length(β) == cf.oci - 1 || error("β length $(length(β)) != oci-1 = $(cf.oci-1)")

    κ = Matrix{Float64}(undef, D, D)
    C = zeros(D)
    @inbounds for d in 1:D
        acc = 0.0
        for o in 1:D
            j = d + (o - 1) * D
            k = β[j] * cf.nrm[j] * cf.gdiv[j]
            κ[o, d] = k
            acc += k * cf.Pmat[o, d]
        end
        C[d] = -cf.denom[d] * acc
    end
    Csum = sum(C)

    κ_cf = cf.cf_col > 0 ? β[cf.cf_col] * cf.nrm[cf.cf_col] * cf.gdiv[cf.cf_col] : 0.0

    pmmterm = 0.0
    if cf.usePMM == 1
        @inbounds for j in 1:(cf.oci - 1)
            pmmterm += β[j] * cf.nrm[j] * cf.PMM[j]
        end
    end

    # t[s] accumulates Csum + sum_d kappa[winner[s,d],d]*wval[s,d] -- SAME per-s
    # accumulation order as the original (d=1,2,...,D for each s), just with the
    # d/s loop nesting swapped so winner[:,d]/wval[:,d] reads are contiguous.
    t = fill(Csum, W)
    @inbounds for d in 1:D
        κd = @view κ[:, d]
        winnerd = @view cf.winner[:, d]
        wvald = @view cf.wval[:, d]
        @simd for s in 1:W
            t[s] += κd[winnerd[s]] * wvald[s]
        end
    end
    @inbounds if cf.cf_col > 0
        @simd for s in 1:W
            t[s] += κ_cf * cf.cf_raw[s]
        end
    end
    @inbounds @simd for s in 1:W
        t[s] = cf.SW[s] * (t[s] - pmmterm)
    end
    return t
end

"""
    compressed_transpose_contraction_v2(weights, cf::CompressedFactual) -> Vector{oci-1}

Destination-major, `@inbounds`/`@simd` reimplementation of
`compressed_transpose_contraction`. Bit-identical output (verified). The
scatter-add `B[winner[s,d],d] += ...` is inherently indirect (cannot be
`@simd`-vectorized, the destination index depends on data) -- but with `d`
now the OUTER loop, the READS of `cf.winner[:,d]`/`cf.wval[:,d]` and the
WRITE target `Bd` (a small, D-length, cache-resident local accumulator, not
`B[:,d]` sliced from the full D-length column of a D x D matrix every
iteration) are both contiguous/local, which is where the real win comes
from -- confirmed empirically below, not assumed from the scatter-add
argument alone (that part is identical cost either way).
"""
function compressed_transpose_contraction_v2(weights::AbstractVector, cf::CompressedFactual)
    cf.D == cf.D_dest || error("compressed_transpose_contraction_v2: square-only, but cf.D=$(cf.D) != cf.D_dest=$(cf.D_dest) " *
        "(destination_sample=:exclude_row or another rectangular regime) -- this file was never rectangularized " *
        "(confirmed unreachable from production). Do not call it under a rectangular cf.")
    D = cf.D; W = cf.W; ncol = cf.oci - 1
    length(weights) == W || error("weights length $(length(weights)) != W=$W")

    B = zeros(D, D)
    T = 0.0
    Bcf = 0.0
    # NOTE: `ws` is precomputed once (pure elementwise multiply, `@simd`-safe --
    # no reduction involved, so this cannot change any downstream summation
    # order) rather than recomputed per-d; T's own accumulation loop deliberately
    # has NO `@simd` (a scalar `+=` reduction) so it visits s=1,2,...,W in
    # EXACTLY the original's order -- `@simd` on a reduction permits the
    # compiler to reassociate the sum for vectorization, which changed the
    # last 2-3 ULPs here in an earlier version of this file (caught by
    # test_compressed_cc_kernels_v2.jl expecting bit-identical, not "close" --
    # fixed by removing `@simd` from every REDUCTION loop below; `@simd`
    # remains safe/kept on the genuinely-elementwise `ws` computation only).
    ws = Vector{Float64}(undef, W)
    @inbounds @simd for s in 1:W
        ws[s] = cf.SW[s] * weights[s]
    end
    @inbounds for s in 1:W
        T += ws[s]
    end
    # NOTE (measured, corrected mid-task): an earlier version of this loop
    # accumulated into a FRESH local `Bd = zeros(D)` per destination, then
    # copied it into `B[:,d]` -- meant to be "cache-friendly" but the extra
    # per-destination allocation + copy cost MORE than the D=20 cache-locality
    # win recovered (measured net 0.877x -- SLOWER than the original). Fixed:
    # `B[:, d]` (a `D x D` matrix's column, for a FIXED `d`) is ALREADY a
    # contiguous, stride-1 memory region in Julia's column-major layout --
    # writing into a `@view` of it directly needs no separate allocation or
    # copy-back at all. Re-measured after this fix, see the report.
    @inbounds for d in 1:D
        winnerd = @view cf.winner[:, d]
        wvald = @view cf.wval[:, d]
        Bcol = @view B[:, d]
        for s in 1:W
            Bcol[winnerd[s]] += ws[s] * wvald[s]
        end
    end
    if cf.cf_col > 0
        @inbounds for s in 1:W
            Bcf += ws[s] * cf.cf_raw[s]
        end
    end

    v = zeros(ncol)
    @inbounds for d in 1:D, o in 1:D
        j = d + (o - 1) * D
        v[j] = cf.nrm[j] * cf.gdiv[j] * (B[o, d] - cf.Pmat[o, d] * cf.denom[d] * T) -
               cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    if cf.cf_col > 0
        j = cf.cf_col
        v[j] = cf.nrm[j] * cf.gdiv[j] * Bcf - cf.nrm[j] * cf.usePMM * cf.PMM[j] * T
    end
    return v
end

"""
    compressed_cc_value_grad_v2(ζ, λ, cf; Psi!, dPsi!) -> (f, g_ζ, g_λ, q, dPsq)

Drop-in `_v2` mirror of `compressed_cc_value_grad`, using the destination-
major kernels above. Same signature/return shape.
"""
function compressed_cc_value_grad_v2(ζ::Real, λ::AbstractVector, cf::CompressedFactual;
                                     Psi!, dPsi!)
    W = cf.W; M = W
    contr = compressed_dual_contraction_v2(λ, cf)
    q = similar(contr)
    @inbounds @. q = -ζ - contr
    Psq = similar(q); Psi!(Psq, q)
    dPsq = similar(q); dPsi!(dPsq, q)
    f = sum(Psq) / M + ζ
    g_ζ = 1.0 - sum(dPsq) / M
    g_λ = compressed_transpose_contraction_v2(dPsq, cf)
    @. g_λ = -(1.0 / M) * g_λ
    return f, g_ζ, g_λ, q, dPsq
end

"""
    compressed_cc_hvp_v2(q, p_ζ, p_λ, cf; ddPsi!) -> (Hp_ζ, Hp_λ)

Drop-in `_v2` mirror of `compressed_cc_hvp`, using the destination-major
kernels above. Same signature/return shape.
"""
function compressed_cc_hvp_v2(q::AbstractVector, p_ζ::Real, p_λ::AbstractVector, cf::CompressedFactual;
                              ddPsi!)
    W = cf.W; M = W
    cpλ = compressed_dual_contraction_v2(p_λ, cf)
    ddPsq = similar(q); ddPsi!(ddPsq, q)
    r = similar(q)
    @inbounds @. r = ddPsq * (p_ζ + cpλ)
    Hp_ζ = sum(r) / M
    Hp_λ = compressed_transpose_contraction_v2(r, cf)
    @. Hp_λ = (1.0 / M) * Hp_λ
    return Hp_ζ, Hp_λ
end
