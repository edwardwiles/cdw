# diag/compressed-hessian-operator-audit-2026-07-25, Phase 5.4: threaded
# winner-pair Hessian, following the SAME draw-chunk-accumulation design this
# codebase already established and validated for the CM bin-table Hessian
# (cm_hessian_architecture_threaded.jl, "Production integration continuation,
# Section 10"): `1:W` split into `nthreads()` contiguous balanced ranges;
# each `for tid in 1:nt` iteration (indexed by the LOOP VARIABLE,
# not `threadid()` -- Julia's scheduler does not guarantee iteration `tid`
# runs on physical thread `tid`) accumulates into its own PRIVATE
# QQ/u/r/scalar buffers; a deterministic serial reduction (fixed 1:nt order,
# reproducible run-to-run) combines them afterward. No atomics in the inner
# loop, per the task brief's explicit preference, matching this codebase's
# established practice.
#
# Also fuses the serial kernel's three separate W-length passes (scalar
# accumulators; u/r; Q'SQ) into ONE per-draw pass, caching each draw's
# `winner[w,1:Ddest]`/`y[w,1:Ddest]` once per draw before the O(Ddest^2/2)
# pair loop -- a cache-locality improvement applicable to the serial kernel
# too, but introduced here first (not yet back-ported/re-benchmarked into
# winner_pair_hessian.jl, which stays untouched and already-validated).
#
# BENCHMARK/CANDIDATE CODE ONLY -- not yet benchmarked (written and
# correctness-checked at D=4 while the real 300s outer A/B runs were in
# flight, deliberately NOT benchmarked for timing until those complete, to
# avoid CPU contention skewing the timed comparison) -- not wired into any
# production driver.
using Base.Threads: nthreads, @threads

"Split `1:W` into `nt` contiguous, size-balanced (differ by at most 1) ranges, deterministic, covering `1:W` exactly once. Local copy of cm_hessian_architecture_threaded.jl's identical helper -- not shared/included from there to avoid a cross-file name collision if both are loaded in the same session; kept intentionally trivial so duplication is not a maintenance risk."
function winner_pair_balanced_ranges_nm(W::Int, nt::Int)
    base_len, rem = divrem(W, nt)
    ranges = Vector{UnitRange{Int}}(undef, nt)
    start = 1
    for t in 1:nt
        len = base_len + (t <= rem ? 1 : 0)
        ranges[t] = start:(start + len - 1)
        start += len
    end
    @assert start - 1 == W
    return ranges
end

"Threaded analogue of WinnerPairHessCtx: adds nthreads() private reduction buffers + a fixed draw-index partition, built once (sized at construction, not per-call) -- reused across every subsequent Hessian call within an inner solve exactly like the serial ctx's own scratch fields."
struct WinnerPairHessCtxThreadedNM
    base::WinnerPairHessCtx
    nt::Int
    ranges::Vector{UnitRange{Int}}
    QQlocal::Vector{Matrix{Float64}}   # nt x (ncolI x ncolI), upper-triangle entries only ever written
    ulocal::Vector{Vector{Float64}}
    rlocal::Vector{Vector{Float64}}
    Ssum_local::Vector{Float64}
    t0_local::Vector{Float64}
    s0_local::Vector{Float64}
end

function build_winner_pair_ctx_threaded_nm(cf::CompressedFactual; nt::Int = nthreads())
    base = build_winner_pair_ctx(cf)
    ncolI = base.ncolI
    ranges = winner_pair_balanced_ranges_nm(base.W, nt)
    QQlocal = [Matrix{Float64}(undef, ncolI, ncolI) for _ in 1:nt]
    ulocal = [Vector{Float64}(undef, ncolI) for _ in 1:nt]
    rlocal = [Vector{Float64}(undef, ncolI) for _ in 1:nt]
    return WinnerPairHessCtxThreadedNM(base, nt, ranges, QQlocal, ulocal, rlocal,
        zeros(nt), zeros(nt), zeros(nt))
end

"""
    winner_pair_hessian_threaded_noMACRO!(h, obj, wctxt::WinnerPairHessCtxThreadedNM)

Threaded winner-pair Hessian, same packing convention/precondition as
`winner_pair_hessian!`. Draw-chunk accumulation into private per-thread
buffers (no atomics), deterministic serial reduction, single fused pass per
draw (scalar accumulators + u/r + Q'SQ pair loop together, unlike the serial
kernel's three separate passes).
"""
function winner_pair_hessian_threaded_noMACRO!(h::AbstractVector, obj, wctxt::WinnerPairHessCtxThreadedNM)
    CS._enter_callback!(obj)
    try
    wctx = wctxt.base
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    M = obj.M
    Ddest = wctx.Ddest; ncolI = wctx.ncolI
    n = 1 + ncolI
    kappa0 = wctx.kappa0; pi_vec = wctx.pi_vec; nu = wctx.nu; y = wctx.y; winner = wctx.winner
    has_cf = wctx.has_cf
    length(h) == n * (n + 1) ÷ 2 || error("winner_pair_hessian_threaded_noMACRO!: length(h)=$(length(h)) != n(n+1)/2 for n=$n")

    nt = wctxt.nt
    for tid in 1:nt
        QQ = wctxt.QQlocal[tid]; u = wctxt.ulocal[tid]; r = wctxt.rlocal[tid]
        fill!(QQ, 0.0); fill!(u, 0.0); fill!(r, 0.0)
        Ssum_t = 0.0; t0_t = 0.0; s0_t = 0.0
        ywbuf = Vector{Float64}(undef, Ddest)   # per-draw cache: y[w,:] gathered once -- thread-private, allocated once outside the w-loop (not per draw)
        obuf = Vector{Int}(undef, Ddest)        # per-draw cache: winner[w,:] gathered once
        @inbounds for w in wctxt.ranges[tid]
            Sw = S[w]; nuw = nu[w]
            Ssum_t += Sw
            snu = Sw * nuw
            t0_t += snu
            snu2 = snu * nuw
            s0_t += snu2

            for slot in 1:Ddest
                o = winner[w, slot]
                obuf[slot] = o
                ywbuf[slot] = y[w, slot]
            end

            if has_cf
                crsw = wctx.cf_raw_scaled[w]
                jcf = ncolI
                u[jcf] += snu * crsw
                r[jcf] += snu2 * crsw
            end

            for slot in 1:Ddest
                o = obuf[slot]; yv = ywbuf[slot]
                j = slot + (o - 1) * Ddest
                u[j] += snu * yv
                r[j] += snu2 * yv
                # diagonal (slot,slot) term
                QQ[j, j] += snu2 * yv * yv
                # cross terms against every OTHER slot' > slot (unordered pairs only, d'>=d)
                for slotp in (slot+1):Ddest
                    op = obuf[slotp]; ypv = ywbuf[slotp]
                    jp = slotp + (op - 1) * Ddest
                    v = snu2 * yv * ypv
                    if j <= jp
                        QQ[j, jp] += v
                    else
                        QQ[jp, j] += v
                    end
                end
                if has_cf
                    jcf = ncolI
                    crsw = wctx.cf_raw_scaled[w]
                    v = snu2 * yv * crsw
                    if j <= jcf
                        QQ[j, jcf] += v
                    else
                        QQ[jcf, j] += v
                    end
                end
            end
            if has_cf
                jcf = ncolI
                crsw = wctx.cf_raw_scaled[w]
                QQ[jcf, jcf] += snu2 * crsw * crsw
            end
        end
        wctxt.Ssum_local[tid] = Ssum_t
        wctxt.t0_local[tid] = t0_t
        wctxt.s0_local[tid] = s0_t
    end

    # ---- deterministic serial reduction (fixed 1:nt order every call) ----
    S_sum = 0.0; t0 = 0.0; s0 = 0.0
    for tid in 1:nt
        S_sum += wctxt.Ssum_local[tid]
        t0 += wctxt.t0_local[tid]
        s0 += wctxt.s0_local[tid]
    end
    u = zeros(ncolI); r = zeros(ncolI)
    for tid in 1:nt
        @inbounds for j in 1:ncolI
            u[j] += wctxt.ulocal[tid][j]
            r[j] += wctxt.rlocal[tid][j]
        end
    end
    QQ = zeros(ncolI, ncolI)
    for tid in 1:nt
        QQl = wctxt.QQlocal[tid]
        @inbounds for i in 1:ncolI
            for j in i:ncolI
                QQ[i, j] += QQl[i, j]
            end
        end
    end

    invM = 1.0 / M
    k = 1
    h[k] = S_sum * invM; k += 1
    @inbounds for j in 1:ncolI
        h[k] = (u[j] - t0 * pi_vec[j]) * invM
        k += 1
    end
    @inbounds for i in 1:ncolI
        for j in i:ncolI
            val = QQ[i, j] - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0 * pi_vec[i] * pi_vec[j]
            h[k] = val * invM
            k += 1
        end
    end
    return h
    finally
        CS._exit_callback!(obj)
    end
end
