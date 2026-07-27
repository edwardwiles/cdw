# ============================================================================
# port/shared-winner-pair-core-hessian-production-2026-07-25
#
# ONE shared exact core-Hessian backend (H_EE, the common Ricardian
# economic/trade-share moment block) for every production restriction family
# (unrestricted, flexible CM, CM+mean/ZC, origin-specific ZC).
#
# This file ports the validated winner-pair kernels from
# diag/compressed-hessian-operator-audit-2026-07-25 (serial: correctness
# oracle, workers=1-equivalent code path; parallel: destination-pair-owned,
# validated port_ready_10_workers) essentially UNCHANGED -- see
# docs/SHARED_WINNER_PAIR_CORE_HESSIAN_PRODUCTION_PORT_2026-07-25.md for the
# exact provenance/diff against the diag branch -- and adds the ONE genuine
# gap that branch's own docs flagged as unbuilt (HVP_OPERATOR_READINESS_MAP,
# HESSIAN_BLOCK_OPERATOR_GAP_ANALYSIS): insertion of the core block into a
# LARGER family-level Hessian (CM/CM+meanZC/origin-ZC all have restriction
# columns beyond the core).
#
# Design decision (see deliverable doc for the alternative considered and
# rejected): every family already materializes a DENSE symmetric scratch
# matrix before packing into KNITRO's row-major upper-triangle format
# (obj.∂∂f_∂∂x for unrestricted/origin-ZC's generic `hessian!`; cctx.Hfull for
# CM/CM+meanZC's Architecture C). So "insertion into the correct locations of
# a larger family Hessian" is handled by simply writing into a `@view` of
# that EXISTING dense scratch at the right offset -- Julia's own view
# indexing IS the local-to-global map, exact and allocation-free, with no
# hand-rolled index table that could drift from the real packing convention.
# `fill_core_hessian_upper!` below is deliberately dense-in/dense-out for
# this reason; each family's OWN existing pack-to-KNITRO-triangle step is
# left completely untouched.
# ============================================================================

isdefined(Main, :CompressedFactual) || include(joinpath(@__DIR__, "compressed_moments.jl"))

using Base.Threads: nthreads

# ============================================================================
# Runtime backend-use counters (2026-07-25 continuation, task §2). The
# startup manifest states which backend was REQUESTED; these counters prove
# which backend actually RAN, per Hessian-callback execution -- mirrors this
# codebase's own existing convention for such counters
# (oracle_fast.jl's InnerCallCounters/_INNER_CALL_COUNTERS,
# compressed_live.jl's COMPRESSED_FALLBACK_COUNT).
# ============================================================================

"Explicit, closed set of dense-fallback reasons -- `:other` is the only catch-all, and its use should be rare/investigated, not routine."
const CORE_HESSIAN_FALLBACK_REASONS = (:compressed_state_unavailable, :tied_winner, :unsupported_layout,
    :debug_reference_requested, :workspace_mismatch, :other)

mutable struct CoreHessianCallCounters
    winner_pair_hessian_calls::Int
    winner_pair_serial_calls::Int
    winner_pair_parallel_calls::Int
    dense_core_fallback_calls::Int
    dense_fallback_reason_counts::Dict{Symbol,Int}
    compressed_core_rebuilds::Int
end

CoreHessianCallCounters() = CoreHessianCallCounters(0, 0, 0, 0, Dict(r => 0 for r in CORE_HESSIAN_FALLBACK_REASONS), 0)

const CORE_HESSIAN_COUNTERS = Ref(CoreHessianCallCounters())

"Reset the core-Hessian backend-use counters -- call at the start of a fresh run/benchmark (same discipline as `reset_compressed_fallback_count!`)."
reset_core_hessian_counters!() = (CORE_HESSIAN_COUNTERS[] = CoreHessianCallCounters())

"""
    resolve_core_hessian_workers_default() :: Int

2026-07-25 final-gate continuation (task §3, worker-count policy): the winner-pair parallel
kernel's own 20-thread sweep (`docs/WINNER_PAIR_20_THREAD_WORKER_SELECTION_2026-07-25.md`) found
`workers=20` genuinely (not tied-within-noise) 13-20% faster than `workers=10` at BOTH a
near-`delta=1` feasible point and a harder point, once 20 real Julia threads are actually
available -- so a hard-coded `10` is leaving real throughput on the table whenever
`JULIA_NUM_THREADS>=20`, and is retained ONLY as the safe fallback below that. `workers` can never
legally exceed `Threads.nthreads()` (see the parallel kernel's own `max_workers` bound), so this
also protects any environment with fewer available threads from a default that would silently
degrade or error.
"""
function resolve_core_hessian_workers_default()
    n = nthreads()
    n >= 20 && return 20
    n >= 10 && return 10
    return max(1, n)
end

"Human-readable label for which branch of `resolve_core_hessian_workers_default()`'s piecewise rule is currently active -- printed in every public startup manifest (task §3) alongside the resolved worker count itself, so a manifest reader sees WHY that count was chosen, not just the number."
function core_hessian_worker_policy_label()
    n = nthreads()
    n >= 20 && return :ge20_threads_use_20
    n >= 10 && return :ge10_lt20_threads_use_10
    return :lt10_threads_use_available
end

"""
    CM_CORE_HESSIAN_BACKEND_DEFAULT / ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT

2026-07-25 continuation (task §6, matched outer A/B): CM/CM+meanZC and origin-ZC build their
`CMBinHessCtx`/`OriginZCCoreHessCtx` INSIDE their own checkpointed driver
(`run_cm_upper_checkpointed`/`run_originzc_upper_checkpointed`), with no `core_hessian_backend`
kwarg threaded through the driver's own public signature. Rather than add and thread a new kwarg
through every driver (a larger, riskier interface change than this task needs), these two central
Refs are what `build_cm_bin_ctx`/`build_cm_meanzc_bin_ctx`/`build_originzc_core_hess_ctx` default
to -- mirroring `UNRESTRICTED_CORE_HESSIAN_BACKEND` (`compressed_live.jl`) exactly, so an external
benchmark/gate script can flip ONE global before calling the real public driver, same discipline
for all four families.
"""
const CM_CORE_HESSIAN_BACKEND_DEFAULT = Ref{Symbol}(:exact_winner_pair_parallel)
const CM_CORE_HESSIAN_WORKERS_DEFAULT = Ref{Int}(resolve_core_hessian_workers_default())
const CM_CORE_HESSIAN_STORAGE_DEFAULT = Ref{Symbol}(:full_stride)

"""
    CM_CROSS_HESSIAN_BACKEND_DEFAULT

Winner-aware H_ER phase (2026-07-27): which backend `hessian_cm_structured!`/`_v2!`
(cm_hessian_architectures.jl / cm_hessian_threaded.jl) use for the economic x CM-restriction
cross block (`H_EC`). `:winner_bin` (winner_pair_cross_hessian.jl, `H_ER = Q'SR - pi*(nu'SR)`,
PRODUCTION DEFAULT as of this phase) reuses the already-validated `WinnerPairHessCtx`/
`CoreExactHessianWorkspace` H_EE precompute the shared exact-winner-pair backend already builds
for this same context -- no dense read of `E = H[:, 2:1+NCORE]` at all in the fast path. Flipped
from `:dense_reference` after D=4 (8 configs x L in {10,20,50} x 2 extra perturbed points, both
serial `hessian_cm_structured!` and threaded-production `hessian_cm_structured_v2!`) and real
D=20/W=80,000/L=50 (both contrasts, calib + near-delta=1 perturbed) gates ALL PASSED to machine
precision (max|Delta H|~1e-13 to 1e-16 against the dense reference), with the complete inner solve
status/dual point matching too, and a genuine speedup (real D=20: ~7x serial cold, ~2x
threaded-warm) -- see FLEXIBLE_CM_WINNER_BIN_HER_RELEASE_2026-07-27.md. Only available when
`cctx.core_hessian_backend !== :dense_reference` AND `cctx.ncore_core == cctx.NCORE` (no
CM+mean/pair-ZC widening -- that layout is out of this backend's validated scope, see Section 4 of
the winner-aware-H_ER task); `hessian_cm_structured!` checks both and falls back to
`:dense_reference` (not silently, `record_dense_cross_hessian_call!`) whenever either fails --
e.g. CM+meanZC always falls back here automatically since its `ncore_core < NCORE`.
`:dense_reference` remains available as an explicit, named, non-default backend.
"""
const CM_CROSS_HESSIAN_BACKEND_DEFAULT = Ref{Symbol}(:winner_bin)
const ORIGINZC_CORE_HESSIAN_BACKEND_DEFAULT = Ref{Symbol}(:exact_winner_pair_parallel)
const ORIGINZC_CORE_HESSIAN_WORKERS_DEFAULT = Ref{Int}(resolve_core_hessian_workers_default())
const ORIGINZC_CORE_HESSIAN_STORAGE_DEFAULT = Ref{Symbol}(:full_stride)

"""
Remediation task Phase B1 (production-audit continuation, 2026-07-26); flipped to `:cm_lookup`
by the Phase 5.5 allocation-fix remediation (2026-07-26): which inner FG (forward/backward)
callback the CM family's KNITRO inner dual solve registers.
:dense_reference (legacy/reference -- byte-identical to every pre-Phase-B1 production run) |
:cm_lookup (DEFAULT -- validated O(W*(D-1)) lookup kernel, cm_lookup_kernels.jl/
cm_lookup_production.jl -- plain flexible CM ONLY, see cm_lookup_production.jl's own header for
why common_frechet/meanzc are out of scope for this specific kernel). Global default + per-
CMBinHessCtx override, same discipline as CM_CORE_HESSIAN_BACKEND_DEFAULT above -- Hessian
backend selection is completely independent of this.

Flip rationale (task §5.6 criteria, all real D=20/W=80,000/L=50 unless noted): D=4 AND D=20
correctness (test_phaseB1_cmlookup_production_correctness.jl) ALL PASS, both contrasts, calib
+ perturbed points, KNITRO-solved quantities agreeing to ~1e-13 or tighter. Complete inner
solve FASTER at every tested thread count (test_phaseB1_performance_gate.jl, workers=
1/4/8/10/20): 1.108x-1.616x vs :dense_reference (previously, before the Phase 5.5 allocation
fix, only 1.096x at the single thread count then tested). Allocation: previously :cm_lookup
allocated ~12.6% MORE than :dense_reference per complete inner solve (the reason it shipped
`AVAILABLE_BUT_NOT_DEFAULT`) -- the Phase 5.5 fix (persistent CMLookupState scratch buffers,
cached across inner solves on `cctx.cmlookup_st`, zero per-FG-callback allocation) brought
median allocation to EXACT PARITY with :dense_reference (ratio 1.000x) at every thread count,
i.e. the lookup kernel itself is now allocation-free relative to dense -- the measured bytes are
common `archC_verified_state`/KNITRO overhead shared by both backends. No stability regression
(identical nStatus, Delta_dual agreement ~1e-17-1e-18). full_G_materializations=0 by
construction (the lookup kernel never touches obj.H's CM columns). See
docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md for full results.
"""
const CM_INNER_FG_BACKEND_DEFAULT = Ref{Symbol}(:cm_lookup)

"""
    CM_FRECHET_INNER_FG_BACKEND_DEFAULT

Phase 5.2 remediation (2026-07-26): common-Frechet analogue of `CM_INNER_FG_BACKEND_DEFAULT`.
`:dense_reference` (default, unchanged) | `:cm_frechet_lookup` (new matrix-free CM+level operator,
`cm_frechet_lookup_kernels.jl`/`cm_frechet_lookup_production.jl` -- only reachable when
`cm_hessian_backend=:structured`, see `build_cm_frechet_production_context`'s own check). Kept
`:dense_reference` until D=4/D=20 correctness and performance gates pass (mirrors
`CM_INNER_FG_BACKEND_DEFAULT`'s own history: available-but-not-default until validated, THEN
flipped -- see docs/RESTRICTED_OPERATOR_FG_PRODUCTION_PORT_2026-07-26.md for the flip criteria and
this family's own gate results once run).
"""
const CM_FRECHET_INNER_FG_BACKEND_DEFAULT = Ref{Symbol}(:dense_reference)

"""
    ORIGINZC_FG_BACKEND_DEFAULT

port/finish-operator-stack-no-dense-G-and-CM-basis-diagnosis-2026-07-26, Phase A item 6 gate.
`:dense_reference` | `:operator` (shared `economic_forward!`/`economic_transpose!` + `G=[E|Z]`
`ZCRestrictionOperator`, `cm_originzc_lookup_kernels.jl`). Flipped to `:operator` after real
D=4 + D=20/W=80,000 complete-inner-solve A/B (`bench_originzc_meanzc_operator_vs_dense.jl`):
correctness ALL PASS (Delta_dual agreement 8.3e-16 at D=20), speedup 1.001x (within-5% criterion
met), median allocation 13,239.9MB (dense) vs 13,242.4MB (operator) -- ratio 1.0002x, i.e. AT
PARITY, not materially reduced, for the SAME reason `CM_INNER_FG_BACKEND_DEFAULT`'s own flip
docstring above already establishes as this codebase's precedent: `archOZ_verified_state`'s shared
post-solve `CS.select_G_from_H`/KKT-residual step and KNITRO's own C-side per-iterate solve
overhead dominate total per-call bytes at W=80,000 scale, common to both backends -- the operator's
own FG-callback-level saving (previously measured in isolation at 20,081x for the shared economic
block) is real but a rounding error against that shared total. No stability regression (identical
nStatus, machine-precision Delta_dual agreement). Flipping removes the last dense `obj.H`
BLAS.gemv! from this family's ordinary FG hot path, matching this task's Phase A architecture
requirement (`G=[E|Z]` composed of shared operators, no per-family dense-column re-derivation).
"""
const ORIGINZC_FG_BACKEND_DEFAULT = Ref{Symbol}(:operator)

"""
    CM_MEANZC_INNER_FG_BACKEND_DEFAULT

Analogous to `ORIGINZC_FG_BACKEND_DEFAULT` for CM+ZC's `G=[E|C|Z]` operator FG
(`cm_meanzc_lookup_kernels.jl`). Flipped to `:operator` on the same real D=4+D=20/W=80,000 gate
evidence: correctness ALL PASS (Delta_dual agreement 3.7e-15), speedup 1.084x, allocation at parity
(ratio 1.0002x, same shared-overhead explanation as `ORIGINZC_FG_BACKEND_DEFAULT`).
"""
const CM_MEANZC_INNER_FG_BACKEND_DEFAULT = Ref{Symbol}(:operator)

"""
    record_core_hessian_call!(backend; fallback_reason=nothing)

Record ONE Hessian-callback EXECUTION at the given backend -- called from
inside `fill_core_hessian_upper!` (CM/CM+meanZC/origin-ZC) and directly from
unrestricted's `_callbackEvalH_inner_compressed!` (which bypasses
`fill_core_hessian_upper!` for its packed-direct fast path), so every
production Hessian callback that touches the shared core goes through this
one function, not four independent copies of the same counter logic.
`fallback_reason` is required for any non-winner-pair backend value; an
unrecognized or missing reason is recorded as `:other` (never silently
dropped -- `:other`'s own count is a signal something needs a real reason
added to `CORE_HESSIAN_FALLBACK_REASONS`).
"""
function record_core_hessian_call!(backend::Symbol; fallback_reason::Union{Nothing,Symbol} = nothing)
    c = CORE_HESSIAN_COUNTERS[]
    if backend === :exact_winner_pair_serial
        c.winner_pair_hessian_calls += 1
        c.winner_pair_serial_calls += 1
    elseif backend === :exact_winner_pair_parallel
        c.winner_pair_hessian_calls += 1
        c.winner_pair_parallel_calls += 1
    else
        reason = fallback_reason === nothing ? :other : fallback_reason
        reason in CORE_HESSIAN_FALLBACK_REASONS || (reason = :other)
        c.dense_core_fallback_calls += 1
        c.dense_fallback_reason_counts[reason] = get(c.dense_fallback_reason_counts, reason, 0) + 1
    end
    return nothing
end

"Record that a `CoreExactHessianWorkspace` was (re)built from a FRESH `CompressedFactual` object (a new outer point) -- not a call that reused an already-built workspace because the `cf` object identity was unchanged."
record_compressed_core_rebuild!() = (CORE_HESSIAN_COUNTERS[].compressed_core_rebuilds += 1; nothing)

"Human-readable print of the counters, flushed immediately -- same discipline as `print_production_backend_manifest`. Only prints fallback-reason lines with a nonzero count."
function print_core_hessian_counters(c::CoreHessianCallCounters = CORE_HESSIAN_COUNTERS[])
    println("[core-hessian-counters] winner_pair_hessian_calls=", c.winner_pair_hessian_calls)
    println("[core-hessian-counters]   winner_pair_serial_calls=", c.winner_pair_serial_calls)
    println("[core-hessian-counters]   winner_pair_parallel_calls=", c.winner_pair_parallel_calls)
    println("[core-hessian-counters] dense_core_fallback_calls=", c.dense_core_fallback_calls)
    for r in CORE_HESSIAN_FALLBACK_REASONS
        n = get(c.dense_fallback_reason_counts, r, 0)
        n > 0 && println("[core-hessian-counters]   fallback_reason[", r, "]=", n)
    end
    println("[core-hessian-counters] compressed_core_rebuilds=", c.compressed_core_rebuilds)
    flush(stdout)
    return nothing
end

# ----------------------------------------------------------------------------
# Serial kernel (correctness oracle / :exact_winner_pair_serial backend).
# Ported verbatim from diag/compressed-hessian-operator-audit-2026-07-25
# (winner_pair_hessian.jl, commit 3fceb71) -- CompressedFactual's field set is
# byte-identical between that branch's base (b7435ee) and this port's base
# (production tip 39b89c5; `git diff b7435ee..HEAD -- compressed_moments.jl`
# is empty), so no adaptation was needed beyond this file's own header.
# ----------------------------------------------------------------------------

"""
    WinnerPairHessCtx

Precomputed, theta-fixed scratch for the winner-pair Hessian. Built once per
outer point (same lifetime as `CompressedFactual`), reused across every
Hessian-callback call within one inner KNITRO solve (only `obj.arg2`, i.e.
S, changes call to call).
"""
struct WinnerPairHessCtx
    D::Int
    Ddest::Int
    W::Int
    ncolI::Int              # = cf.oci - 1 (bilateral + optional cf column)
    has_cf::Bool
    kappa0::Vector{Float64}      # length ncolI
    pi_vec::Vector{Float64}      # length ncolI
    nu::Vector{Float64}          # = SW, length W (REAL sampling weights, not assumed uniform)
    y::Matrix{Float64}           # W x Ddest, kappa0-scaled winner value
    winner::Matrix{Int}          # W x Ddest (alias of cf.winner)
    cf_raw_scaled::Vector{Float64}  # kappa0[cf_col]*cf_raw[w], length W (empty if !has_cf)
    Snu_buf::Vector{Float64}
    Snu2_buf::Vector{Float64}
    u_buf::Vector{Float64}
    r_buf::Vector{Float64}
    QQ_buf::Matrix{Float64}
end

"""
    build_winner_pair_ctx(cf::CompressedFactual) -> WinnerPairHessCtx

O(W*Ddest) construction (dominated by computing `y`), theta-fixed -- build
once per outer point/inner solve.
"""
function build_winner_pair_ctx(cf::CompressedFactual)
    D = cf.D; Ddest = cf.D_dest; W = cf.W; ncolI = cf.oci - 1
    has_cf = cf.cf_col > 0

    kappa0 = Vector{Float64}(undef, ncolI)
    pi_vec = Vector{Float64}(undef, ncolI)
    @inbounds for slot in 1:Ddest, o in 1:D
        j = slot + (o - 1) * Ddest
        k0 = cf.nrm[j] * cf.gdiv[j]
        kappa0[j] = k0
        pi_vec[j] = k0 * cf.Pmat[o, slot] * cf.denom[slot] + cf.nrm[j] * cf.usePMM * cf.PMM[j]
    end

    y = Matrix{Float64}(undef, W, Ddest)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = cf.winner[w, slot]
            j = slot + (o - 1) * Ddest
            y[w, slot] = kappa0[j] * cf.wval[w, slot]
        end
    end

    cf_raw_scaled = Float64[]
    if has_cf
        jcf = cf.cf_col
        k0cf = cf.nrm[jcf] * cf.gdiv[jcf]
        kappa0[jcf] = k0cf
        pi_vec[jcf] = cf.nrm[jcf] * cf.usePMM * cf.PMM[jcf]
        cf_raw_scaled = k0cf .* cf.cf_raw
    end

    return WinnerPairHessCtx(D, Ddest, W, ncolI, has_cf, kappa0, pi_vec, copy(cf.SW), y, cf.winner, cf_raw_scaled,
        Vector{Float64}(undef, W), Vector{Float64}(undef, W),
        Vector{Float64}(undef, ncolI), Vector{Float64}(undef, ncolI),
        Matrix{Float64}(undef, ncolI, ncolI))
end

"""
    winner_pair_hessian!(h::AbstractVector, obj, wctx::WinnerPairHessCtx)

Fills the packed row-major upper-triangular Hessian `h` (length
n*(n+1)/2, n = 1 + wctx.ncolI), IDENTICAL packing convention to
`hessian!`/`hessian_cm_structured!`. O(W*Ddest^2 + W*Ddest) total,
O(Ddest^2) extra memory -- never allocates or touches a W x m array.
"""
function winner_pair_hessian!(h::AbstractVector, obj, wctx::WinnerPairHessCtx)
    CS._enter_callback!(obj)
    try
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2
    M = obj.M
    Ddest = wctx.Ddest; W = wctx.W
    ncolI = wctx.ncolI
    n = 1 + ncolI
    kappa0 = wctx.kappa0; pi_vec = wctx.pi_vec; nu = wctx.nu; y = wctx.y

    length(h) == n * (n + 1) ÷ 2 || error("winner_pair_hessian!: length(h)=$(length(h)) != n(n+1)/2 for n=$n")

    S_sum = 0.0; t0 = 0.0; s0 = 0.0
    Snu = wctx.Snu_buf
    Snu2 = wctx.Snu2_buf
    @inbounds for w in 1:W
        Sw = S[w]; nuw = nu[w]
        S_sum += Sw
        snu = Sw * nuw
        Snu[w] = snu
        t0 += snu
        snu2 = snu * nuw
        Snu2[w] = snu2
        s0 += snu2
    end

    u = wctx.u_buf; r = wctx.r_buf
    fill!(u, 0.0); fill!(r, 0.0)
    @inbounds for slot in 1:Ddest
        for w in 1:W
            o = wctx.winner[w, slot]
            j = slot + (o - 1) * Ddest
            yv = y[w, slot]
            u[j] += Snu[w] * yv
            r[j] += Snu2[w] * yv
        end
    end
    if wctx.has_cf
        jcf = ncolI
        crs = wctx.cf_raw_scaled
        uu = 0.0; rr = 0.0
        @inbounds for w in 1:W
            uu += Snu[w] * crs[w]
            rr += Snu2[w] * crs[w]
        end
        u[jcf] = uu; r[jcf] = rr
    end

    QQ = wctx.QQ_buf
    fill!(QQ, 0.0)
    @inbounds for slot in 1:Ddest
        for slotp in slot:Ddest
            if slot == slotp
                for w in 1:W
                    o = wctx.winner[w, slot]
                    j = slot + (o - 1) * Ddest
                    QQ[j, j] += Snu2[w] * y[w, slot] * y[w, slot]
                end
            else
                for w in 1:W
                    o = wctx.winner[w, slot]; op = wctx.winner[w, slotp]
                    j = slot + (o - 1) * Ddest
                    jp = slotp + (op - 1) * Ddest
                    v = Snu2[w] * y[w, slot] * y[w, slotp]
                    if j <= jp
                        QQ[j, jp] += v
                    else
                        QQ[jp, j] += v
                    end
                end
            end
        end
    end
    if wctx.has_cf
        jcf = ncolI
        crs = wctx.cf_raw_scaled
        qcc = 0.0
        @inbounds for w in 1:W
            qcc += Snu2[w] * crs[w] * crs[w]
        end
        QQ[jcf, jcf] += qcc
        @inbounds for slot in 1:Ddest
            for w in 1:W
                o = wctx.winner[w, slot]
                j = slot + (o - 1) * Ddest
                v = Snu2[w] * y[w, slot] * crs[w]
                if j <= jcf
                    QQ[j, jcf] += v
                else
                    QQ[jcf, j] += v
                end
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
            qq = i == j ? QQ[i, i] : (i <= j ? QQ[i, j] : QQ[j, i])
            val = qq - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0 * pi_vec[i] * pi_vec[j]
            h[k] = val * invM
            k += 1
        end
    end
    return h
    finally
        CS._exit_callback!(obj)
    end
end

# ----------------------------------------------------------------------------
# Parallel kernel (destination-pair-ownership, validated port_ready_10_workers
# in diag/compressed-hessian-operator-audit-2026-07-25). Ported verbatim from
# winner_pair_hessian_parallel.jl (commit 3fceb71). The draw-partitioned
# `_threaded`/`_threaded_nomacro` variants are DELIBERATELY NOT ported -- that
# branch's own docs record an unresolved correctness bug once nthreads()>1 for
# that design; only the pair-ownership design (proven bug-free at D=4 and
# D=20, including non-last omitted destination) is brought into production.
# ----------------------------------------------------------------------------

"One destination-group: either a bilateral destination slot (D possible winning origins) or the counterfactual column (1 'origin', no winner structure)."
struct WPGroup
    is_cf::Bool
    slot::Int
    cols::Vector{Int}
end

"Balanced partition of `1:N` into `nchunks` contiguous, size-differ-by-at-most-1 ranges."
function wp_balanced_ranges(N::Int, nchunks::Int)
    base_len, rem = divrem(N, nchunks)
    ranges = Vector{UnitRange{Int}}(undef, nchunks)
    start = 1
    for t in 1:nchunks
        len = base_len + (t <= rem ? 1 : 0)
        ranges[t] = start:(start + len - 1)
        start += len
    end
    return ranges
end

"""
    WinnerPairParallelWorkspace

Persistent, theta-fixed workspace for the pair-ownership parallel kernel.
Built once per outer point, reused across every Hessian-callback call within
one inner solve.
"""
struct WinnerPairParallelWorkspace
    base::WinnerPairHessCtx
    ngroups::Int
    groups::Vector{WPGroup}
    pairs::Vector{Tuple{Int,Int}}
    QQ::Matrix{Float64}
    u::Vector{Float64}
    r::Vector{Float64}
    PackIdx::Matrix{Int}
    Snu::Vector{Float64}
    Snu2::Vector{Float64}
    draw_ranges_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    group_chunks_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    pair_chunks_by_workers::Dict{Int,Vector{UnitRange{Int}}}
    Ssum_local::Vector{Float64}
    t0_local::Vector{Float64}
    s0_local::Vector{Float64}
    tasks::Vector{Task}
    max_workers::Int
end

"""
    build_winner_pair_parallel_workspace(cf::CompressedFactual; worker_counts, max_workers) -> WinnerPairParallelWorkspace

`worker_counts` (default the standard scaling sweep, including the validated
production default 10) is precomputed ONCE so `hessian_core_winner_pair!`
never builds a partition inside a timed call.
"""
function build_winner_pair_parallel_workspace(cf::CompressedFactual;
        worker_counts::Vector{Int} = [1, 2, 4, 8, 10, 19, 20],
        max_workers::Int = max(nthreads(), maximum(worker_counts)))
    base = build_winner_pair_ctx(cf)
    D = base.D; Ddest = base.Ddest; ncolI = base.ncolI; has_cf = base.has_cf; W = base.W
    ngroups = Ddest + (has_cf ? 1 : 0)
    groups = Vector{WPGroup}(undef, ngroups)
    for slot in 1:Ddest
        cols = [slot + (o - 1) * Ddest for o in 1:D]
        groups[slot] = WPGroup(false, slot, cols)
    end
    if has_cf
        groups[Ddest + 1] = WPGroup(true, 0, [ncolI])
    end
    pairs = Tuple{Int,Int}[]
    for g in 1:ngroups, gp in g:ngroups
        push!(pairs, (g, gp))
    end

    n = 1 + ncolI
    PackIdx = fill(0, n, n)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            PackIdx[i, j] = k
            k += 1
        end
    end

    draw_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    group_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    pair_by_w = Dict{Int,Vector{UnitRange{Int}}}()
    for wc in unique(vcat(worker_counts, 1))
        draw_by_w[wc] = wp_balanced_ranges(W, wc)
        group_by_w[wc] = wp_balanced_ranges(ngroups, wc)
        pair_by_w[wc] = wp_balanced_ranges(length(pairs), wc)
    end

    return WinnerPairParallelWorkspace(base, ngroups, groups, pairs,
        zeros(ncolI, ncolI), zeros(ncolI), zeros(ncolI), PackIdx,
        Vector{Float64}(undef, W), Vector{Float64}(undef, W),
        draw_by_w, group_by_w, pair_by_w,
        zeros(max_workers), zeros(max_workers), zeros(max_workers),
        Vector{Task}(undef, max_workers), max_workers)
end

@inline function wp_accumulate_pair!(QQ::Matrix{Float64}, g1::WPGroup, g2::WPGroup, Snu2::Vector{Float64},
        winner::Matrix{Int}, y::Matrix{Float64}, cf_raw_scaled::Vector{Float64}, W::Int, Ddest::Int)
    if !g1.is_cf && !g2.is_cf && g1.slot == g2.slot
        slot = g1.slot
        @inbounds for w in 1:W
            o = winner[w, slot]
            j = slot + (o - 1) * Ddest
            QQ[j, j] += Snu2[w] * y[w, slot] * y[w, slot]
        end
    elseif !g1.is_cf && !g2.is_cf
        s1 = g1.slot; s2 = g2.slot
        @inbounds for w in 1:W
            o1 = winner[w, s1]; o2 = winner[w, s2]
            j1 = s1 + (o1 - 1) * Ddest
            j2 = s2 + (o2 - 1) * Ddest
            v = Snu2[w] * y[w, s1] * y[w, s2]
            if j1 <= j2
                QQ[j1, j2] += v
            else
                QQ[j2, j1] += v
            end
        end
    elseif !g1.is_cf && g2.is_cf
        s1 = g1.slot; jcf = g2.cols[1]
        @inbounds for w in 1:W
            o1 = winner[w, s1]
            j1 = s1 + (o1 - 1) * Ddest
            QQ[j1, jcf] += Snu2[w] * y[w, s1] * cf_raw_scaled[w]
        end
    else
        jcf = g1.cols[1]
        @inbounds for w in 1:W
            QQ[jcf, jcf] += Snu2[w] * cf_raw_scaled[w] * cf_raw_scaled[w]
        end
    end
    return nothing
end

@inline function wp_accumulate_group!(u::Vector{Float64}, r::Vector{Float64}, g::WPGroup,
        Snu::Vector{Float64}, Snu2::Vector{Float64}, winner::Matrix{Int}, y::Matrix{Float64},
        cf_raw_scaled::Vector{Float64}, W::Int, Ddest::Int)
    if g.is_cf
        jcf = g.cols[1]
        acc_u = 0.0; acc_r = 0.0
        @inbounds for w in 1:W
            acc_u += Snu[w] * cf_raw_scaled[w]
            acc_r += Snu2[w] * cf_raw_scaled[w]
        end
        u[jcf] += acc_u; r[jcf] += acc_r
    else
        slot = g.slot
        @inbounds for w in 1:W
            o = winner[w, slot]
            j = slot + (o - 1) * Ddest
            yv = y[w, slot]
            u[j] += Snu[w] * yv
            r[j] += Snu2[w] * yv
        end
    end
    return nothing
end

"""
    hessian_core_winner_pair!(hess_packed, curvature_weights, obj, workspace; workers=1, storage=:full_stride)

Shared serial/parallel interface. `workers=1` runs the SAME code path as
workers>1 (2 spawn rounds of 1 task each), so there is no separate "fast
path" that could silently drift from the parallel path.
"""
function hessian_core_winner_pair!(hess_packed::AbstractVector, curvature_weights, obj, workspace::WinnerPairParallelWorkspace;
        workers::Int = 1, storage::Symbol = :full_stride)
    storage in (:full_stride, :direct_packed) || error("hessian_core_winner_pair!: storage must be :full_stride or :direct_packed, got :$storage")
    haskey(workspace.draw_ranges_by_workers, workers) || error("hessian_core_winner_pair!: workers=$workers not in workspace's precomputed worker_counts")
    CS._enter_callback!(obj)
    try
    base = workspace.base
    ddPsi! = obj.ddPsi!
    ddPsi!(obj.arg2, obj.arg0)
    S = obj.arg2; M = obj.M
    Ddest = base.Ddest; ncolI = base.ncolI; W = base.W
    nu = base.nu; y = base.y; winner = base.winner; cf_raw_scaled = base.cf_raw_scaled
    pi_vec = base.pi_vec
    n = 1 + ncolI
    length(hess_packed) == n * (n + 1) ÷ 2 || error("hessian_core_winner_pair!: length mismatch")

    QQ = workspace.QQ; u = workspace.u; r = workspace.r
    Snu = workspace.Snu; Snu2 = workspace.Snu2
    fill!(QQ, 0.0); fill!(u, 0.0); fill!(r, 0.0)

    draw_ranges = workspace.draw_ranges_by_workers[workers]
    group_chunks = workspace.group_chunks_by_workers[workers]
    pair_chunks = workspace.pair_chunks_by_workers[workers]
    tasks = workspace.tasks

    for wk in 1:workers
        rng = draw_ranges[wk]
        tasks[wk] = Threads.@spawn begin
            Ss = 0.0; t0 = 0.0; s0 = 0.0
            @inbounds for w in rng
                Sw = S[w]; nuw = nu[w]
                snu = Sw * nuw
                Snu[w] = snu
                snu2 = snu * nuw
                Snu2[w] = snu2
                Ss += Sw; t0 += snu; s0 += snu2
            end
            (Ss, t0, s0)
        end
    end
    S_sum = 0.0; t0_tot = 0.0; s0_tot = 0.0
    for wk in 1:workers
        Ss, t0v, s0v = fetch(tasks[wk])
        S_sum += Ss; t0_tot += t0v; s0_tot += s0v
    end

    for wk in 1:workers
        gr = group_chunks[wk]; pr = pair_chunks[wk]
        tasks[wk] = Threads.@spawn begin
            for gi in gr
                wp_accumulate_group!(u, r, workspace.groups[gi], Snu, Snu2, winner, y, cf_raw_scaled, W, Ddest)
            end
            for pidx in pr
                (g, gp) = workspace.pairs[pidx]
                wp_accumulate_pair!(QQ, workspace.groups[g], workspace.groups[gp], Snu2, winner, y, cf_raw_scaled, W, Ddest)
            end
            nothing
        end
    end
    for wk in 1:workers
        fetch(tasks[wk])
    end

    invM = 1.0 / M
    if storage == :full_stride
        k = 1
        hess_packed[k] = S_sum * invM; k += 1
        @inbounds for j in 1:ncolI
            hess_packed[k] = (u[j] - t0_tot * pi_vec[j]) * invM
            k += 1
        end
        @inbounds for i in 1:ncolI
            for j in i:ncolI
                val = QQ[i, j] - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0_tot * pi_vec[i] * pi_vec[j]
                hess_packed[k] = val * invM
                k += 1
            end
        end
    else
        PackIdx = workspace.PackIdx
        hess_packed[PackIdx[1, 1]] = S_sum * invM
        @inbounds for j in 1:ncolI
            hess_packed[PackIdx[1, 1 + j]] = (u[j] - t0_tot * pi_vec[j]) * invM
        end
        @inbounds for i in 1:ncolI
            for j in i:ncolI
                val = QQ[i, j] - r[i] * pi_vec[j] - pi_vec[i] * r[j] + s0_tot * pi_vec[i] * pi_vec[j]
                hess_packed[PackIdx[1 + i, 1 + j]] = val * invM
            end
        end
    end
    return hess_packed
    finally
        CS._exit_callback!(obj)
    end
end

# ============================================================================
# Section 2 of the task: the ONE shared production abstraction every family
# calls -- CoreExactHessianWorkspace + fill_core_hessian_upper!.
# ============================================================================

const CORE_HESSIAN_BACKENDS = (:exact_winner_pair_parallel, :exact_winner_pair_serial, :dense_reference)

"""
    CoreExactHessianWorkspace

Wraps BOTH the serial ctx and the parallel workspace (they share the same
`base::WinnerPairHessCtx`-shaped precompute -- `parallel_ws.base` IS a
`WinnerPairHessCtx`, so there is exactly one copy of the theta-fixed
precompute, not two), plus a persistent packed scratch buffer used as the
common local output format before unpacking into whatever dense view the
caller supplies. Built once per outer point (same cadence as
`CompressedFactual`/`CMBinHessCtx`/`WinnerPairHessCtx` itself).
"""
struct CoreExactHessianWorkspace
    parallel_ws::WinnerPairParallelWorkspace
    packed_scratch::Vector{Float64}
    ncolI::Int
end

function build_core_exact_hessian_workspace(cf::CompressedFactual;
        worker_counts::Vector{Int} = [1, 2, 4, 8, 10, 19, 20],
        max_workers::Int = max(nthreads(), maximum(worker_counts)))
    parallel_ws = build_winner_pair_parallel_workspace(cf; worker_counts = worker_counts, max_workers = max_workers)
    ncolI = parallel_ws.base.ncolI
    n = 1 + ncolI
    return CoreExactHessianWorkspace(parallel_ws, Vector{Float64}(undef, n * (n + 1) ÷ 2), ncolI)
end

"Serial ctx accessor (workers=1 through the SAME parallel code path is preferred for the production default; this returns the base ctx for the standalone :exact_winner_pair_serial oracle/fallback backend, which uses the dedicated non-threaded `winner_pair_hessian!` kernel, not `hessian_core_winner_pair!(...; workers=1)`)."
serial_ctx(ws::CoreExactHessianWorkspace) = ws.parallel_ws.base

"""
    _dense_reference_core_hessian!(packed, obj, n)

Generic dense-BLAS reference/debug backend (`:dense_reference`): recomputes
the SAME (1+ncolI)x(1+ncolI) core block via the ordinary weighted Gram matrix
`(1/M) Ĝ'SĜ`, `Ĝ = obj.H[:, 2:1+n]` (n columns: `H`'s own column 2 is
ALREADY the literal zeta ones-column -- populated by the caller, e.g.
`inner_loop_internal_archgeneric`'s `obj.H[:, 2] .= 1.0` -- NOT manufactured
here; columns 3:1+n are the n-1 core bilateral/cf columns), exactly the
pattern every family used before this port
(`cc_algo/PsiObjectiveBundle.jl::hessian!`'s `H_copy[:, 2:1+outer_constr_index]
.= H[:, 2:1+outer_constr_index]` with `outer_constr_index == n` here --
`cm_hessian_architectures.jl::hessian_cm_structured!`'s H_EE carve-out reads
the analogous `E = H[:, 2:1+NCORE]` view). Preserved as a NAMED,
explicitly-opt-in fallback per task §5 ("acceptable to preserve dense BLAS as
a named fallback... must not remain the silent default"), and as the
anti-regression comparison target.
"""
function _dense_reference_core_hessian!(packed::AbstractVector, obj, n::Int)
    CS._enter_callback!(obj)
    try
    @unpack H, H_copy, M, arg0, arg2, ddPsi! = obj
    ddPsi!(arg2, arg0)
    @views H_copy[:, 2:1+n] .= H[:, 2:1+n]
    @views H_copy[:, 2:1+n] .*= .√arg2
    Hd = Matrix{Float64}(undef, n, n)
    BLAS.gemm!('T', 'N', 1 / M, @view(H_copy[:, 2:1+n]), @view(H_copy[:, 2:1+n]), 0.0, Hd)
    k = 1
    @inbounds for i in 1:n
        for j in i:n
            packed[k] = Hd[i, j]
            k += 1
        end
    end
    return packed
    finally
        CS._exit_callback!(obj)
    end
end

"""
    fill_core_hessian_upper!(Hdense, curvature_weights, compressed_core, workspace, global_layout=nothing;
                              backend=:exact_winner_pair_parallel, workers=10, storage=:full_stride)

THE shared entry point every family calls to get the common core Hessian
block H_EE (extended with the zeta row/column and, when active, the
counterfactual column). Fills the SYMMETRIC dense `(1+ncolI) x (1+ncolI)`
block into `Hdense`, which may be:
  - the family's ENTIRE dense Hessian scratch (unrestricted: H_EE is the
    whole moment block, no restriction columns exist to carve off), or
  - a `@view` into the top-left corner of a LARGER family scratch
    (`cctx.Hfull[1:NCORE,1:NCORE]` for CM/CM+meanZC; a top-left corner view
    of `obj.∂∂f_∂∂x` for origin-ZC) -- the view's own offset bookkeeping IS
    the local-to-global packed-index map (task §2's "precomputed
    local-to-global packed-index maps" requirement), so there is no separate
    hand-rolled index table to keep in sync with each family's packing order.

`curvature_weights` is accepted (and expected to already equal `obj.arg2`
post-`ddPsi!`) purely for interface-contract documentation, matching every
other Hessian backend's calling convention in this codebase -- every backend
here re-derives it from `obj` internally (same as `winner_pair_hessian!`/
`hessian_cm_structured!`), so a stale value passed here has no effect.

`global_layout` is accepted for interface-contract symmetry with the task
brief's specified signature. Unused today (this implementation's
local-to-global map is `Hdense`'s own view offset, see above) -- kept as an
explicit no-op parameter, not silently dropped, so a future backend that DOES
need an explicit index table has a place to plug in without an interface
change.

Computation of the exact core upper triangle (this function, into
`workspace.packed_scratch`) is kept SEPARATE from assembly into the caller's
larger Hessian (the final unpack loop below) per task §2's explicit
instruction not to let family-specific code reimplement the core algebra.
"""
function fill_core_hessian_upper!(Hdense::AbstractMatrix, curvature_weights, obj,
        workspace::CoreExactHessianWorkspace, global_layout = nothing;
        backend::Symbol = :exact_winner_pair_parallel, workers::Int = 10, storage::Symbol = :full_stride)
    backend in CORE_HESSIAN_BACKENDS || error("fill_core_hessian_upper!: unknown backend :$backend (expected one of $CORE_HESSIAN_BACKENDS)")
    n = 1 + workspace.ncolI
    size(Hdense) == (n, n) || error("fill_core_hessian_upper!: Hdense size $(size(Hdense)) != ($n,$n)")
    packed = workspace.packed_scratch

    if backend === :exact_winner_pair_parallel
        hessian_core_winner_pair!(packed, curvature_weights, obj, workspace.parallel_ws; workers = workers, storage = storage)
        record_core_hessian_call!(:exact_winner_pair_parallel)
    elseif backend === :exact_winner_pair_serial
        winner_pair_hessian!(packed, obj, serial_ctx(workspace))
        record_core_hessian_call!(:exact_winner_pair_serial)
    else # :dense_reference
        _dense_reference_core_hessian!(packed, obj, n)
        record_core_hessian_call!(:dense_reference; fallback_reason = :debug_reference_requested)
    end

    k = 1
    @inbounds for i in 1:n
        for j in i:n
            v = packed[k]; k += 1
            Hdense[i, j] = v
            Hdense[j, i] = v
        end
    end
    return Hdense
end
