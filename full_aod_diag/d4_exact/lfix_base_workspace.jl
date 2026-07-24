# ============================================================================
# Addendum "test persistent preallocation and eliminate redundant full price
# tensors", Backend A: persistent caller-owned workspace for
# `build_lfix_base_cache`'s large arrays.
#
# MOTIVATION (measured in docs/fullA_price_tensor_audit.md +
# results/fullA_d4/c15_price_tensor/price_tensor_benchmark.csv): at real
# D=20/W=80000, `build_lfix_base_cache` allocates two fresh W*D*D=3.2e7-element
# Float64 tensors (price0, pTσ0, ~256MB each = ~512MB) PLUS several smaller
# W*D and W-length arrays (winner0/runnerup0/third0 and their price/pTσ levels,
# contrib0, q0, cf pieces, SW) EVERY SINGLE call, even though every one of
# those arrays is fully OVERWRITTEN (not read-then-extended) on every call and
# nothing outside the returned `LFixBaseCache` retains a reference to the old
# one once a new gradient starts (verified: `grep`-audited, no call site in
# this codebase stores an `LFixBaseCache` across gradient calls -- see the
# audit doc's confirmation this file's docstrings below rely on).
#
# DESIGN: `LFixBaseCache` itself is UNCHANGED (still an immutable struct,
# still constructed the same way, still consumed identically by every existing
# `dest_contrib_*`/`count_winner_flips*`/`select_bandwidth`/`bandwidth_quantile`
# function -- none of those are touched by this file). What changes is HOW the
# arrays INSIDE it are allocated: `build_lfix_base_cache!(ws, ...)` fills a
# caller-owned `LFixBaseWorkspace`'s persistent arrays in place, then wraps
# them in a freshly-constructed (but zero-large-allocation) `LFixBaseCache`
# whose big-array fields ALIAS the workspace's buffers rather than pointing at
# `undef`-allocated fresh memory. Every existing consumer function is
# unaffected because it never knew or cared where `cache.price0` etc. came
# from in the first place.
#
# CORRECTNESS INVARIANT THIS RELIES ON (mechanically re-verified by this
# file's own grep-based comment, not just asserted): no code anywhere in this
# repo stores a `LFixBaseCache` returned by ANY builder (allocating or
# workspace-backed) across more than one gradient/value evaluation. If a
# caller ever violates this (e.g. stashes `meta.cache` from one KNITRO
# iterate and reads it again after the NEXT iterate has refilled the same
# workspace), IT WILL SEE CORRUPTED, POINT-MISMATCHED DATA -- this is the
# exact hazard the addendum's brief calls out ("no mutable workspace arrays
# may be stored by reference inside immutable exact-cache records"). The one
# existing cross-call cache in this codebase, `CrossDeltaExactCache`
# (cross_delta_cache.jl), stores only a `NamedTuple` of SCALAR/small results
# (Delta_dual, θ_full, zeta, lambda, moment_resid, gravity_value) -- never an
# `LFixBaseCache` -- so it is unaffected by this file.
#
# THREAD SAFETY: one `LFixBaseWorkspace` must NEVER be shared by two
# concurrently-running gradient evaluations (matches `GradWorkspacePool`'s own
# per-thread-slot discipline in gradient_workspace.jl -- reused here, not
# reinvented). Callers running multiple gradients concurrently (e.g. a
# multi-start batch) must give each concurrent gradient its own workspace.
# ============================================================================
include(joinpath(@__DIR__, "lfix_incremental.jl"))

"""
    LFixBaseWorkspace

Caller-owned, mutable, persistent backing store for `LFixBaseCache`'s large arrays. Built ONCE
per `(D, W)` via `build_lfix_base_workspace`, refilled in place on every subsequent gradient
call via `build_lfix_base_cache!` -- never reallocated as long as `(D, W)` stay fixed (the
steady-state case across an entire KNITRO run / staged δ-continuation at one problem size).

`valid` is `false` from construction and after any failed/partial `build_lfix_base_cache!` call
(tie error, dimension mismatch, or any exception mid-build) -- callers should not read the
workspace's arrays directly (as opposed to through a just-returned `LFixBaseCache`) unless
`valid` is `true`. `fingerprint` records a fast, non-cryptographic hash of the `(x_free0,
context-fingerprint)` pair the workspace's CURRENT contents were built from -- purely a
diagnostic/debugging aid (e.g. to catch a caller reusing a workspace's raw fields at the wrong
point); it is NOT used to skip rebuilding (every call to `build_lfix_base_cache!` always
recomputes from scratch, matching the allocating reference builder's own semantics -- no
implicit staleness-based caching is introduced here).
"""
mutable struct LFixBaseWorkspace
    D::Int
    W::Int
    valid::Bool
    fingerprint::UInt64
    price0::Array{Float64,3}
    pTσ0::Array{Float64,3}
    winner0::Matrix{Int}
    winner_price0::Matrix{Float64}
    runnerup0::Matrix{Int}
    runnerup_price0::Matrix{Float64}
    third0::Matrix{Int}
    third_price0::Matrix{Float64}
    third_pTσ0::Matrix{Float64}
    contrib0::Matrix{Float64}
    SW::Vector{Float64}
    denom::Vector{Float64}
    CONST_d::Vector{Float64}
    q0::Vector{Float64}
    Uσ_bi::Vector{Float64}
    cf_contrib0::Vector{Float64}
end

"`build_lfix_base_workspace(D, W)` -- one-time allocation of every persistent array `build_lfix_base_cache!` needs, sized for problem `(D, W)`. `valid=false` until the first successful `build_lfix_base_cache!` call."
function build_lfix_base_workspace(D::Int, W::Int)
    return LFixBaseWorkspace(D, W, false, UInt64(0),
        Array{Float64}(undef, W, D, D), Array{Float64}(undef, W, D, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Int}(undef, W, D), Matrix{Float64}(undef, W, D), Matrix{Float64}(undef, W, D),
        Matrix{Float64}(undef, W, D),
        Vector{Float64}(undef, W), Vector{Float64}(undef, D), Vector{Float64}(undef, D),
        Vector{Float64}(undef, W), Vector{Float64}(undef, W), Vector{Float64}(undef, W))
end

"""
    ensure_lfix_workspace!(ws_ref, D, W) -> LFixBaseWorkspace

`ws_ref` is a `Base.RefValue{LFixBaseWorkspace}` (or any 1-element mutable holder) so this
function can REPLACE the workspace object when `(D, W)` change (a genuinely new problem size),
matching `gradient_workspace.jl::resize_pool_if_needed!`'s own reassignment pattern. Returns the
(possibly rebuilt) workspace. A no-op reallocation-wise when `(D, W)` already match -- the
steady-state case.
"""
function ensure_lfix_workspace!(ws_ref::Base.RefValue{LFixBaseWorkspace}, D::Int, W::Int)
    ws = ws_ref[]
    if ws.D != D || ws.W != W
        ws_ref[] = build_lfix_base_workspace(D, W)
    end
    return ws_ref[]
end

"Fast, non-cryptographic fingerprint of the (x_free0, context) pair a workspace build is for -- diagnostic only, see `LFixBaseWorkspace`'s own docstring."
_lfix_ws_fingerprint(x_free0::AbstractVector, ctx) = hash(x_free0, hash(objectid(ctx)))

"""
    build_lfix_base_cache!(ws::LFixBaseWorkspace, x_free0, ctx, base::BaseDualState; validate_dense=false) -> LFixBaseCache

In-place-backed twin of `build_lfix_base_cache`: SAME formulas, SAME order of operations, SAME
tie-detection/self-validation discipline -- the only difference is every large array is written
into `ws`'s persistent buffers (`.=`/`copyto!`/direct index assignment) instead of freshly
`undef`-allocated. The returned `LFixBaseCache`'s array fields ALIAS `ws`'s buffers (zero-copy).

`ws.valid` is set `false` at entry and only set `true` after every step below (including the tie
check and, if requested, the dense self-validation) has succeeded -- an exception at any point
(dimension mismatch via the `@assert`, `TiedWinnerError`, a failed self-validation `error()`)
leaves `ws.valid == false`, signaling to any caller who inspects `ws` directly (rather than just
using the returned `LFixBaseCache`, which callers should always prefer) that its contents are
not to be trusted. Throws `DimensionMismatch` immediately (before touching any array) if `ws`'s
`(D, W)` don't match `ctx`'s -- callers must `ensure_lfix_workspace!` first; this function never
silently reallocates (that would defeat the entire persistent-preallocation point).

Requires `x_free0`'s Aod block to be strictly positive with no exact price ties, exactly like
the allocating reference -- `TiedWinnerError` propagates identically.
"""
function build_lfix_base_cache!(ws::LFixBaseWorkspace, x_free0::AbstractVector, ctx, base::BaseDualState; validate_dense::Bool = false)
    ws.valid = false
    obj = ctx.obj
    D = ctx.D; W = size(obj.U, 1); oci = obj.outer_constr_index
    # This persistent-workspace backend is still square-only (D x D tensors throughout) --
    # NOT generalized to D x Ddest this pass (out of scope, see lfix_incremental.jl's
    # allocating build_lfix_base_cache for the true-shrink-aware version). Hard-error rather
    # than silently misbehave if ever handed a rectangular (:exclude_row) context.
    Ddest_here = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    Ddest_here == D || error("build_lfix_base_cache!: this workspace-pooled backend is still square-only (D=$D, Ddest=$Ddest_here) -- not implemented for destination_sample=:exclude_row.")
    (ws.D == D && ws.W == W) || throw(DimensionMismatch(
        "build_lfix_base_cache!: workspace is (D=$(ws.D),W=$(ws.W)), context needs (D=$D,W=$W) -- call ensure_lfix_workspace! first"))
    μ = base.θ_full0[1]; σ = ctx.σ; bi = ctx.bi
    γo = ctx.γ
    gammafac = spgamma(μ * (1 - σ) + 1)
    λstar = base.λstar

    ws.SW .= @view γo.SamplingWeights[1:W]
    for d in 1:D
        ws.denom[d] = γo.wHat[d] * γo.L[d]
    end

    price0 = ws.price0; pTσ0 = ws.pTσ0
    for d in 1:D, o in 1:D
        price_and_pTsigma_cell!(@view(price0[:, o, d]), @view(pTσ0[:, o, d]), base.θ_full0, ctx, o, d)
    end

    tied_pairs = detect_price_ties(price0, D, W)
    isempty(tied_pairs) || throw(TiedWinnerError(length(tied_pairs), tied_pairs[1:min(5, end)]))

    winner0 = ws.winner0; winner_price0 = ws.winner_price0
    runnerup0 = ws.runnerup0; runnerup_price0 = ws.runnerup_price0
    third0 = ws.third0; third_price0 = ws.third_price0; third_pTσ0 = ws.third_pTσ0
    @inbounds for d in 1:D, ω in 1:W
        m1, idx1, m2, idx2, m3, idx3 = min_secondthirdmin_with_idx(@view(price0[ω, :, d]))
        winner0[ω, d] = idx1; winner_price0[ω, d] = m1
        runnerup0[ω, d] = idx2; runnerup_price0[ω, d] = m2
        third0[ω, d] = idx3; third_price0[ω, d] = m3
    end
    @inbounds for d in 1:D, ω in 1:W
        t = third0[ω, d]
        third_pTσ0[ω, d] = t == 0 ? Inf : pTσ0[ω, t, d]
    end

    CONST_d = ws.CONST_d
    for d in 1:D
        s = 0.0
        for o in 1:D
            d1 = d + (o - 1) * D
            s += λstar[d1] * (-γo.P[d1] * ws.denom[d])
        end
        CONST_d[d] = s
    end

    contrib0 = ws.contrib0
    @inbounds for d in 1:D, ω in 1:W
        wo = winner0[ω, d]
        d1w = d + (wo - 1) * D
        contrib0[ω, d] = (ws.SW[ω] / gammafac) * (CONST_d[d] + λstar[d1w] * pTσ0[ω, wo, d])
    end

    wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
    wPrime_bi = wPrime[bi]
    τPrime_bi = γo.τPrime[bi, bi]
    LPrime_bi = γo.LPrime[bi]
    ws.Uσ_bi .= @view(γo.Uσ[:, bi]) .^ (-μ)
    d1_cf = D^2 + 1
    λ_cf = oci - 1 >= d1_cf ? λstar[d1_cf] : 0.0

    AodPow_bibi0 = aod_pow_cell(base.θ_full0, ctx, bi, bi)
    γ_prime_bi0 = base.θ_full0[3+D]
    constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi0 * τPrime_bi)^(1 - σ)
    denom_cf0 = γ_prime_bi0^σ * wPrime_bi_gdp(wPrime_bi, LPrime_bi)
    ws.cf_contrib0 .= λ_cf .* ((constConsσ_bibi ./ ws.Uσ_bi .- denom_cf0) ./ gammafac .* ws.SW)

    q0 = ws.q0
    @inbounds for s in 1:W
        q0[s] = -base.ζstar - sum(@view(contrib0[s, :])) - ws.cf_contrib0[s]
    end

    if validate_dense
        K = zeros(W); Gfull = zeros(W, obj.d)
        obj.moments!(K, Gfull, base.θ_full0, obj.U, obj)
        maxerr = 0.0
        @inbounds for s in 1:W
            q0_true_s = -base.ζstar - dot(λstar, @view(Gfull[s, 1:oci-1]))
            maxerr = max(maxerr, abs(q0_true_s - q0[s]))
        end
        maxerr < 1e-8 || error("build_lfix_base_cache!: self-validation FAILED, max|q0_true-q0_cache|=$maxerr")
    end

    ws.fingerprint = _lfix_ws_fingerprint(x_free0, ctx)
    ws.valid = true

    return LFixBaseCache(D, D, oci, W, μ, σ, bi, gammafac, ws.SW, ws.denom, ws.CONST_d, price0, pTσ0,
        winner0, winner_price0, runnerup0, runnerup_price0,
        third0, third_price0, third_pTσ0, contrib0,
        λstar, base.ζstar, q0, wPrime_bi, τPrime_bi, LPrime_bi, ws.Uσ_bi, λ_cf, ws.cf_contrib0)
end
