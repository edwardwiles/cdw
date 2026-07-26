# ============================================================================
# 2026-07-25 production wall-clock/allocation audit, Phase 9 low-risk fix (finding F1,
# PRODUCTION_HOTPATH_STATIC_ALLOCATION_REVIEW_2026-07-25.md).
#
# ADDITIVE ONLY -- does not modify compressed_moments.jl::build_compressed_factual or any
# existing call site. Opt-in via an explicit CompressedFactualWorkspace argument; every
# existing caller of build_compressed_factual is unaffected.
#
# build_compressed_factual's own `winner::Matrix{Int}(undef,W,Ddest)`,
# `wval::Matrix{Float64}(undef,W,Ddest)`, `cf_raw::Vector{Float64}(undef,W)` (compressed_
# moments.jl:151-152,195) are freshly heap-allocated EVERY call (~23.8 MiB total at
# W=80000/Ddest=19) even though W/Ddest are fixed for the whole campaign -- this file
# provides a workspace-reusing alternative, following the SAME pattern this codebase
# already uses for `LFixFactorizedWorkspace` (lfix_factorized_workspace.jl, which owns an
# identically-shaped `winner::Matrix{Int}(W,Ddest)` field as a genuinely persistent,
# campaign-lifetime buffer -- the precedent this fix follows exactly).
# ============================================================================

"Persistent, campaign-lifetime scratch for `build_compressed_factual!` -- owns the SAME-shaped `winner`/`wval`/`cf_raw` buffers `build_compressed_factual` otherwise allocates fresh every call."
mutable struct CompressedFactualWorkspace
    D::Int
    Ddest::Int
    W::Int
    winner::Matrix{Int}
    wval::Matrix{Float64}
    cf_raw::Vector{Float64}
end

"`build_compressed_factual_workspace(D, Ddest, W)` -- one-time allocation, matches `build_lfix_factorized_workspace`'s own construction pattern."
function build_compressed_factual_workspace(D::Int, Ddest::Int, W::Int)
    return CompressedFactualWorkspace(D, Ddest, W,
        Matrix{Int}(undef, W, Ddest), Matrix{Float64}(undef, W, Ddest), Vector{Float64}(undef, W))
end

"Rebuilds only on a genuine (D,Ddest,W) change -- mirrors `ensure_lfix_factorized_workspace!`."
function ensure_compressed_factual_workspace!(ws_ref::Base.RefValue{CompressedFactualWorkspace}, D::Int, Ddest::Int, W::Int)
    ws = ws_ref[]
    if ws.D != D || ws.Ddest != Ddest || ws.W != W
        ws_ref[] = build_compressed_factual_workspace(D, Ddest, W)
    end
    return ws_ref[]
end

"""
    build_compressed_factual!(ws::CompressedFactualWorkspace, θ_full, ctx; check_ties=true) -> CompressedFactual

Buffer-reusing variant of `build_compressed_factual` (compressed_moments.jl) -- IDENTICAL
formula/control-flow, but fills `ws.winner`/`ws.wval`/`ws.cf_raw` in place instead of allocating
fresh `Matrix{Int}(undef,W,Ddest)`/`Matrix{Float64}(undef,W,Ddest)`/`zeros(W)` every call. The
returned `CompressedFactual` ALIASES `ws`'s buffers (does not copy them) -- exactly the aliasing
discipline `LFixFactorizedWorkspace`/`GradWorkspacePool` already use elsewhere in this codebase:
the caller must not retain the returned `CompressedFactual` across a SUBSEQUENT call to this
function with the same `ws`, since that call overwrites `ws.winner`/`ws.wval`/`ws.cf_raw` in
place. Safe within the scope this is used for in production (one `CompressedFactual` per inner
solve, consumed immediately by `CompressedCBState` and not retained afterward).

Bit-identical output to `build_compressed_factual` given the same `(θ_full, ctx, check_ties)` --
verified in `test_compressed_factual_buffer_reuse.jl` (D=4 synthetic and real D=20/W=80000 points).
"""
function build_compressed_factual!(ws::CompressedFactualWorkspace, θ_full::AbstractVector, ctx; check_ties::Bool = true)
    γo = ctx.γ
    D = ctx.D; Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    U = ctx.U; W = size(U, 1)
    (D == ws.D && Ddest == ws.Ddest && W == ws.W) ||
        error("build_compressed_factual!: workspace shape ($(ws.D),$(ws.Ddest),$(ws.W)) does not match ctx ($D,$Ddest,$W) -- call ensure_compressed_factual_workspace! first")
    μ = θ_full[1]; σ = θ_full[2]
    ind = γo.indicators
    oci = ctx.obj.outer_constr_index

    pp = canonical_price_precompute(θ_full, ctx)
    constCons = pp.constCons; constConsσ = pp.constConsσ; logCC = pp.logCC
    mulU = pp.mulU; UPow = pp.UPow; UσPow = pp.UσPow; AodPow = pp.AodPow
    denom = [γo.wHat[global_destination(ctx, s)] * γo.L[global_destination(ctx, s)] for s in 1:Ddest]
    Pmat = [γo.P[s + (o - 1) * Ddest] for o in 1:D, s in 1:Ddest]

    winner = ws.winner
    wval = ws.wval
    tied = Tuple{Int,Int}[]
    n_tied = 0
    @inbounds for s in 1:Ddest
        for w in 1:W
            bo, _ = canonical_winner_argmin(logCC, mulU, w, s, D)
            best = constCons[bo, s] / UPow[w, bo]
            winner[w, s] = bo
            wval[w, s] = constConsσ[bo, s] / UσPow[w, bo]
            if check_ties
                c = 0
                for o in 1:D
                    (constCons[o, s] / UPow[w, o] <= best) && (c += 1)
                end
                if c > 1
                    n_tied += 1
                    length(tied) < 5 && push!(tied, (w, s))
                end
            end
        end
    end
    if check_ties && n_tied > 0
        throw(TiedWinnerError(n_tied, tied))
    end

    gammafac = spgamma(μ * (1 - σ) + 1)
    ncol = oci - 1
    ncell = D * Ddest
    gdiv = [j <= ncell + 1 ? 1.0 / gammafac : 1.0 for j in 1:ncol]
    NM = ind.NormalizeMoments
    without = γo.moments_without_var
    nrm = [(NM == 1 && !(j in without)) ? 1.0 / γo.σ_Moments[j] : 1.0 for j in 1:ncol]
    usePMM = ind.usePMM
    PMMv = usePMM == 1 ? Float64[γo.PMM[j] for j in 1:ncol] : zeros(ncol)
    SW = γo.SamplingWeights[1:W]

    cf_col = ncell + 1
    cf_raw = ws.cf_raw
    if cf_col <= ncol
        bi = ctx.bi
        wPrime = copy(γo.wPrimeHat); insert!(wPrime, bi, 1.0)
        wPrime_bi = wPrime[bi]
        τPrime_bi = γo.τPrime[bi, bi]
        LPrime_bi = γo.LPrime[bi]
        AodPow_bibi = AodPow[bi, dest_slot(ctx, bi)]
        γ_prime_bi = θ_full[3 + D]
        constConsσ_bibi = wPrime_bi^(1 - σ) * (AodPow_bibi * τPrime_bi)^(1 - σ)
        denom_cf = γ_prime_bi^σ * (wPrime_bi * LPrime_bi)
        UσPow_bi = @view(γo.Uσ[:, bi]) .^ (-μ)
        @. cf_raw = constConsσ_bibi / UσPow_bi - denom_cf
    else
        fill!(cf_raw, 0.0)   # NOT a no-op here unlike the fresh-`zeros(W)` original -- a REUSED
                              # buffer can carry a nonzero cf_raw from a PRIOR call, so this branch
                              # must explicitly clear it (the original never needed to, since a
                              # fresh `zeros(W)` is already all-zero).
        cf_col = 0
    end

    return CompressedFactual(D, Ddest, W, oci, winner, wval, Pmat, denom, gdiv, nrm,
        PMMv, usePMM, SW, gammafac, cf_raw, cf_col, 0, Tuple{Int,Int}[])
end

# ============================================================================
# Production wiring (allocation/Hessian port task, 2026-07-25, section 3.1). `ctx` is a plain
# NamedTuple threaded everywhere `screened_eval` is (see run_profile_checkpointed/
# run_polish_checkpointed, c10_d20_production_driver.jl) -- attaching the workspace here, rather
# than adding a new argument to screened_eval/evaluate_fullA_screened_ranged, means every call
# site downstream that already receives `ctx` picks it up for free. `merge(ctx, (...,))` preserves
# every existing field by reference (same guarantee `set_context_delta!` already relies on for
# `ctx.obj` -- reusable_context.jl), so a workspace attached once survives a same-process
# `set_context_delta!` delta-stage change untouched.
# ============================================================================

"""
    attach_compressed_factual_workspace(ctx, D, Ddest, W) -> ctx

Returns `ctx` merged with a `cf_workspace::CompressedFactualWorkspace` field. Builds a fresh
workspace only if `ctx` doesn't already carry one of the correct `(D,Ddest,W)` shape -- so calling
this on an already-augmented `ctx` (e.g. one inherited via `reuse=`/from a resumed checkpoint's
reconstructed context) is a no-op reuse, not a rebuild, matching this codebase's existing
`ensure_lfix_factorized_workspace!`/`ensure_compressed_factual_workspace!` "resize only on genuine
shape change" discipline. Call once per outer-solve process, immediately after `ctx` is built/
reused/resumed, before any `screened_eval` call.
"""
function attach_compressed_factual_workspace(ctx, D::Int, Ddest::Int, W::Int)
    existing = hasproperty(ctx, :cf_workspace) ? ctx.cf_workspace : nothing
    ws = (existing isa CompressedFactualWorkspace && existing.D == D && existing.Ddest == Ddest && existing.W == W) ?
        existing : build_compressed_factual_workspace(D, Ddest, W)
    return merge(ctx, (cf_workspace = ws,))
end

"""
    cf_build(θ_full, ctx; check_ties=true) -> CompressedFactual

Phase E remediation (production-audit continuation, 2026-07-26): dispatch helper used by the four
restricted families' `moments!` closures (`cm_hessian_architectures.jl`, `cm_meanzc_moments.jl`,
`cm_frechet_level.jl`, `cm_originzc_moments.jl`) -- reuses `ctx.cf_workspace` via
`build_compressed_factual!` when the caller attached one (`attach_compressed_factual_workspace`),
falls back to the original allocating `build_compressed_factual` otherwise (every pre-existing
caller that never attaches a workspace is completely unaffected). Bit-identical output either way
(same guarantee `build_compressed_factual!`'s own docstring already establishes).
"""
cf_build(θ_full::AbstractVector, ctx; check_ties::Bool = true) =
    hasproperty(ctx, :cf_workspace) ?
        build_compressed_factual!(ctx.cf_workspace, θ_full, ctx; check_ties = check_ties) :
        build_compressed_factual(θ_full, ctx; check_ties = check_ties)
