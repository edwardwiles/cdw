# ============================================================================
# Winner-margin certificates + coordinate-update specialization for repeated
# hard-winner recomputes when the draw matrix U and (mu,sigma) are FIXED and
# only the bilateral log-A shifts vary.
#
# ADDITIVE / DIAGNOSTIC ONLY. Does NOT modify the trusted dense constructor,
# the tie-breaking rule (misc/smoothMinIndNew!.jl::MinInd!), winners.jl,
# winners_v2.jl, lfix_incremental.jl, compressed_moments.jl, or the exact hard
# estimand. The TRUSTED reference throughout is winners_v2.jl::compute_winners_fast
# (== winners.jl::compute_winners, first-occurrence argmin, bit-for-bit).
#
# ----------------------------------------------------------------------------
# SCORE CONVENTION (verified from code, not assumed) -- see winners.jl /
# compressed_moments.jl / hFunction.jl. For draw s (=omega), origin o,
# destination d, hFunction!/MinInd! pick the winner as
#
#     winner_{s,d} = argmin_o  price_{s,o,d},
#     price_{s,o,d} = constCons_{o,d} / U_{s,o}^{-mu}  =  constCons_{o,d} * U_{s,o}^{mu}
#     constCons_{o,d} = wHat_o * AodPow_{o,d} * tau_{o,d}
#     AodPow_{o,d}    = (Aod_{o,d}/cHat_{o,d})^{-mu}
#     Aod_{o,d}       = Aod_theta_{o,d} * cHat_{o,d} * ((...)^{1/mu}) * (lambda_od/lambda_1d)
#
# Only Aod_theta (= exp(z), z = the reduced log-A coordinate) moves during the
# optimization; wHat, tau, cHat, U, mu, sigma, lambda are FIXED data.
#
# Taking logs and writing the DESTINATION-d score S_{sod} := log price_{s,o,d}:
#
#     S_{sod} = B_{sod} + a_{od},
#     a_{od}  = log constCons_{o,d}'s A-dependent part = -mu * log Aod_theta_{o,d} = -mu * z_{od},
#     B_{sod} = log(wHat_o) + log(tau_{o,d}) - mu*log((...)^{1/mu} * lambda_od/lambda_1d)
#               + mu*log U_{s,o}       (everything FIXED across the optimization).
#
# (The user's schematic writes B_{so}; B genuinely also depends on d through
#  tau_{o,d} and the lambda ratio, but the certificate below is INVARIANT to
#  that -- every d-dependent, optimization-fixed term cancels in the winner
#  gap S_{skd}-S_{swd}. Verified: only the per-(o,d) shift a_{od} enters.)
#
# Because a_{od} does NOT depend on the draw s, moving from theta -> theta'
# shifts EVERY draw's score for cell (o,d) by the SAME amount
#
#     delta_{od} := S'_{sod} - S_{sod} = a'_{od} - a_{od}
#                 = log constCons'_{o,d} - log constCons_{o,d}
#                 = -mu * (z'_{od} - z_{od}).
#
# That single D x D shift matrix (O(D^2), no draw loop) is what the certificate
# screens against -- this is the whole reason the certificate is cheap.
# ============================================================================
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "winners_v2.jl"))
include(joinpath(@__DIR__, "winners.jl"))
using LinearAlgebra: dot

# ---- exact code convention helpers -----------------------------------------

"""
    constCons_matrix(theta_full, ctx) -> (constCons::Matrix, logCC::Matrix, AodPow::Matrix)

D x D `constCons[o,d] = wHat_o * AodPow_{o,d} * tau_{o,d}` exactly as hFunction!
receives it (via factual_prices), plus its elementwise log (the additive score
offset) and AodPow. O(D^2), no draw loop.
"""
function constCons_matrix(θ_full::AbstractVector, ctx)
    γo = ctx.γ
    D = ctx.D; μ = θ_full[1]
    lambda = reshape(γo.P, (D, D))'
    Aod_θ = reshape(θ_full[ctx.Aod_offset+1:ctx.Aod_offset+D^2], (D, D))
    Aod = Aod_θ .* γo.cHat .* (((γo.wHat .* γo.τ) ./ (γo.wHat[1, 1] .* γo.τ[1, :]')) .^ (1 / μ)) .* (lambda ./ lambda[1, :]')
    AodPow = (Aod ./ γo.cHat) .^ (-μ)
    constCons = [γo.wHat[o] * AodPow[o, d] * γo.τ[o, d] for o in 1:D, d in 1:D]
    return constCons, log.(constCons), AodPow
end

# ---- reference cache --------------------------------------------------------

"""
    WinnerRefCache

Winner-margin reference at an evaluated point. Stores, per (draw s, destination d),
the top-3 competitors by score (winner / runner-up / third) and the winner-runner-up
margin, plus the O(W*D) precompute `mulU_{s,o}=mu*log U_{s,o}` needed to rescan any
cell exactly. O(W*D) memory (NOT the O(W*D^2) dense price array).

Scores are `S_{sod}=logCC_{o,d}+mulU_{s,o}` (log price). Winner = argmin (lowest
score) with first-occurrence tie-break, IDENTICAL to compute_winners_fast/MinInd!.
"""
struct WinnerRefCache
    D::Int
    W::Int
    μ::Float64
    σ::Float64
    logCC0::Matrix{Float64}      # D x D reference log constCons
    mulU::Matrix{Float64}        # W x D : mu*log U  (draw-dependent, A-independent)
    winner::Matrix{Int}          # W x D : rank-1 origin (== compute_winners_fast winner)
    sw::Matrix{Float64}          # W x D : winner score
    runnerup::Matrix{Int}        # W x D : rank-2 origin
    sr::Matrix{Float64}          # W x D : runner-up score
    third::Matrix{Int}           # W x D : rank-3 origin (0 if D<3)
    st3::Matrix{Float64}         # W x D : third score (Inf if D<3)
    margin::Matrix{Float64}      # W x D : sr - sw  (>= 0, the min gap)
    x_free0::Vector{Float64}
    θ_full0::Vector{Float64}
end

"""
    top3_scan(col) -> (i1,s1,i2,s2,i3,s3)

First-occurrence rank-1/2/3 by `isless` over `col` (a length-D score view).
Matches winners_v2.jl::min_and_secondmin for the top-2 exactly; extends to
rank-3 (needed for exact 2-changed-origin coordinate updates).
"""
@inline function top3_scan(col)
    n = length(col)
    s1 = col[1]; i1 = 1
    s2 = oftype(s1, Inf); i2 = 0
    s3 = oftype(s1, Inf); i3 = 0
    @inbounds for i in 2:n
        v = col[i]
        if isless(v, s1)
            s3 = s2; i3 = i2
            s2 = s1; i2 = i1
            s1 = v;  i1 = i
        elseif isless(v, s2)
            s3 = s2; i3 = i2
            s2 = v;  i2 = i
        elseif isless(v, s3)
            s3 = v;  i3 = i
        end
    end
    return i1, s1, i2, s2, i3, s3
end

"""
    build_winner_ref(x_free0, ctx) -> WinnerRefCache

Full O(W*D^2) reference scan (the only full winner scan needed; every later
nearby point is screened/updated incrementally). Detects exact price ties
(reusing detect-tie semantics) and throws `TiedWinnerError` if any (draw,dest)
has 2+ origins bit-exactly tied at the row minimum -- the unique-winner
assumption fails there (same contract as lfix/compressed prior art).
"""
function build_winner_ref(x_free0::AbstractVector, ctx; check_ties::Bool = true)
    θ_full0 = CS.reconstruct_full(x_free0, ctx.m)
    D = ctx.D; U = ctx.U; W = size(U, 1)
    μ = θ_full0[1]; σ = θ_full0[2]
    constCons0, logCC0, _ = constCons_matrix(θ_full0, ctx)
    mulU = μ .* log.(U)                       # W x D, mu*log U_{s,o}
    UPow = U .^ (-μ)                          # W x D, matches MinInd!'s price = constCons/UPow (for tie detection)

    winner = Matrix{Int}(undef, W, D); sw = Matrix{Float64}(undef, W, D)
    runnerup = Matrix{Int}(undef, W, D); sr = Matrix{Float64}(undef, W, D)
    third = Matrix{Int}(undef, W, D); st3 = Matrix{Float64}(undef, W, D)
    margin = Matrix{Float64}(undef, W, D)
    scol = Vector{Float64}(undef, D)
    n_tied = 0; tied = Tuple{Int,Int}[]
    @inbounds for d in 1:D
        for s in 1:W
            for o in 1:D
                scol[o] = logCC0[o, d] + mulU[s, o]
            end
            i1, s1, i2, s2, i3, s3 = top3_scan(scol)
            winner[s, d] = i1; sw[s, d] = s1
            runnerup[s, d] = i2; sr[s, d] = s2
            third[s, d] = i3; st3[s, d] = s3
            margin[s, d] = s2 - s1
            if check_ties
                # detect ties in PRICE space (constCons/UPow), exactly as MinInd! compares,
                # so an exact price tie is caught even if the log round-trip would not reproduce it.
                pmin = constCons0[1, d] / UPow[s, 1]
                for o in 2:D
                    p = constCons0[o, d] / UPow[s, o]
                    (p < pmin) && (pmin = p)
                end
                c = 0
                for o in 1:D
                    (constCons0[o, d] / UPow[s, o] <= pmin) && (c += 1)
                end
                if c > 1
                    n_tied += 1
                    length(tied) < 5 && push!(tied, (s, d))
                end
            end
        end
    end
    check_ties && n_tied > 0 && throw(TiedWinnerError(n_tied, tied))
    return WinnerRefCache(D, W, μ, σ, logCC0, mulU, winner, sw, runnerup, sr,
                          third, st3, margin, collect(x_free0), θ_full0)
end

# ---- Section 1: winner-margin certificate ----------------------------------

"""
    CertStats

Per-call bookkeeping for a certificate evaluation.
"""
struct CertStats
    n_cells::Int          # W*D
    n_certified::Int      # certified-unchanged (no rescan)
    n_rescan::Int         # required an exact rescan
    n_switched::Int       # of the rescanned, how many actually switched winner
    fell_back_full::Bool  # true if the whole call fell back to a full scan
    maxabs_delta::Float64 # max_{o,d}|delta_{od}| (step-size proxy)
    reason::String
end

"""
    shift_matrix(ref, ctx, theta_full') -> (delta::Matrix, logCC'::Matrix, maxabsδ)

delta_{od} = log constCons'_{o,d} - logCC0_{o,d}, the per-cell score shift
(same for all draws). Also cross-checks delta == -mu*(z'-z) to machine
precision (verifies the exact code convention), erroring loudly otherwise.
"""
function shift_matrix(ref::WinnerRefCache, ctx, θ_full′::AbstractVector)
    D = ref.D
    _, logCC′, _ = constCons_matrix(θ_full′, ctx)
    δ = logCC′ .- ref.logCC0
    # convention cross-check: delta must equal -mu*(z' - z), z = log Aod_theta
    z0 = log.(reshape(ref.θ_full0[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
    z1 = log.(reshape(θ_full′[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
    δ_from_z = -ref.μ .* (z1 .- z0)
    conv_err = maximum(abs.(δ .- δ_from_z))
    conv_err < 1e-9 || error("shift_matrix: score-convention mismatch, |delta - (-mu*dz)|=$conv_err")
    return δ, logCC′, maximum(abs.(δ))
end

"""
    certified_winner_update(ref, ctx, x_free'; tol_far=Inf) -> (winner'::Matrix{Int}, CertStats)

Section 1 deliverable. Screens every (draw,destination) with the exact
winner-margin certificate

    winner unchanged if   margin_{sd} > max_{k != w_{sd}} ( delta_{kd} - delta_{wd} ),

(equivalently m_{sd} > max_k delta_kd(k!=w) - delta_wd), certified cells reuse
the cached winner; only uncertified cells are exactly rescanned (O(D) each,
using logCC' + mulU). The returned winner matrix is IDENTICAL to a full
compute_winners_fast scan (proved: the margin inequality is strict, so a
certified cell's cached winner is the unique strict argmin at theta').

`tol_far`: if maxabs delta exceeds this, fall back to a full scan immediately
(callback point too far from the reference). Cache use never changes the
mathematical value.
"""
function certified_winner_update(ref::WinnerRefCache, ctx, x_free′::AbstractVector; tol_far::Float64 = Inf)
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    D = ref.D; W = ref.W
    δ, logCC′, maxabsδ = shift_matrix(ref, ctx, θ_full′)

    winner′ = Matrix{Int}(undef, W, D)

    if maxabsδ > tol_far
        # too far: full rescan (still exact), flagged as a fallback
        @inbounds for d in 1:D, s in 1:W
            bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
            for o in 2:D
                v = logCC′[o, d] + ref.mulU[s, o]
                (v < bs) && (bs = v; bo = o)
            end
            winner′[s, d] = bo
        end
        return winner′, CertStats(W * D, 0, W * D, -1, true, maxabsδ, "far: full rescan (tol_far=$tol_far)")
    end

    # per-destination MIN / 2nd-MIN of delta over origins (for the O(1) threshold).
    # Winner = argMIN(log price) [code convention]; w stays iff for all k!=w:
    #   gap_k := S_k - S_w > delta_w - delta_k, sufficient with margin=min gap:
    #   margin > delta_w - min_{k!=w} delta_k.  (min, not max: making a competitor
    #   CHEAPER -- lower delta -- is what threatens the incumbent minimum.)
    minδ = Vector{Float64}(undef, D); argminδ = Vector{Int}(undef, D); min2δ = Vector{Float64}(undef, D)
    @inbounds for d in 1:D
        m1 = Inf; a1 = 0; m2 = Inf
        for o in 1:D
            v = δ[o, d]
            if v < m1
                m2 = m1; m1 = v; a1 = o
            elseif v < m2
                m2 = v
            end
        end
        minδ[d] = m1; argminδ[d] = a1; min2δ[d] = m2
    end

    n_cert = 0; n_rescan = 0; n_switch = 0
    @inbounds for d in 1:D
        for s in 1:W
            w = ref.winner[s, d]
            mink = (argminδ[d] == w) ? min2δ[d] : minδ[d]   # min_{k!=w} delta_kd
            thr = δ[w, d] - mink
            if ref.margin[s, d] > thr
                winner′[s, d] = w         # certified unchanged
                n_cert += 1
            else
                # exact rescan of this cell only (O(D))
                bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
                for o in 2:D
                    v = logCC′[o, d] + ref.mulU[s, o]
                    (v < bs) && (bs = v; bo = o)
                end
                winner′[s, d] = bo
                n_rescan += 1
                (bo != w) && (n_switch += 1)
            end
        end
    end
    return winner′, CertStats(W * D, n_cert, n_rescan, n_switch, false, maxabsδ,
                              "certificate: $n_cert certified, $n_rescan rescanned")
end

"""
    winners_from_certificate(ref, ctx, x_free'; tol_far) -> (winner', wval', CertStats)

Feeds the certified/rescanned winner matrix into the compressed winning-CES-value
recompute (wval'_{s,d} = constConsσ'_{w',d} * Uσ_{s,w'}^{mu}), i.e. produces
exactly what a moment build needs, reusing the certificate. wval is only ever
evaluated for the ONE winning origin per (s,d) (O(W*D)).
"""
function winners_from_certificate(ref::WinnerRefCache, ctx, x_free′::AbstractVector; tol_far::Float64 = Inf)
    winner′, stats = certified_winner_update(ref, ctx, x_free′; tol_far = tol_far)
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    γo = ctx.γ; D = ref.D; W = ref.W; σ = ref.σ; μ = ref.μ
    _, _, AodPow = constCons_matrix(θ_full′, ctx)
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    constConsσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    UσPow = γo.Uσ .^ (-μ)
    wval = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, s in 1:W
        wo = winner′[s, d]
        wval[s, d] = constConsσ[wo, d] / UσPow[s, wo]
    end
    return winner′, wval, stats
end

# ---- Section 6: draw-level-threaded certificate ----------------------------

"""
    certified_winner_update_threaded(ref, ctx, x_free'; tol_far) -> (winner', CertStats)

Draw-level-threaded version of `certified_winner_update` for ORDINARY exact
value evaluations (NOT for use inside a coordinate-parallel L_fix gradient --
see the report's threading discipline). Each thread owns a disjoint draw-chunk;
winner' writes are disjoint by (s,d); the CertStats counters are accumulated in
thread-local buffers and reduced DETERMINISTICALLY (fixed thread order), so the
result is bit-identical to the single-threaded path and order-independent.
"""
function certified_winner_update_threaded(ref::WinnerRefCache, ctx, x_free′::AbstractVector; tol_far::Float64 = Inf)
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    D = ref.D; W = ref.W
    δ, logCC′, maxabsδ = shift_matrix(ref, ctx, θ_full′)
    winner′ = Matrix{Int}(undef, W, D)

    if maxabsδ > tol_far
        Threads.@threads for s in 1:W
            @inbounds for d in 1:D
                bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
                for o in 2:D
                    v = logCC′[o, d] + ref.mulU[s, o]
                    (v < bs) && (bs = v; bo = o)
                end
                winner′[s, d] = bo
            end
        end
        return winner′, CertStats(W * D, 0, W * D, -1, true, maxabsδ, "far: full rescan (threaded)")
    end

    minδ = Vector{Float64}(undef, D); argminδ = Vector{Int}(undef, D); min2δ = Vector{Float64}(undef, D)
    @inbounds for d in 1:D
        m1 = Inf; a1 = 0; m2 = Inf
        for o in 1:D
            v = δ[o, d]
            if v < m1
                m2 = m1; m1 = v; a1 = o
            elseif v < m2
                m2 = v
            end
        end
        minδ[d] = m1; argminδ[d] = a1; min2δ[d] = m2
    end

    nT = Threads.nthreads()
    cert_t = zeros(Int, nT); resc_t = zeros(Int, nT); sw_t = zeros(Int, nT)
    Threads.@threads for s in 1:W
        tid = Threads.threadid()
        @inbounds for d in 1:D
            w = ref.winner[s, d]
            mink = (argminδ[d] == w) ? min2δ[d] : minδ[d]
            thr = δ[w, d] - mink
            if ref.margin[s, d] > thr
                winner′[s, d] = w
                cert_t[tid] += 1
            else
                bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
                for o in 2:D
                    v = logCC′[o, d] + ref.mulU[s, o]
                    (v < bs) && (bs = v; bo = o)
                end
                winner′[s, d] = bo
                resc_t[tid] += 1
                (bo != w) && (sw_t[tid] += 1)
            end
        end
    end
    # deterministic reduction (fixed thread order)
    n_cert = 0; n_rescan = 0; n_switch = 0
    for t in 1:nT
        n_cert += cert_t[t]; n_rescan += resc_t[t]; n_switch += sw_t[t]
    end
    return winner′, CertStats(W * D, n_cert, n_rescan, n_switch, false, maxabsδ,
                              "certificate(threaded): $n_cert certified, $n_rescan rescanned")
end

"""
    winners_from_certificate_threaded(ref, ctx, x_free'; tol_far) -> (winner', wval', CertStats)

Continuation 8 addition: threaded-winner-matrix counterpart to
`winners_from_certificate`, for ORDINARY standalone value evaluations at large
W*D -- see `certified_winner_update_threaded`'s own threading-discipline
docstring (Section 6 above, and this repo's docs/winner_certificate_report.md
Context A: draw-level threading is a wash-to-slight-LOSS at the production
D=4/W=8000 scale, and only pays off at large W*D, e.g. D=10/W=80000). Do NOT
use inside an already-coordinate-threaded context (Context B: nested inner
threading measured 7.33x SLOWER than outer-threaded+inner-serial) -- this is
for a caller-level, non-nested value evaluation only.
"""
function winners_from_certificate_threaded(ref::WinnerRefCache, ctx, x_free′::AbstractVector; tol_far::Float64 = Inf)
    winner′, stats = certified_winner_update_threaded(ref, ctx, x_free′; tol_far = tol_far)
    θ_full′ = CS.reconstruct_full(x_free′, ctx.m)
    γo = ctx.γ; D = ref.D; W = ref.W; σ = ref.σ; μ = ref.μ
    _, _, AodPow = constCons_matrix(θ_full′, ctx)
    wPow = [γo.wHat[o]^(1 - σ) for o in 1:D]
    constConsσ = [wPow[o] * (AodPow[o, d] * γo.τ[o, d])^(1 - σ) for o in 1:D, d in 1:D]
    UσPow = γo.Uσ .^ (-μ)
    wval = Matrix{Float64}(undef, W, D)
    @inbounds for d in 1:D, s in 1:W
        wo = winner′[s, d]
        wval[s, d] = constConsσ[wo, d] / UσPow[s, wo]
    end
    return winner′, wval, stats
end

# ---- Section 2: coordinate-update specialization ---------------------------

"""
    coord_winner_update!(winner_out, ref, ctx, x_free', changed_cells) -> winner_out

Section 2 deliverable. EXACT winner recompute for a step that changes only a
SMALL set of A-cells `changed_cells` (a vector of (o,d)) -- the FD-gradient /
coordinate-descent case (1 cell for a direct coordinate, 2 cells for a
gravity-pivot coordinate, both landing in <=2 destinations). Uses the cached
top-3 ranking so NO O(D) rescan is needed even when the winner AND runner-up
both change in the same destination:

  * destinations with NO changed origin: winner unchanged (copied from ref).
  * a destination d with changed origins C_d (|C_d| in {1,2}): the new winner
    is argmin over { changed origins' NEW scores } U { best UNCHANGED origin }.
    The best unchanged origin is the first of (rank1,rank2,rank3) not in C_d --
    exact because with <=2 changed origins the best surviving unchanged origin
    is at worst rank 3 (top-3 cache suffices; proven, not assumed).

Falls back to the generic O(D) rescan for |C_d| > 2 (never happens for a single
reduced coordinate, kept for safety). Result is IDENTICAL to a full scan.
"""
function coord_winner_update!(winner_out::AbstractMatrix{Int}, ref::WinnerRefCache, ctx,
                              θ_full′::AbstractVector, changed_cells::AbstractVector{<:Tuple{Int,Int}})
    D = ref.D; W = ref.W
    _, logCC′, _ = constCons_matrix(θ_full′, ctx)
    # default: copy cached winners (unchanged destinations stay exact)
    copyto!(winner_out, ref.winner)
    # group changed origins by destination
    dests = unique(last.(changed_cells))
    for d in dests
        Cd = Int[o for (o, dd) in changed_cells if dd == d]
        if length(Cd) > 2
            @inbounds for s in 1:W          # generic O(D) rescan fallback
                bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
                for o in 2:D
                    v = logCC′[o, d] + ref.mulU[s, o]
                    (v < bs) && (bs = v; bo = o)
                end
                winner_out[s, d] = bo
            end
            continue
        end
        # precompute the new scores' logCC for the changed origins in this dest
        @inbounds for s in 1:W
            # candidate 1: best UNCHANGED origin via the top-3 ranking
            best_o = 0; best_s = Inf
            r1 = ref.winner[s, d]; r2 = ref.runnerup[s, d]; r3 = ref.third[s, d]
            if !(r1 in Cd)
                best_o = r1; best_s = ref.sw[s, d]
            elseif !(r2 in Cd)
                best_o = r2; best_s = ref.sr[s, d]
            elseif r3 != 0 && !(r3 in Cd)
                best_o = r3; best_s = ref.st3[s, d]
            end
            # candidate 2: the changed origins' NEW scores
            bo = best_o; bs = best_s
            for o in Cd
                v = logCC′[o, d] + ref.mulU[s, o]
                # Remediation task Part E (finding F5): canonical exact-tie convention --
                # lowest origin index wins (matches every generic-rescan tier's own natural
                # behavior, which scans o=1..D with strict `<`). Previously `if v < bs` alone,
                # which on an exact tie kept whichever candidate was considered FIRST (the
                # cached top-3 survivor, regardless of its index vs the tying changed origin's)
                # -- disagreed with the generic-rescan tiers whenever the survivor's index
                # exceeded the tying origin's. Measure-zero in exact arithmetic; see
                # test_winner_forced_tie.jl.
                if v < bs || (v == bs && o < bo)
                    bs = v; bo = o
                end
            end
            if bo == 0
                # extremely defensive: all of top-3 were changed (needs D<=3 & |Cd|>=3);
                # unreachable for |Cd|<=2 with D>=3. Full rescan.
                bo = 1; bs = logCC′[1, d] + ref.mulU[s, 1]
                for o in 2:D
                    v = logCC′[o, d] + ref.mulU[s, o]
                    (v < bs) && (bs = v; bo = o)
                end
            end
            winner_out[s, d] = bo
        end
    end
    return winner_out
end

# ============================================================================
# Continuation 8: persistent winner-margin cache for repeated NEARBY VALUE
# evaluations (line search / profile continuation / successive KNITRO outer
# iterates), per the standing brief's Section 3 second half. This is a
# WIRING layer on top of Sections 1+6 above -- it does not re-derive or
# change the certificate math, only owns the caller-persisted WinnerRefCache
# object across many calls and logs certified/rescanned/fallback fractions.
#
# SCOPE, decided by measurement (not assumed): only the WINNER identity (and
# hence the winner's wval/pTsigma) is provably exact under the winner-margin
# certificate -- certified_winner_update's proof is specifically that a
# certified cell's cached WINNER is the unique strict argmin at the new point;
# it says nothing about whether the runner-up/third-place identities are also
# unchanged (a lower-ranked competitor can cross the runner-up without ever
# threatening the winner). Reusing a stale runner-up/third under the
# winner-only certificate would NOT be provably exact, so this layer is
# deliberately scoped to what IS exact: winner + wval, i.e. an L_fix VALUE
# evaluator (lfix_value_certified, composite_gradient_fast.jl), not a
# replacement for the exact-top-3-dependent gradient tiers (those need exact
# runner-up/third and are handled by Section 2's per-call coordinate update /
# this continuation's count_winner_flips_multi_top3 / dest_contrib_incremental_
# top3 instead -- see composite_gradient.jl / lfix_incremental.jl).
#
# Measured justification for targeting build_lfix_base_cache's WINNER
# construction specifically: at D=4/W=8000 (warm), build_lfix_base_cache costs
# ~13ms, of which price0/pTsigma0 construction (the O(D^2) redundant-power-call
# loop over price_and_pTsigma_cell) is ~50% (~6.5ms) and the mandatory
# self-validation obj.moments! call is ~37% (~4.7ms, untouched here -- it is
# NOT a winner-computation cost and this layer does not skip it). By contrast
# build_winner_ref's log-decomposition (logCC + mulU, O(D^2)+O(W*D) one-time,
# NO per-cell power-call redundancy) gets the winner/runnerup/third ranking in
# ~1.85ms COLD -- already ~3.5x cheaper than price0's construction even before
# any persistence/certification; persistence across repeated nearby calls (via
# certified_winner_update's certify-then-rescan-only-uncertified path) drives
# the WARM per-call winner cost toward zero as the certified fraction rises to
# 97-100% at accepted/line-search step sizes (Section 1's own measurement).
# ============================================================================

"""
    PersistentWinnerCache

Caller-owned, mutable, PERSISTENT wrapper around a `WinnerRefCache` reference
plus cumulative `Ref`-style call counters (this codebase's own instrumentation
convention -- see `instrumentation.jl`'s `@prof`/`PROF_COUNTS` and
`oracle_fast.jl`'s `InnerCallCounters` -- reused here, not reinvented). Built
ONCE by the caller (e.g. at the first accepted outer point / first
profile-continuation step) and passed into `winner_value_update!` /
`lfix_value_certified` across MANY subsequent nearby calls -- NOT rebuilt every
call, per the standing brief's explicit "persistent cache object the caller
owns and passes across repeated calls" requirement.

Fields:
- `ref`: the current `WinnerRefCache` anchor, or `nothing` before first use.
- `tol_far`: forwarded to `certified_winner_update`/`_threaded` on every call
  (see that function's own docstring for the fallback semantics).
- `n_calls`, `n_cells_total`, `n_certified_cells`, `n_rescanned_cells`,
  `n_full_fallback_calls`, `n_rebuilds`: cumulative counters across the
  cache's lifetime (reset with `reset_counters!`).
- `total_cert_s`, `total_full_s`: cumulative wall time (seconds) spent in the
  cheap certified path vs. a full O(W*D^2)-ish rebuild/fallback path
  respectively -- the raw numbers `winner_cache_report(wc)` derives its speedup estimate
  from.
"""
mutable struct PersistentWinnerCache
    ref::Union{Nothing,WinnerRefCache}
    tol_far::Float64
    n_calls::Int
    n_cells_total::Int
    n_certified_cells::Int
    n_rescanned_cells::Int
    n_full_fallback_calls::Int
    n_rebuilds::Int
    total_cert_s::Float64
    total_full_s::Float64
end

"""
    PersistentWinnerCache(; tol_far=0.3)

Constructs an EMPTY persistent cache (`ref === nothing`) -- the reference is
built lazily on the first call to `winner_value_update!`/`lfix_value_certified`
(so construction never pays the O(W*D)-ish `build_winner_ref` cost until it is
actually needed). `tol_far` default 0.3 matches the "continuation" step-size
regime in `docs/winner_certificate_report.md`'s own measured table (Section 1:
still 89-95% certified at that magnitude, only falling to a full-scan-comparable
regime well beyond it) -- callers doing much larger jumps (e.g. a cold restart
at an arbitrary new candidate) should pass a smaller `tol_far` explicitly, or
just call `reset!` to force a fresh anchor at the new point.
"""
PersistentWinnerCache(; tol_far::Float64 = 0.3) =
    PersistentWinnerCache(nothing, tol_far, 0, 0, 0, 0, 0, 0, 0.0, 0.0)

"Force the next `winner_value_update!` call to rebuild the reference from scratch at its own point (e.g. after a large/rejected step, or when the caller knows the anchor is stale for a reason the tol_far guard alone would not catch)."
function reset!(wc::PersistentWinnerCache)
    wc.ref = nothing
    return wc
end

"Reset only the cumulative counters/timers (keeps the current reference -- use between benchmark phases without discarding a still-good anchor)."
function reset_counters!(wc::PersistentWinnerCache)
    wc.n_calls = 0; wc.n_cells_total = 0; wc.n_certified_cells = 0; wc.n_rescanned_cells = 0
    wc.n_full_fallback_calls = 0; wc.n_rebuilds = 0
    wc.total_cert_s = 0.0; wc.total_full_s = 0.0
    return wc
end

"""
    winner_value_update!(wc::PersistentWinnerCache, ctx, x_free'; threaded=false, rebuild_on_fallback=true) -> (winner', wval', CertStats)

THE core wiring primitive: get the EXACT winner matrix and winning-origin
sigma-value (`wval`, == pTsigma at the winner -- exactly what `winners_from_
certificate` produces) at a new nearby point `x_free'`, reusing `wc`'s
persistent `WinnerRefCache` anchor across calls instead of rebuilding one every
time. Builds the anchor lazily on first use (counted as a "rebuild"). On a
`tol_far` full-scan fallback (point too far from the current anchor -- see
`certified_winner_update`'s own docstring), the RETURNED winner/wval are still
EXACT (the fallback is itself a full trusted scan, just slower) -- and, if
`rebuild_on_fallback` (default true), the anchor is REBASED at the new point
so FUTURE nearby calls certify cheaply again. Rebasing is a PURE PERFORMANCE
policy: it never changes any returned value (every value returned by this
function, certified or fallback, is bit-identical to a full
`compute_winners_fast` scan by construction -- Section 1's proof, not
re-derived here), so cache use never depends on callback order or history,
per the standing brief's explicit correctness requirement.

Accumulates `wc`'s cumulative counters (certified/rescanned cell counts, full-
fallback call count, rebuild count, cert-path vs. full-path wall time) for
`winner_cache_report(wc)` to summarize.

`threaded`: forwards to `certified_winner_update_threaded`/`winners_from_
certificate_threaded` when true. Per this repo's own measured threading
discipline (docs/winner_certificate_report.md Section 6): only set `true` for
an ordinary STANDALONE value evaluation at large W*D (e.g. D=10/W=80000-ish),
called OUTSIDE an already-coordinate-threaded context. Leave `false` (default)
when called from inside `composite_gradient_fast.jl`'s `threaded=true`
per-coordinate loop, or at the production D=4/W=8000 scale where draw-level
threading is a wash-to-slight-loss (Context A's own measured table).
"""
function winner_value_update!(wc::PersistentWinnerCache, ctx, x_free′::AbstractVector;
        threaded::Bool = false, rebuild_on_fallback::Bool = true)
    if wc.ref === nothing
        t0 = time_ns()
        wc.ref = build_winner_ref(x_free′, ctx)
        wc.total_full_s += (time_ns() - t0) / 1e9
        wc.n_rebuilds += 1
    end

    t0 = time_ns()
    winner′, wval′, stats = threaded ?
        winners_from_certificate_threaded(wc.ref, ctx, x_free′; tol_far = wc.tol_far) :
        winners_from_certificate(wc.ref, ctx, x_free′; tol_far = wc.tol_far)
    elapsed = (time_ns() - t0) / 1e9

    wc.n_calls += 1
    wc.n_cells_total += stats.n_cells
    wc.n_certified_cells += stats.n_certified
    wc.n_rescanned_cells += stats.n_rescan
    if stats.fell_back_full
        wc.n_full_fallback_calls += 1
        wc.total_full_s += elapsed
        if rebuild_on_fallback
            t0 = time_ns()
            wc.ref = build_winner_ref(x_free′, ctx)
            wc.total_full_s += (time_ns() - t0) / 1e9
            wc.n_rebuilds += 1
        end
    else
        wc.total_cert_s += elapsed
    end
    return winner′, wval′, stats
end

"""
    winner_cache_report(wc::PersistentWinnerCache) -> NamedTuple

Summary of `wc`'s cumulative counters: certified/rescanned/cell fractions,
full-fallback-call fraction, rebuild count, and an implied speedup estimate
(total time actually spent vs. what `n_calls` full O(W*D^2)-ish scans would
have cost at the SAME per-call full-scan rate this cache itself measured via
its own fallback/rebuild calls -- a lower-bound estimate when few/no fallbacks
occurred, since then the "full-scan cost per call" has to be extrapolated from
the anchor-build cost alone; `benchmark_winner_accelerator.jl` cross-checks
this against a directly-measured always-full-scan baseline).
"""
function winner_cache_report(wc::PersistentWinnerCache)
    n = wc.n_calls
    cert_frac = wc.n_cells_total > 0 ? wc.n_certified_cells / wc.n_cells_total : NaN
    rescan_frac = wc.n_cells_total > 0 ? wc.n_rescanned_cells / wc.n_cells_total : NaN
    fallback_frac = n > 0 ? wc.n_full_fallback_calls / n : NaN
    total_s = wc.total_cert_s + wc.total_full_s
    mean_full_s_per_call = wc.n_rebuilds > 0 ? wc.total_full_s / wc.n_rebuilds : NaN
    implied_always_full_s = n > 0 && wc.n_rebuilds > 0 ? n * mean_full_s_per_call : NaN
    speedup = (isfinite(implied_always_full_s) && total_s > 0) ? implied_always_full_s / total_s : NaN
    return (n_calls = n, n_cells_total = wc.n_cells_total,
            certified_frac = cert_frac, rescanned_frac = rescan_frac,
            full_fallback_call_frac = fallback_frac, n_rebuilds = wc.n_rebuilds,
            total_cert_s = wc.total_cert_s, total_full_s = wc.total_full_s, total_s = total_s,
            mean_full_s_per_call = mean_full_s_per_call, implied_always_full_s = implied_always_full_s,
            implied_speedup = speedup)
end
