# Continuation 13, Section 6: genuinely nested quantile-probability grids.
#
# ⚠️ NO LONGER THE PRODUCTION GRID (2026-08-12). The live CM/CM+ZC/Common-Fréchet campaign path
# (`resolve_cm_probs` -> `paper_upper_v1_orchestrator/family_start_chain.jl`) now resolves to
# `cm_equal_mass_probs(L)` (common_marginals_moments.jl): `k/L` for `k=1..L-1`, i.e. exactly `L`
# buckets of mass `1/L`, per the user's specification that "L = 50 means 50 equally sized buckets".
# This file's dyadic grid is NOT equal-mass -- measured at L=50 it gives 50 levels, hence 51
# buckets, whose masses are only ever 0.015625 or 0.03125.
#
# It is kept, unchanged, because the property below is real and is a different property: genuine
# NESTING across L. Equal-mass grids nest only when the sizes divide (Q_10 ⊂ Q_20, Q_10 ⊂ Q_50, but
# Q_20 ⊄ Q_50 since 20 ∤ 50), so anything that needs a nested L-ladder over {10,20,50} specifically
# -- `c13_d20_cm_upper_continuation.jl`'s warm-start continuation, `CMConfig`'s `:nested_family`
# rule and `cm_production_stage_runner.jl` -- still calls `nested_grid_sequence` directly and is
# untouched. For a nested EQUAL-MASS ladder, choose sizes that divide (e.g. {10,50}).
#
# The Continuation 12 L=10/20/50 grids used `k/L` for k=1:L-1. This gives L=10's 9 cutpoints as
# an exact subset of L=20's 19 (both multiples of 0.05) -- but L=50's 49 cutpoints (multiples of
# 0.02) are NOT a superset of L=20's 19 (0.05 is not a multiple of 0.02), confirmed directly in
# docs/fullA_common_marginals_handoff.md Section 5. That non-nestedness is flagged there as a
# plausible contributor to the observed non-monotonic kappa(L) (0.1522 < 0.1552 < 0.1564), since a
# clean nested-nested-COMPARISON reading of "more restrictions -> weakly lower kappa" requires
# actual nesting, not merely similarly-sized nominal grids.
#
# This file builds a DETERMINISTIC, genuinely nested sequence of probability sets via largest-gap
# bisection: starting from the trivial 2-point boundary {0,1} (not real cutpoints, just the unit
# interval's endpoints), repeatedly insert the midpoint of the CURRENT largest gap. Because points
# are only ever ADDED, never removed or reflowed, every earlier state is a strict subset of every
# later one by construction -- Q_10 ⊂ Q_20 ⊂ Q_50 holds trivially, not merely for this specific
# target-size choice.

"""
    largest_gap_bisection_grid(n_target::Int) -> Vector{Float64}

Returns `n_target` probabilities in (0,1), built by repeated largest-gap bisection starting from
the boundary {0,1}. Deterministic (ties broken by lowest-left-endpoint order, itself deterministic
given floating point). Sorted ascending on return.
"""
function largest_gap_bisection_grid(n_target::Int)
    pts = Float64[0.0, 1.0]
    while length(pts) - 2 < n_target   # -2: the two boundary points are not themselves cutpoints
        sort!(pts)
        gaps = diff(pts)
        i = argmax(gaps)   # first-occurrence argmax -- deterministic tie-break
        mid = (pts[i] + pts[i+1]) / 2
        push!(pts, mid)
    end
    sort!(pts)
    return pts[2:end-1]   # drop the 0/1 boundary sentinels
end

# NOTE on the sort-then-slice discipline below: `pts[2:end-1]` is only correct as "drop the two
# boundary sentinels" immediately AFTER a fresh `sort!` -- a bare `push!` appends unsorted, so
# slicing before re-sorting would drop the just-inserted point (usually still mid-array) instead
# of the true 1.0 sentinel (left stranded wherever it last was), corrupting the snapshot. Caught
# directly: an unfixed first draft of `nested_grid_sequence` returned literal `1.0` as one of
# Q_10's "cutpoints" -- passing `1.0` to `quantile(U,probs)` as a CDF threshold is degenerate
# (trivially true for every draw). Every snapshot below re-sorts a FRESH COPY before slicing.

"""
    nested_grid_sequence(sizes::Vector{Int}) -> Dict{Int,Vector{Float64}}

Builds ONE bisection sequence out to `maximum(sizes)` points, then returns the prefix state at
each requested size -- guarantees Q_{sizes[1]} ⊂ Q_{sizes[2]} ⊂ ... by construction, not merely by
post-hoc checking. This is the ONLY correct way to get genuine nesting from this algorithm: calling
`largest_gap_bisection_grid` independently at each size would NOT nest, since the largest-gap choice
at each independent call depends on the full target count reached, and different target counts can
produce a different bisection ORDER for early points (they don't, actually, for THIS specific
algorithm since each step only depends on the CURRENT point set, not the target -- but the sequence
form below removes any doubt and is the auditable, obviously-correct construction).
"""
function nested_grid_sequence(sizes::Vector{Int})
    nmax = maximum(sizes)
    pts = Float64[0.0, 1.0]
    snapshots = Dict{Int,Vector{Float64}}()
    for step in 1:nmax
        sort!(pts)
        gaps = diff(pts)
        i = argmax(gaps)
        mid = (pts[i] + pts[i+1]) / 2
        push!(pts, mid)
        if step in sizes
            snapshots[step] = sort(pts)[2:end-1]
        end
    end
    return snapshots
end

"""
    quantiles_from_probs(U_ref::AbstractVector{Float64}, probs::Vector{Float64}) -> Vector{Float64}

Converts a probability grid to actual quantile CUTPOINTS of the reference origin's draws -- the
same `quantile(U[:,refIndex1], probs)` convention `common_marginals_quantiles`/
`precalc_common_marginals_cdf` already use, so these probabilities plug directly into the existing
CM machinery (just pass `probs` in place of the `k/L` grid those functions build internally --
see `precalc_common_marginals_cdf`'s `range(1/L,(L-1)/L,length=L)` call site, which this file's
probs are a drop-in, EXPLICIT replacement for once threaded through).
"""
quantiles_from_probs(U_ref::AbstractVector{Float64}, probs::Vector{Float64}) = quantile(U_ref, probs)

using Printf

if abspath(PROGRAM_FILE) == @__FILE__
    snaps = nested_grid_sequence([10, 20, 50])
    for n in (10, 20, 50)
        p = snaps[n]
        @printf "Q_%d (%d cutpoints):\n" n length(p)
        println("  ", round.(p, digits=6))
    end
    println()
    println("Nesting check:")
    println("  Q10 ⊂ Q20 : ", issubset(Set(snaps[10]), Set(snaps[20])))
    println("  Q20 ⊂ Q50 : ", issubset(Set(snaps[20]), Set(snaps[50])))
    println("  Q10 ⊂ Q50 : ", issubset(Set(snaps[10]), Set(snaps[50])))
    println()
    println("Old (non-nested) k/L convention, for contrast:")
    old10 = collect(1:9) ./ 10
    old20 = collect(1:19) ./ 20
    old50 = collect(1:49) ./ 50
    println("  old Q10 ⊂ old Q20 : ", issubset(Set(old10), Set(old20)))
    println("  old Q20 ⊂ old Q50 : ", issubset(Set(old20), Set(old50)))
end
