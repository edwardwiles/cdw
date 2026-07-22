# ============================================================================
# Remediation task Part E, finding F5: forced-exact-tie test for the winner tie-break
# convention.
#
# Before this fix, the "top-3 update" comparison (`for o in Cd; if v < bs; bs=v;bo=o; end;
# end`, present identically in lfix_factorized.jl x2, lfix_kbplus.jl x2,
# lfix_factorized_workspace.jl, lfix_kbplus_workspace.jl, winner_certificate.jl,
# composite_gradient.jl, lfix_incremental.jl) kept whichever candidate was considered FIRST on
# an exact score tie -- the cached top-3 survivor (r1/r2/r3, any index) if one existed, else the
# first `o` encountered while iterating `Cd`. The "generic rescan" tiers (`for o in 1:D; v < bs
# && ...`, scanning from index 1 upward) already kept the LOWEST index on a tie, by construction
# of scanning in increasing order with a strict `<`. These two conventions disagreed whenever the
# survivor/first-considered candidate's index exceeded the tying changed-origin's index.
#
# The fix (`v < bs || (v == bs && o < bo)`) makes every top-3 site match the generic-rescan
# tiers' own "lowest index wins" convention. This is measure-zero in exact real-data arithmetic
# (two distinct origins scoring EXACTLY equal is astronomically unlikely with real tariffs/
# wages), so this test validates the COMPARISON IDIOM directly (isolated from the full
# production cache/moment machinery, which would need byte-identical origins to force a real
# tie) rather than reproducing a tie end-to-end through a real D=4/D=20 solve.
# ============================================================================
using Test

"Mirrors the exact comparison idiom now used at every top-3 winner-update site (see e.g.
lfix_factorized.jl's dest_contrib_incremental_top3_C): scan `candidates` (index => score) and
return the (score, index) that is either strictly lower, or -- on an exact tie -- has the lower
index. `bo`/`bs` seed from a possibly-higher-index 'survivor' candidate, exactly as the real
top-3 sites do (best_o/best_s from the cached r1/r2/r3 winner/runnerup/third, whichever is not in
the changed-origin set)."
function pick_lowest_index_on_tie(survivor_idx::Int, survivor_score::Float64,
                                   candidates::Vector{Tuple{Int,Float64}})
    bo, bs = survivor_idx, survivor_score
    for (o, v) in candidates
        if v < bs || (v == bs && o < bo)
            bs = v; bo = o
        end
    end
    return bo, bs
end

"Old (pre-fix) idiom, for a direct before/after contrast in the test output."
function pick_first_considered_on_tie(survivor_idx::Int, survivor_score::Float64,
                                       candidates::Vector{Tuple{Int,Float64}})
    bo, bs = survivor_idx, survivor_score
    for (o, v) in candidates
        if v < bs
            bs = v; bo = o
        end
    end
    return bo, bs
end

@testset "Winner tie-break: canonical lowest-origin-index convention (F5)" begin
    # Case 1: survivor has a HIGH index (7), a changed origin with a LOW index (2) ties it
    # exactly. New convention: origin 2 wins (lower index). Old convention: origin 7 (the
    # survivor, considered first) would have won -- this is exactly the disagreement F5 flagged.
    bo1, bs1 = pick_lowest_index_on_tie(7, 3.14159, [(2, 3.14159), (9, 5.0)])
    @test bo1 == 2
    @test bs1 == 3.14159
    bo1_old, _ = pick_first_considered_on_tie(7, 3.14159, [(2, 3.14159), (9, 5.0)])
    @test bo1_old == 7   # demonstrates the old idiom's disagreement, for contrast

    # Case 2: two changed origins tie each other exactly (survivor is strictly worse, not
    # tied) -- lower of the two tying indices wins.
    bo2, bs2 = pick_lowest_index_on_tie(1, 100.0, [(5, 2.5), (3, 2.5)])
    @test bo2 == 3
    @test bs2 == 2.5

    # Case 3: no tie at all -- strictly-lower score wins regardless of index (sanity check that
    # the tie-break clause doesn't perturb the non-tied case).
    bo3, bs3 = pick_lowest_index_on_tie(1, 100.0, [(5, 2.5), (3, 2.4)])
    @test bo3 == 3
    @test bs3 == 2.4

    # Case 4: three-way exact tie -- lowest of all three wins.
    bo4, bs4 = pick_lowest_index_on_tie(9, 0.0, [(9, 0.0), (4, 0.0), (6, 0.0)])
    @test bo4 == 4

    # Case 5: consistency with the generic-rescan convention -- scanning 1:D directly with a
    # plain strict `<` (no explicit tie-break needed, by construction of ascending order) must
    # agree with pick_lowest_index_on_tie on the SAME tied data.
    scores = Dict(1 => 5.0, 2 => 5.0, 3 => 5.0, 4 => 1.0, 5 => 1.0)
    bo_generic, bs_generic = 1, scores[1]
    for o in 2:5
        v = scores[o]
        v < bs_generic && (bs_generic = v; bo_generic = o)
    end
    bo_topk, bs_topk = pick_lowest_index_on_tie(1, scores[1], [(o, scores[o]) for o in 2:5])
    @test bo_generic == bo_topk == 4
    @test bs_generic == bs_topk == 1.0
end
println("All winner forced-tie tests passed.")
