# ============================================================================
# Claude Code task 2026-08-01, §4 gate: ProfiledEconomicMomentLayout at D=4
# (symmetric, square). Confirms live dimensions (16 -> 12 retained -> 13
# total with France ratio), no anchor collisions, structural assertions pass.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))

ctx = d4_exact_setup()
D = ctx.D; Ddest = D
θ0 = copy(ctx.θ0_up)

println("="^78); println("TEST 1: build_anchor_spec_from_ctx matches default_anchor_spec (own-cell + one override)"); println("="^78)
spec_old = default_anchor_spec(D, Ddest; overrides = Dict(3 => 1))
spec_new = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
@assert spec_old.anchor_origin == spec_new.anchor_origin "anchor_origin mismatch: $(spec_old.anchor_origin) vs $(spec_new.anchor_origin)"
println("PASS -- anchor_origin identical: $(spec_new.anchor_origin)")

cf = build_compressed_factual(θ0, ctx; check_ties = false)
has_france = cf.cf_col > 0
println("cf.cf_col=$(cf.cf_col) has_france=$has_france")

println("\n" * "="^78); println("TEST 2: layout dimensions"); println("="^78)
layout = build_profiled_economic_moment_layout(ctx, spec_new; has_france_ratio = has_france)
println("D=$(layout.D) Ddest=$(layout.Ddest)")
println("full factual moments = $(D*Ddest)")
println("retained factual moments = $(length(layout.retained_full_factual_j))")
println("france_ratio_reduced_j = $(layout.france_ratio_reduced_j)")
println("total_reduced_economic_moments = $(layout.total_reduced_economic_moments)")
@assert D * Ddest == 16
@assert length(layout.retained_full_factual_j) == 12
@assert layout.total_reduced_economic_moments == (has_france ? 13 : 12)
println("PASS -- 16 -> 12 retained -> $(layout.total_reduced_economic_moments) total (matches task's stated D=4 figures)")

println("\n" * "="^78); println("TEST 3: exactly one omitted moment per destination, anchor cell == outer-A anchor cell"); println("="^78)
for s in 1:Ddest
    o_anchor = spec_new.anchor_origin[s]
    @assert reduced_index(layout, o_anchor, s) == 0 "anchor cell (o=$o_anchor,slot=$s) unexpectedly has a reduced index"
    @assert is_anchor_cell(layout, o_anchor, s)
    n_retained_here = count(==(s), layout.retained_slot)
    @assert n_retained_here == D - 1 "slot $s has $n_retained_here retained moments, expected D-1=$(D-1)"
end
println("PASS -- every destination has exactly one omitted (anchor) moment and D-1 retained moments")

println("\n" * "="^78); println("TEST 4: no duplicates, no gaps in reduced index space"); println("="^78)
@assert sort(layout.reduced_to_full_factual) == layout.retained_full_factual_j
@assert Set(layout.full_factual_to_reduced[layout.retained_full_factual_j]) == Set(1:length(layout.retained_full_factual_j))
println("PASS")

println("\n" * "="^78); println("TEST 5: structural price-index/anchor-moment assertions"); println("="^78)
assert_no_factual_price_index_moment(layout)
println("PASS -- assert_no_factual_price_index_moment succeeds on the well-formed layout")

# Negative control: a deliberately malformed layout (fake extra column claiming a full D count)
# must be REJECTED by the assertion, not silently accepted.
bad_layout = ProfiledEconomicMomentLayout(layout.D, layout.Ddest, layout.destination_ids,
    layout.anchor_origin_by_slot, vcat(layout.retained_full_factual_j, [layout.anchor_origin_by_slot[1]]),
    vcat(layout.retained_origin, [layout.anchor_origin_by_slot[1]]), vcat(layout.retained_slot, [1]),
    layout.full_factual_to_reduced, vcat(layout.reduced_to_full_factual, [layout.anchor_origin_by_slot[1]]),
    layout.france_ratio_reduced_j, layout.total_reduced_economic_moments)
threw = false
try
    assert_no_factual_price_index_moment(bad_layout)
catch e
    global threw = true
    println("negative control correctly threw: ", sprint(showerror, e))
end
@assert threw "negative control FAILED to reject a malformed (D-count) layout"
println("PASS -- malformed layout correctly rejected")

println("\n" * "="^78); println("ALL TESTS PASSED"); println("="^78)
