# Phase "PROFILED AND ACTIVE CELL SETS": derive gravity_included / profiled_cell / active_cell
# masks strictly from the canonical MelitzGravitySample object (src/melitz/gravity_sample.jl),
# never by recreating the exclusion logic. Print exact label-based lists and assert counts.
REPO2 = "/bbkinghome/edav/gravity_robustness/worktrees/melitz-profile-38-nongravity-nonfocal-2026-08-01"
include(joinpath(REPO2, "misc", "doubleDiff.jl"))
include(joinpath(REPO2, "src", "melitz", "include_melitz.jl"))
using DelimitedFiles, Printf

real_dir = joinpath(REPO2, "real_data", "noah_D20")
countries = vec(readdlm(joinpath(real_dir, "countries.csv"), ',', String))
D = length(countries)
focal_label = "fra"
focal_origin = resolve_country_index(countries, focal_label; kind="focal country")

gs = melitz_gravity_sample(countries)  # canonical production default

gravity_included = gs.mask  # D x D BitMatrix, mask[o,d]
profiled_cell = falses(D, D)
active_cell = falses(D, D)
for d in 1:D, o in 1:D
    profiled_cell[o, d] = !gravity_included[o, d] && (o != focal_origin)
    active_cell[o, d] = gravity_included[o, d] || (o == focal_origin)
end

n_gravity_included = count(gravity_included)
n_profiled = count(profiled_cell)
n_active = count(active_cell)

@printf("count(gravity_included) = %d\n", n_gravity_included)
@printf("count(profiled_cell)    = %d\n", n_profiled)
@printf("count(active_cell)      = %d\n", n_active)

@assert n_gravity_included == 360 "gravity_included count mismatch: got $n_gravity_included"
@assert n_profiled == 38 "profiled_cell count mismatch: got $n_profiled"
@assert n_active == 362 "active_cell count mismatch: got $n_active"

println()
println("=== PROFILED CELLS (label o -> d), n=$(n_profiled) ===")
profiled_pairs = Tuple{String,String}[]
for d in 1:D, o in 1:D
    if profiled_cell[o, d]
        push!(profiled_pairs, (countries[o], countries[d]))
    end
end
# categorize by construction rule for the printed breakdown
row_idx = gs.row_index

n_row_nonfocal = 0
n_diag_nonfocal = 0
n_outlier_nonfocal = 0
for (o, d) in profiled_pairs
    global n_row_nonfocal, n_diag_nonfocal, n_outlier_nonfocal
    oi = resolve_country_index(countries, o)
    di = resolve_country_index(countries, d)
    if row_idx !== nothing && di == row_idx
        n_row_nonfocal += 1
    elseif oi == di
        n_diag_nonfocal += 1
    else
        n_outlier_nonfocal += 1
    end
    println("  $o -> $d")
end
println()
@printf("  nonfocal-origin ROW-destination cells: %d\n", n_row_nonfocal)
@printf("  additional nonfocal diagonal cells:    %d\n", n_diag_nonfocal)
@printf("  bilateral outlier (nonfocal) cells:    %d\n", n_outlier_nonfocal)
@assert n_row_nonfocal == 19 "expected 19 nonfocal ROW-dest cells, got $n_row_nonfocal"
@assert n_diag_nonfocal == 18 "expected 18 nonfocal diagonal cells, got $n_diag_nonfocal"
@assert n_outlier_nonfocal == 1 "expected 1 outlier cell, got $n_outlier_nonfocal"

println()
println("=== ACTIVE-BUT-NOT-GRAVITY-INCLUDED CELLS (focal-origin add-backs) ===")
addback = Tuple{String,String}[]
for d in 1:D, o in 1:D
    if active_cell[o, d] && !gravity_included[o, d]
        push!(addback, (countries[o], countries[d]))
        println("  $(countries[o]) -> $(countries[d])")
    end
end
@assert length(addback) == 2 "expected exactly 2 focal add-back cells, got $(length(addback))"
@assert (focal_label, "row") in addback || (focal_label, countries[row_idx]) in addback "France->ROW missing"
@assert (focal_label, focal_label) in addback "France->France missing"

println()
println("ALL ASSERTIONS PASSED.")
println("focal_origin index = $focal_origin ($(countries[focal_origin]))")
println("row_index = $row_idx ($(row_idx === nothing ? "none" : countries[row_idx]))")
println("gs.fingerprint = $(gs.fingerprint)")
