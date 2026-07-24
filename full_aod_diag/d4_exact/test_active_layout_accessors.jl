# Part B step 3 (2026-07-24 release): unit tests for active_origins/active_destinations/
# active_od_cells (cc_algo/active_layout.jl). Pure accessor tests against synthetic NamedTuples
# -- no context/gravity/KNITRO machinery needed, since these functions are generic over anything
# with a `.D` field (and optionally `.active_origins`/`.active_destinations`).
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_active_layout_accessors.jl
include(joinpath(@__DIR__, "..", "..", "cc_algo", "include_cc_algo.jl"))
using .CounterfactualSensitivity: active_origins, active_destinations, active_od_cells

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Section 1: default (pre-omit-ROW) square context -- falls through to 1:D")
println("="^78)

ctx_square = (D = 4,)
check("active_origins defaults to 1:D", collect(active_origins(ctx_square)) == [1, 2, 3, 4])
check("active_destinations defaults to 1:D", collect(active_destinations(ctx_square)) == [1, 2, 3, 4])
check("active_od_cells has D^2=16 cells for the default square case",
      length(collect(active_od_cells(ctx_square))) == 16)
check("active_od_cells contains (1,1) and (4,4) for the square default",
      (1, 1) in collect(active_od_cells(ctx_square)) && (4, 4) in collect(active_od_cells(ctx_square)))

println()
println("="^78)
println("Section 2: synthetic NON-SQUARE mask -- proves the accessor layer itself is")
println("           already rectangular-ready, independent of Part A's omit-ROW work")
println("="^78)

# D=4 origins, but destination 3 (e.g. a dropped ROW column) omitted from the active set.
ctx_nonsquare = (D = 4, active_origins = [1, 2, 3, 4], active_destinations = [1, 2, 4])

check("active_origins reads the explicit field when present",
      collect(active_origins(ctx_nonsquare)) == [1, 2, 3, 4])
check("active_destinations reads the explicit (reduced) field when present",
      collect(active_destinations(ctx_nonsquare)) == [1, 2, 4])
check("omitted destination 3 does not appear in active_destinations",
      !(3 in collect(active_destinations(ctx_nonsquare))))

cells = collect(active_od_cells(ctx_nonsquare))
check("active_od_cells has 4 origins x 3 active destinations = 12 cells (not 16)", length(cells) == 12)
check("no cell in active_od_cells references the omitted destination 3",
      all(d != 3 for (o, d) in cells))
check("active_od_cells still contains a valid non-omitted cell, e.g. (2,4)", (2, 4) in cells)

println()
println("="^78)
println("Section 3: origin-side reduction (symmetry check -- accessor is not destination-only)")
println("="^78)
ctx_origin_reduced = (D = 4, active_origins = [1, 3], active_destinations = [1, 2, 3, 4])
cells2 = collect(active_od_cells(ctx_origin_reduced))
check("active_od_cells has 2 active origins x 4 destinations = 8 cells", length(cells2) == 8)
check("no cell references omitted origin 2 or 4", all(o in (1, 3) for (o, d) in cells2))

println()
n_fail = length(FAILURES)
n_total = n_fail == 0 ? "all" : "some"
println("="^78)
println(">>> RESULT: ", n_fail == 0 ? "ALL PASS" : "$(n_fail) FAILURE(S): $(join(FAILURES, ", "))")
println("="^78)
n_fail == 0 || error("test_active_layout_accessors.jl: $(n_fail) assertion(s) FAILED")
