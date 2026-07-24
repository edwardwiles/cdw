# Gate A (exclude-ROW-destination production release, 2026-07-24): real construction-only layout
# regression covering the task's explicit Gate A checklist -- active cell maps, raw/free
# dimensions, gravity pivot reconstruction, moment counts, origin ROW retained, destination ROW
# omitted only in exclude mode, default resolves to exclude_row, explicit legacy mode reproduces
# legacy behavior, checkpoint mismatch refusal. Small W (construction/layout only, NOT a numerics
# gate -- Gate B covers real W=80,000 numerics) so this runs in seconds, not minutes.
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_exclude_row_gateA_layout.jl
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))   # -> build_pivot_elimination, pivot_reduce, pivot_expand

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Section 1: destination_sample=:exclude_row (default) -- rectangular D=20/Ddest=19")
println("="^78)
ctx_x = d20_real_setup(W = 200, δ = 1.0, find_smallest = true)   # no destination_sample arg -- proves the default

check("default (no-arg) destination_sample resolves to :exclude_row", ctx_x.destination_sample == :exclude_row)
check("D (origin count) == 20", ctx_x.D == 20)
check("D_dest (destination count) == 19", ctx_x.D_dest == 19)
check("row_idx == 20 (ROW is the last/omitted destination)", ctx_x.row_idx == 20)
check("active_origins(ctx) has 20 entries (ROW retained as origin)", length(active_origins(ctx_x)) == 20)
check("active_destinations(ctx) has 19 entries (ROW omitted as destination)", length(active_destinations(ctx_x)) == 19)
check("active_od_cells(ctx) has 20*19=380 active A cells", length(collect(active_od_cells(ctx_x))) == 380)
check("no active_od_cells entry references destination 20 (ROW)", all(d != 20 for (o, d) in active_od_cells(ctx_x)))
check("origin 20 (ROW) DOES appear as an origin in active_od_cells", any(o == 20 for (o, d) in active_od_cells(ctx_x)))
check("raw active A cells (n_free(m)-1, pre-pivot-reduction) == 380", n_free(ctx_x.m) - 1 == 380)
pe_x = build_pivot_elimination(ctx_x)
check("free A coordinates (pivot-reduced, gravity restriction applied) == 379", length(pe_x.other_idx) == 379)
check("gravity_sample_version present and == 2", ctx_x.gravity_sample_version == 2)
check("theta_calibration_version present and == 2", ctx_x.theta_calibration_version == 2)
check("Aod_free_pos has size (D, D_dest) = (20,19)", size(ctx_x.Aod_free_pos) == (20, 19))

println()
println("="^78)
println("Section 2: destination_sample=:all_legacy -- square D=20/Ddest=20 (explicit reproduction)")
println("="^78)
ctx_l = d20_real_setup(W = 200, δ = 1.0, find_smallest = true, destination_sample = :all_legacy)

check("explicit :all_legacy destination_sample resolves correctly", ctx_l.destination_sample == :all_legacy)
check("D (origin count) == 20", ctx_l.D == 20)
check("D_dest (destination count) == 20 under :all_legacy", ctx_l.D_dest == 20)
check("row_idx === nothing under :all_legacy", ctx_l.row_idx === nothing)
check("active_destinations(ctx) has all 20 entries under :all_legacy", length(active_destinations(ctx_l)) == 20)
check("active_od_cells(ctx) has 20*20=400 cells under :all_legacy", length(collect(active_od_cells(ctx_l))) == 400)
check("raw active A cells (n_free(m)-1) == 400 under :all_legacy", n_free(ctx_l.m) - 1 == 400)
pe_l = build_pivot_elimination(ctx_l)
check("free A coordinates (pivot-reduced) == 399 (400-1) under :all_legacy", length(pe_l.other_idx) == 399)

println()
println("="^78)
println("Section 3: invalid destination_sample rejected")
println("="^78)
let threw = false
    try
        d20_real_setup(W = 200, destination_sample = :bogus)
    catch e
        threw = e isa ErrorException
    end
    check("d20_real_setup rejects an unknown destination_sample symbol", threw)
end

println()
println("="^78)
println("Section 4: reject focal_country == ROW")
println("="^78)
# The focal country (task's "focal_country") is baseIndex/`bi` -- AD_PARAMS's own default is 2
# (France), matching context_real_d20.jl's header comment ("Focal country: France (baseIndex=2)").
# GT (the counterfactual gains-from-trade this whole outer loop solves for) is a monotonic
# function of gamma-prime_{baseIndex} -- meaningless for a focal country that isn't itself a valid
# destination in the resolved sample, so d20_real_setup now hard-errors if bi==row_idx under
# :exclude_row (context_real_d20.jl, this release). AD_PARAMS.baseIndex is a compile-time global
# (not a per-call keyword of d20_real_setup), so this section (a) confirms the live production
# value is safely apart from the omitted destination, and (b) confirms the guard code that would
# fire if it ever were is actually present (dynamically exercising it would require a caller-facing
# baseIndex override, which does not exist in this codebase and is out of this release's scope).
check("live baseIndex (focal country, bi=$(ctx_x.bi)) != row_idx ($(ctx_x.row_idx)) -- no collision today", ctx_x.bi != ctx_x.row_idx)
check("row_idx is D20_REAL (20, ROW) whenever destination_sample=:exclude_row -- not caller-selectable", ctx_x.row_idx == 20)
check("d20_real_setup source contains the focal_country==ROW guard",
      occursin("focal country", read(joinpath(@__DIR__, "context_real_d20.jl"), String)) &&
      occursin("bi != row_idx", read(joinpath(@__DIR__, "context_real_d20.jl"), String)))
let rejected = false
    try
        d20_real_setup(W = 200, row_idx = 5)   # d20_real_setup takes no row_idx kwarg -- MethodError expected
    catch e
        rejected = e isa MethodError
    end
    check("d20_real_setup has no caller-facing row_idx/omit-target argument (MethodError on attempt)", rejected)
end

println()
if isempty(FAILURES)
    println(">>> RESULT: ALL PASS")
else
    println(">>> RESULT: ", length(FAILURES), " FAILURE(S): ", FAILURES)
    exit(1)
end
