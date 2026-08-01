# ============================================================================
# Claude Code task 2026-08-01, §10: gravity-pivot invariance gate.
# (a) an anchor scale cannot be perturbed in the profiled outer coordinate
#     vector at all -- structurally, not by a runtime check (no r_free index
#     maps to an anchor cell).
# (b) inserting an arbitrary COMMON destination-scale shift into the anchor
#     gauge leaves the gravity residual (offset_r0/cr, hence the pivot
#     reconstruction) unchanged, verified live via finite differences, not
#     cited from the theory doc.
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(dirname(dirname(@__DIR__)), "cc_algo", "active_layout.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "operator_psi_bundle.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "relative_a_coordinate_2026-07-31.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "gravity_pivot_on_retained_2026-07-31.jl"))
include(joinpath(@__DIR__, "outer_coordinate_layout_profiled_2026-07-31.jl"))
include(joinpath(@__DIR__, "profiled_economic_moment_layout_2026-08-01.jl"))
using LinearAlgebra, Printf

ctx0 = d4_exact_setup()
ctx = build_unrestricted_operator_ctx(ctx0)
D = ctx.D
θ0 = copy(ctx.θ0_up)
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(3 => 1))
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)

println("="^90); println("(a) anchor cells structurally absent from r_free"); println("="^90)
anchor_lin = anchor_linear_indices(spec)
retained_lin = retained_linear_indices(spec)
@assert isempty(intersect(anchor_lin, retained_lin)) "anchor cells must be disjoint from retained cells"
ridx_at_free_positions = [retained_lin[pos] for pos in pe.other_pos]  # the n_free cells actually reachable via r_free
@assert isempty(intersect(anchor_lin, ridx_at_free_positions)) "an anchor cell leaked into r_free's reachable set"
println("anchor_linear_indices = $anchor_lin")
println("n_retained=$(length(retained_lin))  n_free (r_free)=$(length(pe.other_pos))  pivot cell = $(retained_lin[pe.pivot_pos])")
println("PASS: no r_free index maps to an anchor cell (structural, ", length(anchor_lin), " anchors, 0 overlap)")
flush(stdout)

println("\n" * "="^90); println("(b) common destination-scale shift leaves offset_r0/cr (hence the pivot) unchanged"); println("="^90)
Random_seed = 2026
using Random
Random.seed!(Random_seed)
Δ = 0.37 * randn()  # arbitrary common shift applied to EVERY destination's anchor gauge
gauge_shifted = gauge .+ Δ
pe_shifted = build_pivot_elimination_on_retained(ctx, spec, gauge_shifted)

d_offset = abs(pe_shifted.offset_r0 - pe.offset_r0)
d_cr = maximum(abs.(pe_shifted.cr .- pe.cr))
d_pivot_pos = pe_shifted.pivot_pos == pe.pivot_pos
println(@sprintf("Delta=%.6f  |offset_r0 diff|=%.3e  max|cr diff|=%.3e  same pivot cell=%s", Δ, d_offset, d_cr, d_pivot_pos))
@assert d_offset < 1e-10 "offset_r0 changed under a common destination-scale shift -- gravity restriction is NOT invariant to omitted common scales"
@assert d_cr < 1e-10 "cr (gravity coefficients in r-space) changed under a common destination-scale shift"
@assert d_pivot_pos "pivot cell selection changed under a common destination-scale shift"
println("PASS: offset_r0/cr/pivot cell all EXACTLY invariant (to floating-point) under an arbitrary common destination-scale insertion")
flush(stdout)

println("\n" * "="^90); println("(b-2) live finite-difference confirmation: gravity residual at a decoded point is unaffected by shifting the gauge"); println("="^90)
r_free_test = randn(length(pe.other_pos)) .* 0.05
z_from_pe = decode_full_z_on_retained(r_free_test, pe)
z_from_pe_shifted = decode_full_z_on_retained(r_free_test, pe_shifted)
grav_orig = gravity_from_logz(z_from_pe, ctx)
grav_shifted = gravity_from_logz(z_from_pe_shifted, ctx)
println(@sprintf("gravity_from_logz(decode(pe))=%.6e   gravity_from_logz(decode(pe_shifted))=%.6e   |diff|=%.3e", grav_orig, grav_shifted, abs(grav_orig - grav_shifted)))
@assert abs(grav_orig - grav_shifted) < 1e-9 "gravity residual differs after a common destination-scale gauge shift -- SAME r_free must decode to a gravity-feasible point regardless of gauge choice"
println("PASS: gravity residual invariant to gauge choice, confirmed at a live decoded point (not just via offset_r0/cr)")
flush(stdout)

println("\nGRAVITY PIVOT INVARIANCE GATE: PASS")
