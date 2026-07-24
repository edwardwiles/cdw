# Gate A (exclude-ROW-destination UNRESTRICTED-CORE release, 2026-07-24): layout/algebra,
# solver-free-where-possible regression for the newly-rectangularized compressed-moment path
# (compressed_moments.jl/structured_moment_build.jl/compressed_live.jl/fast_range_screen.jl/
# infeasibility_screen.jl) and the new canonical cell-indexing helpers (cc_algo/active_layout.jl).
#
# Complements (does not duplicate) the pre-existing test_exclude_row_gateA_layout.jl (CM-family
# context/dimension checks, still passing unchanged) and test_compressed_moments.jl (D=4 square
# legacy compressed-vs-dense equivalence, still passing unchanged, run separately as the "legacy
# path is machine-identical to pre-change" check).
#
# Sections:
#   1. Canonical helper round-trip with a NON-LAST omission (D=4, omit destination 2 of 4) --
#      the one thing production's own real contexts (row_idx always == D, the last index) cannot
#      exercise, per this file's own explicit task requirement.
#   2. Real D=4/D_dest=3 rectangular context (last-position omission, via context_scaled.jl,
#      the only row_idx position the real data-loading pipeline supports) -- full inner solve,
#      compressed-vs-dense equivalence at a REAL post-solve dual (mirrors test_compressed_moments.jl
#      but genuinely rectangular).
#   3. Real D=20/:exclude_row construction-time equivalence (random theta_full, no inner solve --
#      construction/layout only, matches the existing Gate A's "small W, seconds not minutes" style).
#   4. Gravity pivot reconstruction a=b+Mz exact-to-machine-precision check, legacy square + both
#      rectangular contexts.
#
# Run: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#        full_aod_diag/d4_exact/test_exclude_row_unrestricted_gateA_layout.jl
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "context_scaled.jl"))
include(joinpath(@__DIR__, "context_real_d20.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
using Random, LinearAlgebra, Printf

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^96)
println("Section 1: canonical helper round-trip, NON-LAST omission (D=4, omit destination 2 of 4)")
println("="^96)
# Synthetic ctx-like object -- only the fields active_layout.jl's helpers touch. active_destinations
# deliberately skips 2 (NOT the last index), unlike every real production context in this repo.
ctx_nl = (D = 4, active_origins = 1:4, active_destinations = [1, 3, 4])
J_nl = length(active_destinations(ctx_nl))
check("active_destinations(ctx) has 3 entries (one destination omitted)", J_nl == 3)
check("omitted destination 2 is absent from active_destinations", !(2 in active_destinations(ctx_nl)))
check("dest_slot(ctx,1) == 1", dest_slot(ctx_nl, 1) == 1)
check("dest_slot(ctx,3) == 2 (slot 2, since global 2 is omitted)", dest_slot(ctx_nl, 3) == 2)
check("dest_slot(ctx,4) == 3", dest_slot(ctx_nl, 4) == 3)
check("global_destination(ctx,1) == 1", global_destination(ctx_nl, 1) == 1)
check("global_destination(ctx,2) == 3 (slot 2 maps to global country 3, skipping omitted 2)", global_destination(ctx_nl, 2) == 3)
check("global_destination(ctx,3) == 4", global_destination(ctx_nl, 3) == 4)
let threw = false
    try
        dest_slot(ctx_nl, 2)   # omitted destination -- must error, never silently return a slot
    catch e
        threw = e isa ErrorException
    end
    check("dest_slot(ctx, omitted_global_destination) errors (never a silent wrong answer)", threw)
end
# active_cell_index / active_cell_from_index round-trip over every (origin, slot) pair, and
# confirm no linear index ever aliases a cell that would touch the omitted destination.
let ok_roundtrip = true, seen = Set{Int}()
    for o in 1:4, s in 1:J_nl
        j = active_cell_index(ctx_nl, o, s)
        (o2, s2) = active_cell_from_index(ctx_nl, j)
        (o2 == o && s2 == s) || (ok_roundtrip = false)
        push!(seen, j)
    end
    check("active_cell_index/active_cell_from_index round-trip exactly for all 4*3=12 active cells", ok_roundtrip)
    check("active_cell_index produces exactly 12 distinct linear indices, 1..12, no gaps/collisions", seen == Set(1:12))
end
# active_cell_index_aod (origin-fast Aod-parameter convention) uses a DIFFERENT linear index than
# active_cell_index (destination-fast moments convention) whenever D != 1 and J != 1 -- confirm
# they actually differ (i.e. the test would catch the two conventions being silently conflated).
check("active_cell_index and active_cell_index_aod give DIFFERENT linear indices for (o=2,s=2) (D=4,J=3, so conventions must differ)",
      active_cell_index(ctx_nl, 2, 2) != active_cell_index_aod(ctx_nl, 2, 2))

println()
println("="^96)
println("Section 2: real D=4/D_dest=3 rectangular context (last-position omission) -- inner-solve equivalence")
println("="^96)
ctx_r = d_exact_setup_scaled(D = 4, W = 8000, find_smallest = true, row_idx = 4)
check("ctx_r.D_dest == 3 (one destination omitted)", ctx_r.D_dest == 3)
check("ctx_r.row_idx == 4", ctx_r.row_idx == 4)
Dr = ctx_r.D; Ddest_r = ctx_r.D_dest
pe_r = build_pivot_elimination(ctx_r)
Wr = size(ctx_r.U, 1)
x_free_from_w_r(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe_r))))
dense_G_r(θfull) = begin
    K = zeros(Wr); G = zeros(Wr, ctx_r.obj.d)
    ctx_r.obj.moments!(K, G, θfull, ctx_r.obj.U, ctx_r.obj)
    G
end

rng = MersenneTwister(20260724)
# Real calibration start point (matches unrestricted_stage_runner.jl's own calibration-mode
# construction: zfree0 = pivot_reduce(log(A_od_calibration), pe)) -- guaranteed gravity-feasible
# and a real solved point by construction, unlike an arbitrary random perturbation (which can
# legitimately be primal-infeasible for the inner dual solve, exactly as test_compressed_moments.jl's
# own random-perturbation candidates sometimes are -- that test SKIPs those, it does not treat inner
# solve failure as a code-correctness FAIL; mirrored here).
x_free_calib_r = ctx_r.θ0_up[ctx_r.free_idx]
zfree0_r = pivot_reduce(reshape(log.(x_free_calib_r[2:end]), Dr, Ddest_r), pe_r)
w0_r = vcat(x_free_calib_r[1], zfree0_r)
xf0_r = x_free_from_w_r(w0_r)
local base_r
try
    global base_r = solve_base_state(xf0_r, ctx_r)
    θfull_r = base_r.θ_full0
    Gr = dense_G_r(θfull_r)
    ncol_r = ctx_r.obj.outer_constr_index - 1
    Gfac_r = Gr[:, 1:ncol_r]

    cf_r = build_compressed_factual(θfull_r, ctx_r)
    check("cf.D == 4 (origins)", cf_r.D == 4)
    check("cf.D_dest == 3 (active destinations)", cf_r.D_dest == 3)
    check("cf.Pmat size == (4,3)", size(cf_r.Pmat) == (4, 3))
    check("cf.winner size == (W,3)", size(cf_r.winner) == (Wr, 3))

    Gc_r = materialize_dense_factual(cf_r)
    errA_r = maximum(abs.(Gc_r .- Gfac_r))
    check(@sprintf("materialize_dense_factual matches trusted dense obj.moments! (rectangular D=4/Ddest=3): max|diff|=%.2e < 1e-9", errA_r), errA_r < 1e-9)

    Gs_r = structured_dense_factual(cf_r)
    errS_r = maximum(abs.(Gs_r .- Gfac_r))
    check(@sprintf("structured_dense_factual (structured_moment_build.jl) matches trusted dense: max|diff|=%.2e < 1e-9", errS_r), errS_r < 1e-9)

    t_comp_r = compressed_dual_contraction(base_r.λstar, cf_r)
    t_dense_r = Gfac_r * base_r.λstar
    errB_r = maximum(abs.(t_comp_r .- t_dense_r))
    check(@sprintf("compressed_dual_contraction(lambda_star) matches dense dot-product: max|diff|=%.2e < 1e-9", errB_r), errB_r < 1e-9)

    q_dense_r = [-base_r.ζstar - dot(base_r.λstar, @view(Gr[s, 1:ncol_r])) for s in 1:Wr]
    q_comp_r = -base_r.ζstar .- compressed_dual_contraction(base_r.λstar, cf_r)
    errC_r = maximum(abs.(q_dense_r .- q_comp_r))
    check(@sprintf("draw-level q_omega (dual objective input) matches dense reference: max|diff|=%.2e < 1e-9", errC_r), errC_r < 1e-9)

    Lref_r = fixed_dual_L(xf0_r, ctx_r, base_r)
    Psi_q_r = similar(q_comp_r); CS.Psi!(Psi_q_r, q_comp_r)
    L_comp_r = -(sum(Psi_q_r) / Wr + base_r.ζstar)
    errL_r = abs(L_comp_r - Lref_r)
    check(@sprintf("dual objective value (mean Psi(q)+zeta) matches dense reference: |diff|=%.2e < 1e-9", errL_r), errL_r < 1e-9)

    # random-beta dual-gradient-direction check (weighted contraction at several beta -- proxy for
    # "dual gradient / weighted Hessian-vector product agree with dense reference", since both the
    # dual gradient (w.r.t. lambda) and an HVP are themselves weighted contractions of G against a
    # multiplier/weight vector, exactly what compressed_dual_contraction computes).
    errHVP = 0.0
    for trial in 1:5
        β = randn(rng, ncol_r)
        t_c = compressed_dual_contraction(β, cf_r)
        t_d = Gfac_r * β
        errHVP = max(errHVP, maximum(abs.(t_c .- t_d)))
    end
    check(@sprintf("weighted contraction (dual-gradient/HVP proxy) matches dense at 5 random weight vectors: max|diff|=%.2e < 1e-9", errHVP), errHVP < 1e-9)
catch e
    check("D=4/Ddest=3 rectangular base-state solve + equivalence chain completed without error ($(typeof(e)))", false)
    showerror(stdout, e); println()
end

println()
println("="^96)
println("Section 3: real D=20/:exclude_row construction-time equivalence (random theta_full, no inner solve)")
println("="^96)
ctx20 = d20_real_setup(W = 200, δ = 1.0, find_smallest = true)   # default -> :exclude_row
D20 = ctx20.D; Ddest20 = ctx20.D_dest; W20 = size(ctx20.U, 1)
rng20 = MersenneTwister(724)
θfull20 = copy(ctx20.θ0_up)
# perturb the free A-block (log-space, small, gravity-feasibility not required for this
# construction-only check -- build_compressed_factual/dense moments! both just evaluate at
# whatever theta_full they're given) and gamma_prime, leaving mu/sigma/fixed_vals untouched.
θfull20[ctx20.Aod_offset+1:ctx20.Aod_offset+D20*Ddest20] .*= exp.(0.02 .* randn(rng20, D20 * Ddest20))
θfull20[3+D20] = clamp(θfull20[3+D20] * (1 + 0.01 * randn(rng20)), ctx20.bounds.γp_lo, ctx20.bounds.γp_hi)

dense_G20 = begin
    K = zeros(W20); G = zeros(W20, ctx20.obj.d)
    ctx20.obj.moments!(K, G, θfull20, ctx20.obj.U, ctx20.obj)
    G
end
ncol20 = ctx20.obj.outer_constr_index - 1
try
    cf20 = build_compressed_factual(θfull20, ctx20; check_ties = true)
    check("cf20.D == 20", cf20.D == 20)
    check("cf20.D_dest == 19", cf20.D_dest == 19)
    check("cf20.Pmat size == (20,19)", size(cf20.Pmat) == (20, 19))
    Gc20 = materialize_dense_factual(cf20)
    errA20 = maximum(abs.(Gc20 .- dense_G20[:, 1:ncol20]))
    check(@sprintf("real D=20/:exclude_row materialize_dense_factual matches trusted dense obj.moments! at random theta_full: max|diff|=%.2e < 1e-8", errA20), errA20 < 1e-8)
    Gs20 = structured_dense_factual(cf20)
    errS20 = maximum(abs.(Gs20 .- dense_G20[:, 1:ncol20]))
    check(@sprintf("real D=20 structured_dense_factual matches trusted dense: max|diff|=%.2e < 1e-8", errS20), errS20 < 1e-8)
    β20 = randn(rng20, ncol20)
    errB20 = maximum(abs.(compressed_dual_contraction(β20, cf20) .- dense_G20[:, 1:ncol20] * β20))
    check(@sprintf("real D=20 compressed_dual_contraction matches dense at a random dual vector: max|diff|=%.2e < 1e-8", errB20), errB20 < 1e-8)
catch e
    check("real D=20/:exclude_row construction-time equivalence completed without error ($(typeof(e)))", false)
    showerror(stdout, e); println()
end

println()
println("="^96)
println("Section 4: gravity pivot reconstruction a=b+Mz exact to machine precision")
println("="^96)
function check_pivot_exact(label, ctx)
    pe = build_pivot_elimination(ctx)
    Ddest = hasproperty(ctx, :D_dest) ? ctx.D_dest : ctx.D
    n_free_here = ctx.D * Ddest - 1
    rng_g = MersenneTwister(99)
    zf = 0.1 .* randn(rng_g, n_free_here)
    z = pivot_expand(zf, pe)
    check("$label: pivot_expand output size == (D,D_dest)", size(z) == (ctx.D, Ddest))
    g_row = sum(pe.c .* vec(z)) + pe.g0
    check(@sprintf("%s: gravity row a=b+Mz exact to machine precision: |sum(c.*z)+g0|=%.2e < 1e-10", label, abs(g_row)), abs(g_row) < 1e-10)
    zf2 = pivot_reduce(z, pe)
    check("$label: pivot_reduce(pivot_expand(zf)) round-trips zf exactly", maximum(abs.(zf2 .- zf)) < 1e-12)
end
ctx4 = d4_exact_setup(find_smallest = true)
check_pivot_exact("D=4 legacy square", ctx4)
check_pivot_exact("D=4/Ddest=3 rectangular", ctx_r)
check_pivot_exact("D=20/Ddest=19 real rectangular", ctx20)

println()
if isempty(FAILURES)
    println(">>> RESULT: ALL PASS")
else
    println(">>> RESULT: ", length(FAILURES), " FAILURE(S): ", FAILURES)
    exit(1)
end
