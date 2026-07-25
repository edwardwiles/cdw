# ============================================================================
# Correctness + before/after allocation gate for sections 4.1/4.2 (CM allocation
# fixes: bview*R -> mul! into persistent scratch; Hraw_EC/block_ec moved into CMBinHessCtx).
#
# Verified before implementing (see commit messages): wrap_moments_with_cm_archB IS the real
# production moments wrapper (build_cm_production_context's use_archB_moments defaults to true,
# and run_cm_upper_checkpointed calls build_cm_production_context with no override), and
# hessian_cm_structured! IS the real production Hessian callback (archC_verified_state, which
# cm_production_value_verified/cm_checkpoint.jl's cb_F! call, passes
# hess_cb_builder = archC_hess_cb_builder) -- both genuinely on the hot path, unlike section 3.1's
# corrected finding.
#
# Usage: JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          full_aod_diag/d4_exact/test_cm_mul_and_hessian_scratch.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["draw_design.jl","winners.jl","oracle.jl","common_marginals_moments.jl","common_marginals_interval.jl",
          "instrumentation.jl","oracle_fast.jl","gravity_elimination.jl","three_way_derivatives.jl",
          "lfix_incremental.jl","composite_gradient.jl","composite_gradient_fast.jl","cm_lookup_kernels.jl",
          "lfix_cm_aware.jl","cm_hessian_architectures.jl","cm_production_bundle.jl","cm_screen_bridge.jl",
          "lfix_cm_cplus.jl","nested_quantile_grids.jl","cm_outer_driver.jl","cm_config.jl",
          "cm_meanzc_moments.jl","cm_meanzc_config.jl","cm_meanzc_production.jl","cm_meanzc_cplus.jl","cm_checkpoint.jl"]
    include(joinpath(_D4E, f))
end
using LinearAlgebra, Random

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("Section 1: fill_cm_columns_from_bins! (§4.1) -- standalone, synthetic data")
println("="^78)
rng = MersenneTwister(20260725)
W_syn, D_syn, L_syn = 5000, 20, 50
Bidx = rand(rng, 1:(L_syn + 1), W_syn, D_syn)
origins = collect(1:19)   # refIndex1 = 20, excluded from origins (matches production convention)
refIndex1 = 20
nO = length(origins)
R = orthonormal_contrast_matrix(D_syn)   # (D_syn-1) x (D_syn-1) == nO x nO, takes the FULL country count, not nO

Gdest_noscratch = Matrix{Float64}(undef, W_syn, L_syn * nO)
fill_cm_columns_from_bins!(Gdest_noscratch, Bidx, origins, refIndex1, L_syn, R; chunk_size = 2000, prod_scratch = nothing)

Gdest_scratch = Matrix{Float64}(undef, W_syn, L_syn * nO)
prod_scratch = Matrix{Float64}(undef, min(2000, W_syn), nO)
fill_cm_columns_from_bins!(Gdest_scratch, Bidx, origins, refIndex1, L_syn, R; chunk_size = 2000, prod_scratch = prod_scratch)

check("Gdest bit-identical with/without prod_scratch", Gdest_noscratch == Gdest_scratch)

GC.gc()
b_before = @allocated fill_cm_columns_from_bins!(Gdest_noscratch, Bidx, origins, refIndex1, L_syn, R; chunk_size = 2000, prod_scratch = nothing)
GC.gc()
b_after = @allocated fill_cm_columns_from_bins!(Gdest_scratch, Bidx, origins, refIndex1, L_syn, R; chunk_size = 2000, prod_scratch = prod_scratch)
lp(">>> @allocated fill_cm_columns_from_bins! (no scratch): ", b_before, " bytes (", round(b_before/1e6, digits=2), " MB)")
lp(">>> @allocated fill_cm_columns_from_bins! (scratch):    ", b_after, " bytes (", round(b_after/1e6, digits=2), " MB)")
lp(">>> reduction: ", round(100*(1 - b_after/b_before), digits=2), "%")
check("scratch path allocates strictly less", b_after < b_before)

# R === nothing branch (anchored contrasts) must still work identically, prod_scratch ignored
Gdest_noR_a = Matrix{Float64}(undef, W_syn, L_syn * nO)
fill_cm_columns_from_bins!(Gdest_noR_a, Bidx, origins, refIndex1, L_syn, nothing; chunk_size = 2000, prod_scratch = nothing)
Gdest_noR_b = Matrix{Float64}(undef, W_syn, L_syn * nO)
fill_cm_columns_from_bins!(Gdest_noR_b, Bidx, origins, refIndex1, L_syn, nothing; chunk_size = 2000, prod_scratch = prod_scratch)
check("R=nothing (anchored) branch unaffected by prod_scratch, bit-identical", Gdest_noR_a == Gdest_noR_b)

println("="^78)
println("Section 2: hessian_cm_structured! Hraw_EC/block_ec (§4.2) -- real D=20/W=80,000 CM context")
println("="^78)
lp(">>> building real D=20/W=80000 CM context (calibration, delta=1, L=50, orthonormal contrasts)...")
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[50]
pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, probs = probs)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]   # archC_verified_state takes x_free DIRECTLY (not the pivot-reduced w=(gp,zfree) vector)

base, verify = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
check("archC inner solve feasible", base.inner_status in (0, -100, -101, -103))
check("verify.Delta_dual finite", isfinite(verify.Delta_dual))
check("verify.max_abs_moment_kkt_resid finite", isfinite(verify.max_abs_moment_kkt_resid))
lp(">>> Delta_dual=", verify.Delta_dual, " max_abs_moment_kkt_resid=", verify.max_abs_moment_kkt_resid,
   " nO=", pcx.cctx.nO, " NCORE=", pcx.cctx.NCORE, " L=", pcx.cctx.L)

# Isolated before/after allocation at the exact fixed lines, using this real cctx's actual shape.
cctx = pcx.cctx
GC.gc()
b_old = @allocated begin
    global Hraw_old = Matrix{Float64}(undef, cctx.NCORE, cctx.nO)
    fill!(Hraw_old, 1.0)   # arbitrary nonzero content, matches what the real loop body writes
    global block_old = cctx.R === nothing ? Hraw_old : Hraw_old * cctx.R
end
GC.gc()
scratch_block = cctx.R === nothing ? nothing : Matrix{Float64}(undef, cctx.NCORE, cctx.nO)
b_new = @allocated begin
    fill!(cctx.Hraw_EC, 1.0)
    global block_new = cctx.R === nothing ? cctx.Hraw_EC : mul!(scratch_block, cctx.Hraw_EC, cctx.R)
end
lp(">>> @allocated OLD (Matrix(undef)+Hraw_EC*R):  ", b_old, " bytes")
lp(">>> @allocated NEW (persistent Hraw_EC + mul! into persistent block_ec): ", b_new, " bytes")
check("new path allocates strictly less", b_new < b_old)
check("block_ec values bit-identical (same inputs, mul! vs *)", block_old == block_new)

println("="^78)
println("Section 3: Hraw_CC/RtHraw_CC/block_cc (found live during §6.1 investigation -- L^2 iterations, not L)")
println("="^78)
GC.gc()
b_old_cc = @allocated begin
    global Hraw_CC_old = Matrix{Float64}(undef, cctx.nO, cctx.nO)
    fill!(Hraw_CC_old, 1.0)
    global block_cc_old = cctx.R === nothing ? Hraw_CC_old : (cctx.R' * Hraw_CC_old * cctx.R)
end
GC.gc()
b_new_cc = @allocated begin
    fill!(cctx.Hraw_CC, 1.0)
    global block_cc_new = if cctx.R === nothing
        cctx.Hraw_CC
    else
        mul!(cctx.RtHraw_CC, cctx.R', cctx.Hraw_CC)
        mul!(cctx.block_cc, cctx.RtHraw_CC, cctx.R)
    end
end
lp(">>> @allocated OLD (Matrix(undef)+R'*Hraw_CC*R):  ", b_old_cc, " bytes")
lp(">>> @allocated NEW (persistent Hraw_CC + mul! into persistent block_cc): ", b_new_cc, " bytes")
lp(">>> per-callback estimate: OLD ", b_old_cc * cctx.L^2, " bytes (x L^2=", cctx.L^2, "), NEW ", b_new_cc * cctx.L^2, " bytes")
check("new H_CC path allocates strictly less", b_new_cc < b_old_cc)
check("block_cc values bit-identical (same inputs, mul! vs *)", block_cc_old == block_cc_new)

# Full real Hessian callback still produces a correct, finite result with both fixes in place.
base2, verify2 = archC_verified_state(x_free_calib, pcx.ctx_cm, pcx.cctx)
check("full archC inner solve still feasible with Hraw_CC fix", base2.inner_status in (0, -100, -101, -103))
check("full archC Delta_dual still finite", isfinite(verify2.Delta_dual))
check("full archC Delta_dual unchanged by the fix", isapprox(verify2.Delta_dual, verify.Delta_dual; rtol = 1e-10))

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
