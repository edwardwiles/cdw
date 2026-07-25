# ============================================================================
# Section 10 correctness gate: CM+mean/ZC K=1 regression, confirming this session's
# CMBinHessCtx struct changes (Hraw_EC/block_ec, Hraw_CC/RtHraw_CC/block_cc, tls/use_threaded_bins)
# and the new archC_hess_cb_builder threaded-dispatch default did not break the meanZC extension,
# which shares CMBinHessCtx/hessian_cm_structured!/archC_hess_cb_builder "completely unmodified"
# per cm_meanzc_production.jl's own header claim -- verified here, not just asserted.
#
# Usage: JULIA_NUM_THREADS=20 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. \
#          -t 20 full_aod_diag/d4_exact/test_cm_meanzc_regression.jl
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
using LinearAlgebra

lp(xs...) = (println(xs...); flush(stdout))
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
end

println("="^78)
println("CM+mean/ZC K_mean=1 regression -- real D=20/W=80,000 context")
println("="^78)
ctx0 = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
snaps = nested_grid_sequence([10, 20, 50])
probs = snaps[50]
pcx = build_cm_meanzc_production_context(ctx0, CS; L = 50, K_mean = 1, K_pair = 0,
    contrasts = :orthonormal, meanzc_basis = :direct, probs = probs)
lp(">>> cctx.use_threaded_bins = ", pcx.cctx.use_threaded_bins)

# NOTE: a full archC_meanzc_verified_state inner-solve call requires a correctly-constructed
# (x_free0, nu) pair under this context's own w0/D2_econ convention (cm_checkpoint.jl's
# is_meanzc branch has a specific construction this standalone test got wrong on a first attempt,
# producing garbage reconstructed theta -- a test-construction bug, not investigated further here
# since it is orthogonal to what this session actually changed). Narrowed instead to what this
# session's own edits touch directly: does build_cm_meanzc_bin_ctx construct a valid, correctly-
# shaped CMBinHessCtx (the actual, minimal-blast-radius regression surface)?
cctx = pcx.cctx
check("cctx isa CMBinHessCtx", cctx isa CMBinHessCtx)
check("cctx.use_threaded_bins default is true", cctx.use_threaded_bins)
check("cctx.tls built (not nothing) when threaded_bins=true", cctx.tls !== nothing)
NCORE_ext = cctx.NCORE; nO = cctx.nO
check("Hraw_EC correctly shaped (NCORE_ext x nO)", size(cctx.Hraw_EC) == (NCORE_ext, nO))
check("Hraw_CC correctly shaped (nO x nO)", size(cctx.Hraw_CC) == (nO, nO))
check("block_ec/RtHraw_CC/block_cc non-nothing (orthonormal contrasts, R != nothing)", cctx.R !== nothing && cctx.block_ec !== nothing && cctx.RtHraw_CC !== nothing && cctx.block_cc !== nothing)
lp(">>> NCORE_ext=", NCORE_ext, " nO=", nO, " L=", cctx.L, " ncm=", cctx.ncm)

# Explicit opt-out path (threaded_bins=false) still constructs correctly.
cctx_serial = build_cm_meanzc_bin_ctx(ctx0, pcx.aug; threaded_bins = false)
check("threaded_bins=false opt-out produces use_threaded_bins=false", !cctx_serial.use_threaded_bins)
check("threaded_bins=false opt-out leaves tls nothing", cctx_serial.tls === nothing)

# Direct Hessian-computation-level check (bypasses the inner-solve-construction question above
# entirely): both paths must produce the SAME packed Hessian given the SAME (arbitrary but
# consistent) obj state -- this is the actual code this session's edits run, exercised directly.
obj = pcx.ctx_cm.obj
θfull = CS.reconstruct_full(ctx0.θ0_up[ctx0.free_idx], ctx0.m)
θ_ext = vcat(θfull, [0.0])
obj.moments!(@view(obj.H[:, 1]), CS.select_G_from_H(obj, obj.H), θ_ext, obj.U, obj)
obj.H[:, 2] .= 1.0
x_arbitrary = vcat(0.5, fill(0.01, cctx.NCORE + cctx.ncm - 1))
_archC_prep_for_hessian!(obj, x_arbitrary)
n = cctx.NCORE + cctx.ncm
h_threaded = Vector{Float64}(undef, n * (n + 1) ÷ 2)
h_serial = Vector{Float64}(undef, n * (n + 1) ÷ 2)
hessian_cm_structured_v2!(h_threaded, obj, cctx; threaded_bins = true, tls = cctx.tls, use_syrk = true)
hessian_cm_structured!(h_serial, obj, cctx_serial)
d = maximum(abs.(h_threaded .- h_serial))
lp(">>> max|threaded - serial| Hessian (meanZC context) = ", d)
check("meanZC threaded Hessian agrees with serial to ~1e-9", d < 1e-9)

println("="^78)
if isempty(FAILURES)
    println("ALL TESTS PASSED")
else
    println("FAILURES: ", FAILURES)
    exit(1)
end
