# Debug (2026-07-29): reproduce the nStatus=-500 (KN_RC_CALLBACK_ERR) found when
# build_cm_frechet_production_context's :operator path is exercised as the FIRST-EVER solve in a
# fresh process (no prior :dense_reference build/solve in the same process). Bypass KNITRO
# entirely and call the underlying FG/Hessian callback pieces directly so the real Julia exception
# surfaces instead of being swallowed into a KNITRO -500 status.
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl", "context_real_d20.jl",
          "lfix_factorized_workspace.jl", "lfix_factorized.jl", "cm_screen_bridge.jl",
          "cm_frechet_lookup_kernels.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl",
          "cm_frechet_level.jl", "cm_frechet_lookup_production.jl", "cm_frechet_cplus.jl"]
    include(joinpath(_D4E, f))
end
using Printf

lp(xs...) = (println(xs...); flush(stdout))
lp("Building real D=20 context...")
ctx = d20_real_setup(W = 80_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
x_free_calib = ctx.θ0_up[ctx.free_idx]

lp("Building ONLY the :operator context (nothing dense built first in this process)...")
pcx_o = build_cm_frechet_production_context(ctx, CS; L = 50, contrasts = :anchored,
    cm_hessian_backend = :structured)   # relies on the new :operator default
obj = pcx_o.ctx_cm.obj
println("obj isa OperatorPsiBundle: ", obj isa OperatorPsiBundle)
level_targets = pcx_o.aug.level_targets

θ_full0 = CS.reconstruct_full(x_free_calib, pcx_o.ctx_cm.m)

lp("Priming the operator directly (prime_operator!)...")
try
    prime_operator!(obj, θ_full0, pcx_o.cctx.econ_ctx, pcx_o.cctx.core_cf_ref; restriction_state = pcx_o.cctx)
    lp("  prime_operator! OK")
catch e
    lp("  prime_operator! THREW:")
    showerror(stdout, e, catch_backtrace())
    println()
end

lp("Building CMFrechetLookupState and calling the FG functor directly (bypassing KNITRO)...")
bins_u = pcx_o.cctx.Bidx isa Matrix{UInt32} ? pcx_o.cctx.Bidx : Matrix{UInt32}(pcx_o.cctx.Bidx)
ncm_cm = pcx_o.cctx.ncm - pcx_o.cctx.L
st = CMFrechetLookupState(obj, pcx_o.cctx.NCORE, ncm_cm, pcx_o.cctx.L, pcx_o.cctx.L, pcx_o.cctx.D,
    pcx_o.cctx.origins, pcx_o.cctx.refIndex1, bins_u, pcx_o.cctx.R, level_targets;
    core_cf_ref = pcx_o.cctx.core_cf_ref)
x0 = CS.inner_loop_initial_values(obj)
g0 = zeros(length(x0))
try
    f0 = st(x0, g0)
    @printf "  FG functor OK: f=%.6f\n" f0
catch e
    lp("  FG functor THREW:")
    showerror(stdout, e, catch_backtrace())
    println()
end

lp("Calling the Hessian callback builder + evaluating it directly...")
try
    hess_cb = archC_frechet_hess_cb_builder(pcx_o.cctx, level_targets)
    lp("  archC_frechet_hess_cb_builder OK, built closure: ", typeof(hess_cb))
catch e
    lp("  archC_frechet_hess_cb_builder THREW:")
    showerror(stdout, e, catch_backtrace())
    println()
end

lp("Now attempting the REAL inner_loop_internal_cmfrechetlookup_production directly (not via archC_frechet_base_state)...")
try
    K, x, nStatus, n_fg, n_hess = inner_loop_internal_cmfrechetlookup_production(obj, θ_full0, pcx_o.cctx, level_targets;
        hess_cb_builder = _obj -> archC_frechet_hess_cb_builder(pcx_o.cctx, level_targets))
    @printf "  inner_loop_internal_cmfrechetlookup_production: nStatus=%d\n" nStatus
catch e
    lp("  inner_loop_internal_cmfrechetlookup_production THREW:")
    showerror(stdout, e, catch_backtrace())
    println()
end
