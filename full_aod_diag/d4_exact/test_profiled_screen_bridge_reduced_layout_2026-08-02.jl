# Phase 12 (integration/phase12-13-runner-checkpoints-2026-08-02), item 1 gate:
# proves `profiled_cm_screen_precheck!` (profiled_screen_bridge_2026-08-02.jl) gives IDENTICAL
# screen-certificate outcomes to the existing, unchanged `cm_screen_precheck!` fed directly with
# the equivalent full-space point -- i.e. decoding a reduced/profiled `w_profiled` to `xf` via
# `decode_outer_profiled` does NOT silently misindex anything relative to the dense/full path's
# own (o,d) certificate logic. Real D20 context (build_screen=true default), small W for speed.
const D4X = @__DIR__
for f in ["context.jl", "context_real_d20.jl", "draw_design.jl", "winners.jl", "oracle.jl",
          "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl", "winner_pair_cross_hessian.jl", "threaded_cross_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_production_bundle.jl",
          "cm_outer_driver.jl", "cm_screen_bridge.jl", "nested_quantile_grids.jl", "cm_checkpoint.jl",
          "relative_a_coordinate_2026-07-31.jl", "profiled_economic_moment_layout_2026-08-01.jl",
          "gravity_pivot_on_retained_2026-07-31.jl", "outer_coordinate_layout_profiled_2026-07-31.jl",
          "profiled_screen_bridge_2026-08-02.jl", "infeasibility_screen.jl"]
    include(joinpath(D4X, f))
end
using Test, Printf, Random
flush(stdout)

ALL_PASS = Ref(true)
function check(name::AbstractString, cond::Bool)
    global ALL_PASS[] &= cond
    println(cond ? "PASS  " : "FAIL  ", name)
    flush(stdout)
end

const W_VAL = parse(Int, get(ENV, "SCREEN_BRIDGE_D20_W", "2000"))
println("Building real D=20 :exclude_row context at W=$W_VAL (build_screen=true default) ..."); flush(stdout)
ctx = d20_real_setup(W = W_VAL, δ = 1.0, find_smallest = true, destination_sample = :exclude_row)
check("ctx.pairwise precomputed by default (build_screen=true default)", ctx.pairwise !== nothing)
check("ctx.witness precomputed by default", ctx.witness !== nothing)

korea_idx = 14; brazil_idx = 3
spec = build_anchor_spec_from_ctx(ctx; global_overrides = Dict(korea_idx => brazil_idx))
θ0 = ctx.θ0_up
D = ctx.D; Ddest = ctx.D_dest
z_calib = log.(reshape(θ0[ctx.Aod_offset+1:ctx.Aod_offset+D*Ddest], D, Ddest))
gauge = build_anchor_gauge(z_calib, spec)
pe = build_pivot_elimination_on_retained(ctx, spec, gauge)
gp_calib = θ0[3+D]
x_free_calib = θ0[ctx.free_idx]

println("\n" * "="^78); println("GROUP 1: calibration point -- profiled-decode screen vs direct full-space screen"); println("="^78)
# NOTE: at this small gate W (draw-dependent screens), the calibration point itself may or may not
# be pairwise-certified feasible -- that is a property of the draw sample, not of this wrapper. The
# thing THIS gate proves is that decoding through the reduced/profiled path and feeding the full
# space directly give the IDENTICAL certificate outcome (same throw-or-not, same (o,d), same
# per-stage counters) -- i.e. the reduced-coordinate decode is not silently misindexing anything
# relative to the dense/full path's own certificate logic.
w_profiled_calib = reduce_to_w_profiled(gp_calib, z_calib, pe)
decoded_calib = decode_outer_profiled(w_profiled_calib, ctx, pe)
round_trip_err_calib = maximum(abs.(decoded_calib.xf .- vcat(gp_calib, vec(exp.(z_calib)))))
@printf("calibration round-trip max|xf_decoded - xf_direct| = %.4g (0 expected: gravity-consistent calibration z survives pivot-eliminate/expand exactly)\n", round_trip_err_calib)
check("profiled encode/decode round-trips the GENUINE (gravity-consistent) calibration point exactly (< 1e-4 abs, floating-point log/exp round-trip noise)", round_trip_err_calib < 1e-4)

sc_direct = CMScreenCounters()
sc_profiled = CMScreenCounters()
raised_direct = nothing; raised_profiled = nothing
try
    cm_screen_precheck!(collect(x_free_calib), ctx; counters = sc_direct, use_witness = true)
catch e
    global raised_direct = (typeof(e), sprint(showerror, e))
end
try
    profiled_cm_screen_precheck!(w_profiled_calib, ctx, pe; counters = sc_profiled, use_witness = true)
catch e
    global raised_profiled = (typeof(e), sprint(showerror, e))
end
println("  direct raised:          ", raised_direct === nothing ? "(no throw)" : raised_direct[2])
println("  profiled-decode raised: ", raised_profiled === nothing ? "(no throw)" : raised_profiled[2])
check("direct and profiled-decode paths agree on throw-or-not at calibration", (raised_direct === nothing) == (raised_profiled === nothing))
raised_direct !== nothing && check("direct and profiled-decode paths raise the IDENTICAL certificate message at calibration (same (o,d), same stage)",
      raised_profiled !== nothing && raised_direct[2] == raised_profiled[2])
check("counters identical direct vs profiled-decode at calibration",
      sc_direct.pairwise == sc_profiled.pairwise && sc_direct.witness == sc_profiled.witness &&
      sc_direct.winner == sc_profiled.winner && sc_direct.passed == sc_profiled.passed)

println("\n" * "="^78); println("GROUP 2: large perturbation in REDUCED coordinates -- SAME rejection via both paths"); println("="^78)
# Perturb the REDUCED outer coordinate w_profiled itself (the genuine free variable this runner's
# KNITRO outer loop actually manipulates) by a large amount, decode ONCE via the existing
# decode_outer_profiled to get the equivalent full xf, then confirm: (a) feeding that xf directly
# into the unchanged cm_screen_precheck!, and (b) calling profiled_cm_screen_precheck! on the
# ORIGINAL w_profiled_pathological (which internally re-does the SAME decode), agree exactly --
# proving the wrapper is a correct, non-misindexing thin decode+call, not a re-derivation.
Random.seed!(20260802)
w_profiled_pathological = copy(w_profiled_calib)
w_profiled_pathological[2:end] .+= 6.0 .* randn(length(w_profiled_pathological) - 1)   # large free-A perturbation
xf_pathological = decode_outer_profiled(w_profiled_pathological, ctx, pe).xf

sc_direct2 = CMScreenCounters(); sc_profiled2 = CMScreenCounters()
try
    cm_screen_precheck!(xf_pathological, ctx; counters = sc_direct2, use_witness = true)
catch e
    global raised_direct = (typeof(e), sprint(showerror, e))
end
try
    profiled_cm_screen_precheck!(w_profiled_pathological, ctx, pe; counters = sc_profiled2, use_witness = true)
catch e
    global raised_profiled = (typeof(e), sprint(showerror, e))
end
println("  direct raised:          ", raised_direct === nothing ? "(no throw)" : raised_direct[2])
println("  profiled-decode raised: ", raised_profiled === nothing ? "(no throw)" : raised_profiled[2])
check("direct full-space path (fed the decoded xf) certifies the large perturbation infeasible", raised_direct !== nothing && raised_direct[1] === CMExpectedSolveFailure)
check("profiled-decode wrapper (fed the original w_profiled) certifies the SAME point infeasible, IDENTICAL message",
      raised_profiled !== nothing && raised_profiled[1] === CMExpectedSolveFailure && raised_direct[2] == raised_profiled[2])
check("BOTH paths reject via the SAME screen stage (pairwise/witness/winner counters identical)",
      sc_direct2.pairwise == sc_profiled2.pairwise && sc_direct2.witness == sc_profiled2.witness &&
      sc_direct2.winner == sc_profiled2.winner)

println()
println(ALL_PASS[] ? "ALL PASS" : "SOME FAILURES")
ALL_PASS[] || exit(1)
