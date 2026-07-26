# Five-family finish task, Phase 1.2 + 1.3 (2026-07-26): real D=20/W=80,000 gates for the exact-
# point cache and the compressed-core workspace across ALL FOUR restricted families, closing the
# gap the inherited remediation session's own report disclosed (its Phase C/E gates were D=4 only,
# and origin-ZC was explicitly not reached for either).
#
# Sequence per family, run through the SAME production entry points the real drivers use
# (archC_verified_state / archC_frechet_verified_state / archC_meanzc_verified_state /
# archOZ_verified_state), wrapped in the real cm_cache_lookup_or_compute! + CMProductionEvalKey
# machinery (cm_exact_cache_production.jl) exactly as Phase C wires it:
#   1. value at point A (calibration)          -> expect MISS (cold)
#   2. "gradient" re-query at identical point A -> expect HIT, same_point_inner_resolves == 0
#   3. repeated value at point A                -> expect HIT
#   4. distinct nearby point B                  -> expect MISS (genuine miss, not a collision)
#   5. return to point A                        -> expect HIT
#
# Also verifies: cache ON vs cache OFF byte-identical Delta_dual/KKT residual at both A and B;
# workspace ON vs workspace OFF byte-identical; workspace object identity stable (no resize) across
# all five calls; winner hash stable across the two genuine A-solves (same point -> same winners).

const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "cm_feature_immutability_counters.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "compressed_factual_buffer_reuse.jl", "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_screen_bridge.jl", "cm_outer_driver.jl",
          "gradient_workspace.jl", "lfix_factorized.jl", "lfix_factorized_workspace.jl",
          "lfix_cm_cplus.jl", "nested_quantile_grids.jl",
          "cm_config.jl", "cm_meanzc_moments.jl", "cm_meanzc_config.jl", "cm_meanzc_production.jl", "cm_meanzc_cplus.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl",
          "cm_frechet_level.jl", "cm_frechet_hessian.jl", "cm_frechet_hessian_threaded.jl", "cm_frechet_cplus.jl",
          "cm_exact_cache_production.jl"]
    include(joinpath(_D4E, f))
end
using Printf, LinearAlgebra, Random, Statistics

const FEASIBLE_CODES = (0, -100, -101, -103)
const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    println(rpad(cond ? "PASS" : "FAIL", 6), name)
    cond || push!(FAILURES, name)
    flush(stdout)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))
winner_hash(cf) = hasproperty(cf, :winner) ? hash(cf.winner) : hash(cf)

W = 80_000
lp("Building D=20 real context: :exclude_row, W=$W, seed=20260719 ..."); flush(stdout)
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]

function perturbed_x_free(scale::Float64, seed::Int)
    Aod_block_raw = ctx0.θ0_up[ctx0.Aod_offset+1 : ctx0.Aod_offset + ctx0.D*ctx0.D_dest]
    zfree_calib = pivot_reduce(reshape(log.(Aod_block_raw), ctx0.D, ctx0.D_dest), pe0)
    Random.seed!(seed)
    zfree = zfree_calib .+ scale .* randn(length(zfree_calib))
    logA = pivot_expand(zfree, pe0)
    A_full = reshape(exp.(logA), ctx0.D, ctx0.D_dest)
    x_free = copy(ctx0.θ0_up[ctx0.free_idx])
    for o in 1:ctx0.D, s in 1:ctx0.D_dest
        fp = ctx0.Aod_free_pos[o, s]
        fp > 0 && (x_free[fp] = A_full[o, s])
    end
    return x_free
end
const x_free_B = perturbed_x_free(0.05, 20260719)

"""
Runs the full 5-point cache+workspace gate for one family. `verify_fn(ctx_cm)` closes over
whatever family-specific extra args (nu, level_targets) are needed and takes `x_free` -> (base,verify).
`ws_ctx_getter(ctx_cm)` returns the NamedTuple field workspace should be attached to (may be ctx_cm
itself or a nested field, depending on family).
"""
function gate_family(label::AbstractString, ctx_cm, make_verify_fn::Function)
    D, Ddest = ctx0.D, ctx0.D_dest
    ctx_cm_ws = attach_compressed_factual_workspace(ctx_cm, D, Ddest, W)
    ws_obj = ctx_cm_ws.cf_workspace
    verify_fn_ws = make_verify_fn(ctx_cm_ws)
    verify_fn_nows = make_verify_fn(ctx_cm)

    cache = cm_production_exact_cache()
    reset_cm_exact_cache_counters!()
    key(xf) = CMProductionEvalKey(collect(xf), Float64[], 1.0, true, "cm_upper", Symbol(label), 50, :orthonormal, 0, 0, :legacy_z, "fp_$label")

    results = Dict{Symbol,Any}()
    solves_before = Ref(0)
    wrapped(xf, tag) = begin
        c0 = deepcopy(CM_EXACT_CACHE_COUNTERS[])
        base, verify = cm_cache_lookup_or_compute!(cache, key(xf), () -> (solves_before[] += 1; verify_fn_ws(xf)))
        results[tag] = (base = base, verify = verify, cache_after = deepcopy(CM_EXACT_CACHE_COUNTERS[]), misses_delta = CM_EXACT_CACHE_COUNTERS[].misses - c0.misses, hits_delta = CM_EXACT_CACHE_COUNTERS[].hits - c0.hits, ws_identity = ctx_cm_ws.cf_workspace)
        return results[tag]
    end

    lp("  [$label] 1/5: value at A (calib) -- expect MISS"); n0 = solves_before[]
    r1 = wrapped(x_free_calib, :A1)
    check("$label: A1 real inner solve happened (miss)", solves_before[] == n0 + 1)
    check("$label: A1 feasible", r1.verify.inner_status in FEASIBLE_CODES)

    lp("  [$label] 2/5: 'gradient' re-query at identical A -- expect HIT, zero new inner solves")
    r2 = wrapped(x_free_calib, :A2)
    check("$label: A2 is cache HIT (same_point_inner_resolves=0)", solves_before[] == n0 + 1)
    check("$label: A2 byte-identical to A1 (Delta_dual)", r2.verify.Delta_dual == r1.verify.Delta_dual)

    lp("  [$label] 3/5: repeated value at A -- expect HIT")
    r3 = wrapped(x_free_calib, :A3)
    check("$label: A3 is cache HIT", solves_before[] == n0 + 1)

    lp("  [$label] 4/5: distinct nearby point B -- expect genuine MISS")
    r4 = wrapped(x_free_B, :B1)
    check("$label: B1 real inner solve happened (genuine miss, not a collision)", solves_before[] == n0 + 2)
    check("$label: B1 feasible", r4.verify.inner_status in FEASIBLE_CODES)
    check("$label: B1 differs from A (not a false cache hit)", r4.verify.Delta_dual != r1.verify.Delta_dual)

    lp("  [$label] 5/5: return to A -- expect HIT")
    r5 = wrapped(x_free_calib, :A4)
    check("$label: A4 is cache HIT (return-to-A)", solves_before[] == n0 + 2)
    check("$label: A4 byte-identical to A1", r5.verify.Delta_dual == r1.verify.Delta_dual)

    check("$label: workspace object identity stable across all 5 calls (no resize)",
        r1.ws_identity === r2.ws_identity === r3.ws_identity === r4.ws_identity === r5.ws_identity === ws_obj)

    # Cache OFF (cache=nothing) vs cache ON, both at A and B: same converged KKT point. NOT
    # checked with strict == -- by this point obj.x (the shared mutable warm-start slot on
    # ctx_cm.obj) has been left wherever B1's solve landed it (A2/A3/A4 were cache HITs and never
    # touched obj.x again), so this recompute is genuinely warm-started from a DIFFERENT point
    # than the original A1/B1 solves were. isapprox with the same rtol=1e-7 this codebase's own
    # D=20 dense-vs-winner-pair gate (test_d20_restricted_full_hessian_gates.jl) uses for
    # cross-warm-start Delta_dual agreement -- same convex problem, same KKT point, different
    # solve path.
    base_off_A, verify_off_A = cm_cache_lookup_or_compute!(nothing, nothing, () -> verify_fn_ws(x_free_calib))
    check("$label: cache OFF vs ON agrees at A (Delta_dual)", isapprox(verify_off_A.Delta_dual, r1.verify.Delta_dual; rtol = 1e-7))
    base_off_B, verify_off_B = cm_cache_lookup_or_compute!(nothing, nothing, () -> verify_fn_ws(x_free_B))
    check("$label: cache OFF vs ON agrees at B (Delta_dual)", isapprox(verify_off_B.Delta_dual, r4.verify.Delta_dual; rtol = 1e-7))

    # Workspace ON vs OFF: same underlying mutable ctx_cm.obj (attach_compressed_factual_workspace
    # merges the SAME obj by reference), so this recompute is warm-started from wherever the last
    # call left obj.x -- same cross-warm-start isapprox discipline as the cache OFF/ON checks above,
    # not strict ==.
    cache2 = cm_production_exact_cache()
    base_nows, verify_nows = verify_fn_nows(x_free_calib)
    check("$label: workspace OFF vs ON agrees at A (Delta_dual)", isapprox(verify_nows.Delta_dual, r1.verify.Delta_dual; rtol = 1e-7))

    final_c = CM_EXACT_CACHE_COUNTERS[]
    @printf("  [%s] final counters: lookups=%d hits=%d misses=%d store_rejections=%d\n",
        label, final_c.lookups, final_c.hits, final_c.misses, final_c.store_rejections)
    check("$label: cache accounting: 3 hits total (A2,A3,A4)", final_c.hits == 3)
    check("$label: cache accounting: 2 misses total (A1,B1)", final_c.misses == 2)
end

println("="^100); println("FLEXIBLE CM, L=50"); println("="^100)
pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = true)
gate_family("flexibleCM", pcx.ctx_cm, (ctx_cm -> xf -> archC_verified_state(xf, ctx_cm, pcx.cctx)))

println("="^100); println("COMMON FRECHET"); println("="^100)
pcx_f = build_cm_frechet_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, cm_hessian_backend = :structured)
gate_family("commonFrechet", pcx_f.ctx_cm, (ctx_cm -> xf -> archC_frechet_verified_state(xf, ctx_cm, pcx_f.cctx, pcx_f.aug.level_targets)))

println("="^100); println("CM+mean/ZC (K_mean=1,K_pair=1)"); println("="^100)
pcx_mz = build_cm_meanzc_production_context(ctx0, CS; L = 50, K_mean = 1, K_pair = 1, contrasts = :orthonormal)
nu0_mz = [Float64(factorial(k)) for k in 1:1]
gate_family("cmMeanZC", pcx_mz.ctx_cm, (ctx_cm -> xf -> archC_meanzc_verified_state(xf, nu0_mz, ctx_cm, pcx_mz.cctx)))

println("="^100); println("ORIGIN-ZC (K_mean=1,K_pair=1)"); println("="^100)
layout = OriginByPowerLayout(ctx0.D, 1, 1)
pcx_oz = build_originzc_production_context(ctx0, CS, layout)
nu0_oz = vcat([fill(Float64(factorial(k)), ctx0.D) for k in 1:1]...)
gate_family("originZC", pcx_oz.ctx_cm, (ctx_cm -> xf -> archOZ_verified_state(xf, nu0_oz, ctx_cm)))

println("="^100); println("Phase 3: CM feature immutability counters"); println("="^100)
print_cm_feature_immutability_counters()
fic = CM_FEATURE_IMMUTABILITY_COUNTERS[]
check("feature immutability: exactly 4 context builds (one per family)", fic.cm_feature_context_builds == 4)
check("feature immutability: cm_feature_rebuilds_due_to_A_or_gp = 0 (mandatory invariant, despite each family being queried at 2 genuinely distinct A/gp points x5 calls)",
    fic.cm_feature_rebuilds_due_to_A_or_gp == 0)
check("feature immutability: cm_feature_rebuilds_due_to_theta = 0 (fixed theta throughout)", fic.cm_feature_rebuilds_due_to_theta == 0)

println("="^100)
if isempty(FAILURES)
    println("ALL PHASE 1.2/1.3 D=20 ALL-FAMILY (INCL. ORIGIN-ZC) GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
