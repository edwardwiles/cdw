# 2026-07-25 continuation, task §3: D=20 restricted full-Hessian correctness gates.
#
# D=20 real data, D_dest=19, :exclude_row, W=80,000, seed=20260719 -- the FULL production scale,
# not the D=4 gates from the prior session. Compares dense-reference vs shared exact winner-pair
# H_EE for flexible CM (L=50), CM+mean/ZC (K_mean=1,K_pair in {0,1}), and origin-ZC
# (K_mean=1,K_pair in {0,1}), at TWO points: P0 (genuine calibration) and P1 (a small zfree
# perturbation off calibration, independently checked feasible under BOTH backends before use --
# same discipline as diag/compressed-hessian-operator-audit-2026-07-25's own D=20 benchmark
# script, since no pre-existing checkpointed "near delta=1, hard" incumbent exists in this
# worktree for these specific restriction configurations).
#
# Validates the COMPLETE assembled family Hessian and the COMPLETE inner solve at each point --
# not merely an extracted H_EE block (task's own explicit instruction).
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl",
          "core_exact_hessian.jl",
          "three_way_derivatives.jl", "lfix_incremental.jl", "composite_gradient.jl", "composite_gradient_fast.jl",
          "cm_lookup_kernels.jl", "lfix_cm_aware.jl", "cm_hessian_architectures.jl", "cm_hessian_threaded.jl",
          "cm_production_bundle.jl", "cm_outer_driver.jl",
          "cm_meanzc_moments.jl", "cm_meanzc_production.jl",
          "cm_originzc_target_layout.jl", "cm_originzc_moments.jl", "cm_originzc_production.jl"]
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
winner_hash(cf::CompressedFactual) = hash(cf.winner)
winner_hash(cf) = "NOT_A_COMPRESSEDFACTUAL($(repr(cf)))"

function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

W = 80_000
lp("Building D=20 real context: :exclude_row, W=$W, seed=20260719 ..."); flush(stdout)
ctx0 = d20_real_setup_design(W = W, δ = 1.0, find_smallest = true, draw_seed = 20260719, destination_sample = :exclude_row)
pe0 = build_pivot_elimination(ctx0)
x_free_calib = ctx0.θ0_up[ctx0.free_idx]

"Build a zfree-perturbed x_free off calibration; returns the perturbed x_free (theta_econ full reconstruction happens at the call site via CS.reconstruct_full)."
function perturbed_x_free(scale::Float64, seed::Int)
    Aod_block_raw = ctx0.θ0_up[ctx0.Aod_offset+1 : ctx0.Aod_offset + ctx0.D*ctx0.D_dest]
    zfree_calib = pivot_reduce(reshape(log.(Aod_block_raw), ctx0.D, ctx0.D_dest), pe0)
    Random.seed!(seed)
    zfree = zfree_calib .+ scale .* randn(length(zfree_calib))
    logA = pivot_expand(zfree, pe0)
    Aod_lvl = exp.(logA)
    A_full = reshape(Aod_lvl, ctx0.D, ctx0.D_dest)
    x_free = copy(ctx0.θ0_up[ctx0.free_idx])
    for o in 1:ctx0.D, s in 1:ctx0.D_dest
        fp = ctx0.Aod_free_pos[o, s]
        fp > 0 && (x_free[fp] = A_full[o, s])
    end
    return x_free
end
const x_free_p1 = perturbed_x_free(0.05, 20260719)

"""
    compare_family(label, base_fn, verified_fn, args...; toggle_backend!, get_backend_ctx, hessian_getter, n_getter)

Generic dense-vs-winner-pair comparison at a given point for one family. `toggle_backend!(sym)`
sets the family's own `core_hessian_backend` field to `sym`; `hessian_getter(base, ctx_cm)`
returns the packed Hessian at the converged point (re-running the SAME Hessian callback the
family's own production wiring uses, at the SAME converged dual point, for BOTH arms).
"""
function compare_family(label::AbstractString, point_label::AbstractString,
        run_fn::Function, hessian_fn::Function, n::Int, cf_getter::Function;
        set_backend!::Function)
    lp("  >>> $label/$point_label: starting dense-reference arm")
    reset_core_hessian_counters!()
    set_backend!(:dense_reference)
    base_d, verify_d = run_fn()
    counters_d = deepcopy(CORE_HESSIAN_COUNTERS[])
    lp("  >>> $label/$point_label: dense-reference arm done, nStatus=", base_d.inner_status)

    lp("  >>> $label/$point_label: starting winner-pair arm")
    reset_core_hessian_counters!()
    set_backend!(:exact_winner_pair_parallel)
    base_p, verify_p = run_fn()
    counters_p = deepcopy(CORE_HESSIAN_COUNTERS[])
    lp("  >>> $label/$point_label: winner-pair arm done, nStatus=", base_p.inner_status)

    check("$label/$point_label: both backends feasible", base_d.inner_status in FEASIBLE_CODES && base_p.inner_status in FEASIBLE_CODES)
    check("$label/$point_label: dual solution agrees (zeta)", isapprox(base_d.ζstar, base_p.ζstar; rtol = 1e-7))
    check("$label/$point_label: dual solution agrees (lambda)", isapprox(base_d.λstar, base_p.λstar; rtol = 1e-6, atol = 1e-9))
    check("$label/$point_label: Delta_dual agrees", isapprox(verify_d.Delta_dual, verify_p.Delta_dual; rtol = 1e-7))
    check("$label/$point_label: max_abs_moment_kkt_resid agrees", isapprox(verify_d.max_abs_moment_kkt_resid, verify_p.max_abs_moment_kkt_resid; atol = 1e-7))

    # Bug fixed 2026-07-25 continuation: `set_backend!` is GLOBAL mutable state on the family's
    # own cctx/octx, last set to :exact_winner_pair_parallel above -- without explicitly resetting
    # it back to :dense_reference before computing Hd, `hessian_fn(base_d)` would ALSO compute via
    # the winner-pair backend (both Hd and Hp identical by construction, not by genuine agreement,
    # silently making the "full assembled Hessian agrees" check vacuous). Caught by cross-checking
    # against `counters_d`/`counters_p` (see the two checks above) rather than trusting a
    # suspiciously-perfect max|Δ|=0.0 at face value.
    set_backend!(:dense_reference)
    Hd = hessian_fn(base_d)
    set_backend!(:exact_winner_pair_parallel)
    Hp = hessian_fn(base_p)
    diff = abs.(Hd .- Hp)
    maxabs = maximum(diff)
    idx = argmax(diff)
    relbase = max(1.0, maximum(abs.(Hd)))
    maxrel = maxabs / relbase
    check("$label/$point_label: full assembled Hessian agrees (max|Δ|<1e-6)", maxabs < 1e-6)
    check("$label/$point_label: dense arm used dense/inline backend only (0 winner-pair calls)", counters_d.winner_pair_hessian_calls == 0)
    check("$label/$point_label: winner-pair arm had ZERO unexplained dense fallback", counters_p.dense_core_fallback_calls == 0)

    cf_d = cf_getter()
    @printf("  %s/%s: max|ΔH|=%.3e (rel %.3e) at (%d,%d); Delta_dual d=%.10f p=%.10f; KKT d=%.3e p=%.3e; n=%d\n",
        label, point_label, maxabs, maxrel, idx[1], idx[2], verify_d.Delta_dual, verify_p.Delta_dual,
        verify_d.max_abs_moment_kkt_resid, verify_p.max_abs_moment_kkt_resid, n)
    @printf("  %s/%s: winner_hash=%s  dense-arm counters: wp=%d dense_fb=%d rebuilds=%d | winner-pair-arm counters: wp=%d(serial=%d,par=%d) dense_fb=%d rebuilds=%d\n",
        label, point_label, string(winner_hash(cf_d)),
        counters_d.winner_pair_hessian_calls, counters_d.dense_core_fallback_calls, counters_d.compressed_core_rebuilds,
        counters_p.winner_pair_hessian_calls, counters_p.winner_pair_serial_calls, counters_p.winner_pair_parallel_calls,
        counters_p.dense_core_fallback_calls, counters_p.compressed_core_rebuilds)
    return (base_d = base_d, verify_d = verify_d, base_p = base_p, verify_p = verify_p, maxabs = maxabs, maxrel = maxrel)
end

println("="^100)
println("FLEXIBLE CM, L=50")
println("="^100)
pcx = build_cm_production_context(ctx0, CS; L = 50, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = true)
n_cm = pcx.cctx.NCORE + pcx.cctx.ncm
for (xf, plabel) in [(x_free_calib, "P0_calib"), (x_free_p1, "P1_near_delta1")]
    compare_family("flexibleCM_L50", plabel,
        () -> archC_verified_state(xf, pcx.ctx_cm, pcx.cctx),
        (base) -> begin
            x = vcat(base.ζstar, base.λstar)
            _archC_prep_for_hessian!(pcx.ctx_cm.obj, x)
            h = Vector{Float64}(undef, n_cm*(n_cm+1)÷2)
            hessian_cm_structured_v2!(h, pcx.ctx_cm.obj, pcx.cctx; threaded_bins = true, tls = pcx.cctx.tls, use_syrk = true)
            unpack_packed(h, n_cm)
        end, n_cm, () -> pcx.cctx.core_cf_ref[];
        set_backend! = (b -> pcx.cctx.core_hessian_backend = b))
end

println("="^100)
println("CM+mean/ZC")
println("="^100)
for (K_mean, K_pair, flabel) in [(1, 0, "cmMeanZC_K1_mean_only"), (1, 1, "cmMeanZC_K1_mean_zc")]
    local pcx_mz = build_cm_meanzc_production_context(ctx0, CS; L = 50, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal)
    local nu0 = [Float64(factorial(k)) for k in 1:K_mean]
    local n_mz = pcx_mz.cctx.NCORE + pcx_mz.cctx.ncm
    for (xf, plabel) in [(x_free_calib, "P0_calib"), (x_free_p1, "P1_near_delta1")]
        compare_family(flabel, plabel,
            () -> archC_meanzc_verified_state(xf, nu0, pcx_mz.ctx_cm, pcx_mz.cctx),
            (base) -> begin
                x = vcat(base.ζstar, base.λstar)
                _archC_prep_for_hessian!(pcx_mz.ctx_cm.obj, x)
                h = Vector{Float64}(undef, n_mz*(n_mz+1)÷2)
                hessian_cm_structured!(h, pcx_mz.ctx_cm.obj, pcx_mz.cctx)
                unpack_packed(h, n_mz)
            end, n_mz, () -> pcx_mz.cctx.core_cf_ref[];
            set_backend! = (b -> pcx_mz.cctx.core_hessian_backend = b))
    end
end

println("="^100)
println("ORIGIN-ZC")
println("="^100)
for (K_mean, K_pair, flabel) in [(1, 0, "originZC_K1_mean_only"), (1, 1, "originZC_K1_mean_zc")]
    local layout = OriginByPowerLayout(ctx0.D, K_mean, K_pair)
    local pcx_oz = build_originzc_production_context(ctx0, CS, layout)
    local nu0 = vcat([fill(Float64(factorial(k)), ctx0.D) for k in 1:K_mean]...)
    local n_oz = pcx_oz.ctx_cm.obj.outer_constr_index
    for (xf, plabel) in [(x_free_calib, "P0_calib"), (x_free_p1, "P1_near_delta1")]
        compare_family(flabel, plabel,
            () -> archOZ_verified_state(xf, nu0, pcx_oz.ctx_cm),
            (base) -> begin
                x = vcat(base.ζstar, base.λstar)
                _archC_prep_for_hessian!(pcx_oz.ctx_cm.obj, x)
                h = Vector{Float64}(undef, n_oz*(n_oz+1)÷2)
                cb = archA_partitioned_hess_cb_builder(pcx_oz.octx)
                cb(nothing, nothing, (x = x,), (hess = h,), pcx_oz.ctx_cm.obj)
                unpack_packed(h, n_oz)
            end, n_oz, () -> pcx_oz.octx.core_cf_ref[];
            set_backend! = (b -> pcx_oz.octx.core_hessian_backend = b))
    end
end

println("="^100)
if isempty(FAILURES)
    println("ALL D=20 RESTRICTED FULL-HESSIAN GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
