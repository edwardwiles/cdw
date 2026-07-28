# ============================================================================
# port/shared-winner-pair-core-hessian-production-2026-07-25, task §7: D=4
# exact correctness gates for the shared H_EE backend, across ALL FOUR
# production families (unrestricted, flexible CM, CM+mean/ZC, origin-ZC).
#
# For each family: compare the :dense_reference backend (the ORIGINAL dense
# BLAS gemm/syrk computation, byte-for-byte the pre-port code path) against
# :exact_winner_pair_serial and :exact_winner_pair_parallel (workers=2,4) at
# several random dual points plus a real solved (feasible) point, checking:
#   - the H_EE block in isolation
#   - the FULL assembled family Hessian (H_EE + cross/restriction blocks)
#   - (where cheap) the real inner-solve feasibility/Delta_dual agreement
#
# Usage: JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia \
#          --project=. full_aod_diag/d4_exact/test_shared_core_hessian_d4_gates.jl
# ============================================================================
const _D4E = @__DIR__
for f in ["context.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
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
using Printf, LinearAlgebra, Random

const FAILURES = String[]
function check(name::AbstractString, cond::Bool)
    status = cond ? "PASS" : "FAIL"
    println(rpad(status, 6), name)
    cond || push!(FAILURES, name)
    return cond
end
lp(xs...) = (println(xs...); flush(stdout))

function unpack_packed(h::AbstractVector, n::Int)
    Mx = Matrix{Float64}(undef, n, n)
    k = 1
    @inbounds for i in 1:n, j in i:n
        Mx[i, j] = h[k]; Mx[j, i] = h[k]
        k += 1
    end
    return Mx
end

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
θ_full_calib = CS.reconstruct_full(x_free_calib, ctx.m)
Random.seed!(20260725)

println("="^90)
println("SECTION A: unrestricted family -- shared H_EE backend vs :dense_reference")
println("="^90)
let
    obj = ctx.obj
    n = obj.outer_constr_index
    cf = build_compressed_factual(θ_full_calib, ctx; check_ties = true)
    lp("D=4 unrestricted: n=$n, W=$(size(obj.U,1)), cf.oci=$(cf.oci), has_cf=$(cf.cf_col>0)")
    ws = build_core_exact_hessian_workspace(cf; worker_counts = [1, 2, 4])

    # `_dense_reference_core_hessian!` reads `obj.H` directly (exactly like production's
    # `CS.hessian!` does) -- materialize it once here, mirroring
    # diag/compressed-hessian-operator-audit-2026-07-25's own validate_winner_pair_hessian_d4.jl
    # (grav_raw=0.0 is fine: that column is strictly outside the H[:,2:1+outer_constr_index]
    # range any of these Hessian backends ever read).
    ncolI0 = cf.oci - 1
    materialize_dense_factual_structured!(@view(obj.H[:, 3:2+ncolI0]), cf)
    fill_gravity_column!(obj, 0.0)

    st = CompressedCBState(obj, cf, 0.0, false)
    function compare_at(x::AbstractVector; label = "")
        ζ = x[1]; λ = @view x[2:end]
        f, g_ζ, g_λ, q, _ = compressed_cc_value_grad(ζ, λ, cf; Psi! = obj.Psi!, dPsi! = obj.dPsi!)
        obj.arg0 .= q
        h_dense = Vector{Float64}(undef, n*(n+1)÷2)
        _dense_reference_core_hessian!(h_dense, obj, n)
        h_serial = Vector{Float64}(undef, n*(n+1)÷2)
        winner_pair_hessian!(h_serial, obj, serial_ctx(ws))
        h_par2 = Vector{Float64}(undef, n*(n+1)÷2)
        hessian_core_winner_pair!(h_par2, obj.arg2, obj, ws.parallel_ws; workers = 2, storage = :full_stride)
        h_par4 = Vector{Float64}(undef, n*(n+1)÷2)
        hessian_core_winner_pair!(h_par4, obj.arg2, obj, ws.parallel_ws; workers = 4, storage = :direct_packed)

        e_serial = maximum(abs.(h_serial .- h_dense))
        e_par2 = maximum(abs.(h_par2 .- h_dense))
        e_par4 = maximum(abs.(h_par4 .- h_dense))
        rel = max(1.0, maximum(abs.(h_dense)))
        @printf("  %-24s serial=%.3e (rel %.3e)  par2=%.3e  par4(direct_packed)=%.3e\n", label, e_serial, e_serial/rel, e_par2, e_par4)
        check("$label: serial matches dense", e_serial/rel < 1e-9)
        check("$label: parallel(workers=2,full_stride) matches dense", e_par2/rel < 1e-9)
        check("$label: parallel(workers=4,direct_packed) matches dense", e_par4/rel < 1e-9)
    end
    compare_at(zeros(n); label = "x=0")
    for i in 1:4
        compare_at(0.1 .* randn(n); label = "random[$i]")
    end
    nStatus, objSol, xsol, lambda_, _, _ = inner_loop_KNITRO_compressed(obj, CompressedCBState(obj, cf, 0.0, false))
    lp("real solved point: nStatus=$nStatus")
    if nStatus in (0, -100, -101, -103)
        compare_at(collect(xsol); label = "real solved x*")
    else
        push!(FAILURES, "unrestricted: real inner solve infeasible, nStatus=$nStatus")
    end
end

println("="^90)
println("SECTION B: flexible CM -- H_EE (winner-pair) + H_EC/H_CC (bin-prefix, untouched) full block")
println("="^90)
let
    pcx_d = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = false)
    pcx_p = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = false)
    pcx_d.cctx.core_hessian_backend = :dense_reference
    pcx_p.cctx.core_hessian_backend = :exact_winner_pair_parallel
    pcx_p.cctx.core_hessian_workers = 2

    base_d = archC_base_state(x_free_calib, pcx_d.ctx_cm, pcx_d.cctx)
    base_p = archC_base_state(x_free_calib, pcx_p.ctx_cm, pcx_p.cctx)
    check("CM: both backends' inner solves feasible", base_d.inner_status in (0,-100,-101,-103) && base_p.inner_status in (0,-100,-101,-103))
    check("CM: dual solutions agree (backend must not affect the converged point)", isapprox(base_d.ζstar, base_p.ζstar; rtol=1e-8) && isapprox(base_d.λstar, base_p.λstar; rtol=1e-6, atol=1e-9))

    n = pcx_d.cctx.NCORE + pcx_d.cctx.ncm
    NCORE = pcx_d.cctx.NCORE
    function full_hessian(base, ctx_cm, cctx)
        obj = ctx_cm.obj
        x = vcat(base.ζstar, base.λstar)
        # TEMPORARY DIAG: the dispatcher (_prep_dual_index_for_archC!) keys off inner_fg_backend,
        # which defaults to :cm_lookup for BOTH pcx_d/pcx_p here -- explicitly route by
        # core_hessian_backend instead, matching what this test is actually trying to compare.
        if cctx.core_hessian_backend === :dense_reference
            _archC_prep_for_hessian!(obj, x)
        else
            _prep_dual_index_for_archC!(cctx, obj, x)
        end
        h = Vector{Float64}(undef, n*(n+1)÷2)
        hessian_cm_structured!(h, obj, cctx)
        return unpack_packed(h, n)
    end
    Hd = full_hessian(base_d, pcx_d.ctx_cm, pcx_d.cctx)
    Hp = full_hessian(base_p, pcx_p.ctx_cm, pcx_p.cctx)
    eEE = maximum(abs.(Hd[1:NCORE,1:NCORE] .- Hp[1:NCORE,1:NCORE]))
    eFull = maximum(abs.(Hd .- Hp))
    lp("  max|ΔH_EE|=$eEE   max|ΔH_full|=$eFull")
    check("CM: H_EE block agrees (winner-pair vs dense)", eEE < 1e-8)
    check("CM: full assembled Hessian agrees", eFull < 1e-8)

    # also exercise the THREADED Architecture-C path (production default) with the shared backend
    pcx_t = build_cm_production_context(ctx, CS; L = 10, contrasts = :orthonormal, use_compressed_core = true, threaded_bins = true)
    pcx_t.cctx.core_hessian_backend = :exact_winner_pair_parallel
    base_t = archC_base_state(x_free_calib, pcx_t.ctx_cm, pcx_t.cctx)
    check("CM: threaded-bins inner solve feasible with shared H_EE", base_t.inner_status in (0,-100,-101,-103))
    x_t = vcat(base_t.ζstar, base_t.λstar)
    _prep_dual_index_for_archC!(pcx_t.cctx, pcx_t.ctx_cm.obj, x_t)  # TEMPORARY DIAG: same fix as full_hessian above -- pcx_t is operator, not dense
    h_t = Vector{Float64}(undef, n*(n+1)÷2)
    hessian_cm_structured_v2!(h_t, pcx_t.ctx_cm.obj, pcx_t.cctx; threaded_bins = true, tls = pcx_t.cctx.tls, use_syrk = true)
    Ht = unpack_packed(h_t, n)
    eThreaded = maximum(abs.(Hd .- Ht))
    lp("  threaded-bins vs dense-reference: max|ΔH_full|=$eThreaded")
    check("CM: threaded-bins (production default) full Hessian agrees with dense reference", eThreaded < 1e-8)
end

println("="^90)
println("SECTION C: CM+mean/ZC -- shared H_EE reused verbatim (K_mean=1,K_pair=0 and K_mean=1,K_pair=1)")
println("="^90)
for (K_mean, K_pair, label) in [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc")]
    local pcx_d = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal)
    local pcx_p = build_cm_meanzc_production_context(ctx, CS; L = 10, K_mean = K_mean, K_pair = K_pair, contrasts = :orthonormal)
    pcx_d.cctx.core_hessian_backend = :dense_reference
    pcx_p.cctx.core_hessian_backend = :exact_winner_pair_parallel
    pcx_p.cctx.core_hessian_workers = 2
    nu0 = [Float64(factorial(k)) for k in 1:K_mean]

    base_d = archC_meanzc_base_state(x_free_calib, nu0, pcx_d.ctx_cm, pcx_d.cctx)
    base_p = archC_meanzc_base_state(x_free_calib, nu0, pcx_p.ctx_cm, pcx_p.cctx)
    check("$label: both backends feasible", base_d.inner_status in (0,-100,-101,-103) && base_p.inner_status in (0,-100,-101,-103))
    check("$label: dual solutions agree", isapprox(base_d.ζstar, base_p.ζstar; rtol=1e-8) && isapprox(base_d.λstar, base_p.λstar; rtol=1e-6, atol=1e-9))

    local n = pcx_d.cctx.NCORE + pcx_d.cctx.ncm
    local ncore_core = pcx_d.cctx.ncore_core
    function full_hessian_mz(base, ctx_cm, cctx)
        obj = ctx_cm.obj
        x = vcat(base.ζstar, base.λstar)
        _archC_prep_for_hessian!(obj, x)
        h = Vector{Float64}(undef, n*(n+1)÷2)
        hessian_cm_structured!(h, obj, cctx)
        return unpack_packed(h, n)
    end
    Hd = full_hessian_mz(base_d, pcx_d.ctx_cm, pcx_d.cctx)
    Hp = full_hessian_mz(base_p, pcx_p.ctx_cm, pcx_p.cctx)
    eCore = maximum(abs.(Hd[1:ncore_core,1:ncore_core] .- Hp[1:ncore_core,1:ncore_core]))
    eFull = maximum(abs.(Hd .- Hp))
    lp("  $label: ncore_core=$ncore_core NCORE_ext=$(pcx_d.cctx.NCORE)  max|ΔH_EEcore|=$eCore  max|ΔH_full|=$eFull")
    check("$label: true core (winner-pair) sub-block agrees", eCore < 1e-8)
    check("$label: full assembled Hessian agrees", eFull < 1e-8)
end

println("="^90)
println("SECTION D: origin-ZC -- H_EE (winner-pair) + H_ER/H_RR (dense, partitioned) full block")
println("="^90)
for (K_mean, K_pair, label) in [(1, 0, "K1_mean_only"), (1, 1, "K1_mean_zc")]
    local layout = OriginByPowerLayout(ctx.D, K_mean, K_pair)
    local pcx_d = build_originzc_production_context(ctx, CS, layout)
    local pcx_p = build_originzc_production_context(ctx, CS, layout)
    pcx_d.octx.core_hessian_backend = :dense_reference
    pcx_p.octx.core_hessian_backend = :exact_winner_pair_parallel
    pcx_p.octx.core_hessian_workers = 2
    nu0 = vcat([fill(Float64(factorial(k)), ctx.D) for k in 1:K_mean]...)

    base_d = archOZ_base_state(x_free_calib, nu0, pcx_d.ctx_cm)
    base_p = archOZ_base_state(x_free_calib, nu0, pcx_p.ctx_cm)
    check("originZC $label: both backends feasible", base_d.inner_status in (0,-100,-101,-103) && base_p.inner_status in (0,-100,-101,-103))
    check("originZC $label: dual solutions agree", isapprox(base_d.ζstar, base_p.ζstar; rtol=1e-8) && isapprox(base_d.λstar, base_p.λstar; rtol=1e-6, atol=1e-9))

    local n = pcx_d.ctx_cm.obj.outer_constr_index
    local NCORE = pcx_d.octx.NCORE
    function full_hessian_oz(base, ctx_cm, octx)
        obj = ctx_cm.obj
        x = vcat(base.ζstar, base.λstar)
        _archC_prep_for_hessian!(obj, x)
        h = Vector{Float64}(undef, n*(n+1)÷2)
        cb = archA_partitioned_hess_cb_builder(octx)
        # Exercise the SAME closure KNITRO would call, with a minimal fake evalRequest/evalResult.
        fake_req = (x = x,)
        fake_res = (hess = h,)
        cb(nothing, nothing, fake_req, fake_res, obj)
        return unpack_packed(h, n)
    end
    Hd = full_hessian_oz(base_d, pcx_d.ctx_cm, pcx_d.octx)
    Hp = full_hessian_oz(base_p, pcx_p.ctx_cm, pcx_p.octx)
    eEE = maximum(abs.(Hd[1:NCORE,1:NCORE] .- Hp[1:NCORE,1:NCORE]))
    eFull = maximum(abs.(Hd .- Hp))
    lp("  originZC $label: NCORE=$NCORE n=$n  max|ΔH_EE|=$eEE  max|ΔH_full|=$eFull")
    check("originZC $label: H_EE block agrees", eEE < 1e-8)
    check("originZC $label: full assembled Hessian agrees (H_EE+H_ER+H_RR partition == monolithic dense)", eFull < 1e-8)
end

println("="^90)
if isempty(FAILURES)
    println("ALL SHARED-H_EE D=4 GATES PASSED")
else
    println("FAILURES (", length(FAILURES), "): ")
    for f in FAILURES; println("  - ", f); end
    exit(1)
end
