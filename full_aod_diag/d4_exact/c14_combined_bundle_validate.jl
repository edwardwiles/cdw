# Continuation 12 (D=20 prep): combine the three independently-validated winners into one bundle
# -- interval basis (common_marginals_interval.jl, best conditioning+sparsity per the conditioning
# subagent) + Architecture C structured Hessian (cm_hessian_architectures.jl, up to 2.7x faster
# full inner solve per the hessian-arch subagent) -- and equivalence-test the COMBINATION against
# the original trusted dense reference (cumulative-CDF basis + dense BLAS Hessian, Architecture A).
# The lookup FG kernels (cm_lookup_kernels.jl) are deliberately NOT included in this first combined
# pass: their own subagent found only a modest 1.2-1.3x end-to-end win at D=4 (dominated by KNITRO's
# own per-iteration overhead at this problem size), so the highest-value combination to validate
# first is interval-basis conditioning + Architecture C's Hessian speed/memory win, both of which
# matter more directly for D=20 feasibility (conditioning at large L; memory at large D).
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
using Printf, LinearAlgebra

ctx = d4_exact_setup(δ = 1.0, find_smallest = true, needs_outer_moment_jacobian = false)
x_free_calib = ctx.θ0_up[ctx.free_idx]
Random.seed!(4243)
x_free_perturbed = copy(x_free_calib)
x_free_perturbed[2:end] .*= exp.(0.05 .* randn(length(x_free_perturbed) - 1))

function reference_dense(L, x_free)
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    ctx_cm = merge(ctx, (obj = aug.obj_cm,))
    r = evaluate_fullA(x_free, ctx_cm; use_cache = false, warm = false)
    return (r = r, aug = aug)
end

function combined_cumulative_archC(L, x_free)
    # Architecture C's Hessian (build_cm_bin_ctx/hessian_cm_structured!) 2D-prefix-sums raw bin
    # contingency tables into CUMULATIVE second moments (prefix_sum_tables!) -- it was built and
    # validated by its own subagent against build_cm_augmented_obj (the CUMULATIVE/CDF basis),
    # NOT build_cm_augmented_obj_interval. This is the combination that's actually valid.
    aug = build_cm_augmented_obj(ctx, CS; L = L, contrasts = :anchored)
    obj_cm = aug.obj_cm
    cctx = build_cm_bin_ctx(ctx, aug)
    cctx_callback = archC_hess_cb_builder(cctx)
    hess_builder = (obj_arg) -> cctx_callback

    θ_full = CS.reconstruct_full(x_free, ctx.m)
    t0 = time()
    K_hard, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj_cm, θ_full;
        hess_cb_builder = hess_builder, hvp = false)
    telapsed = time() - t0

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        return (nStatus = nStatus, ok = false, telapsed = telapsed)
    end

    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, obj_cm.d)
    obj_cm.moments!(K, G, θ_full, ctx.U, obj_cm)
    ncon = obj_cm.d - obj_cm.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj_cm(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj_cm.arg1)
    Delta_primal = primal_divergence(m_weights)
    gamma_focal_prime = θ_full[3 + ctx.D]

    return (nStatus = nStatus, ok = true, telapsed = telapsed, Delta_dual = Delta_dual,
            Delta_primal = Delta_primal, gamma_focal_prime = gamma_focal_prime,
            n_fg = n_fg, n_hess = n_hess, obj_cm = obj_cm, θ_full = θ_full,
            inner_x = inner_x, aug = aug)
end

"KNOWN-MISMATCH probe (kept for the record, not a real candidate): interval-basis G paired with
Architecture C's cumulative-basis Hessian. Expected to fail (wrong Newton direction) -- run once
to confirm the diagnosis, not retried/looped."
function combined_interval_archC(L, x_free)
    aug = build_cm_augmented_obj_interval(ctx, CS; L = L, contrasts = :anchored)
    obj_cm = aug.obj_cm
    cctx = build_cm_bin_ctx(ctx, aug)
    cctx_callback = archC_hess_cb_builder(cctx)
    hess_builder = (obj_arg) -> cctx_callback   # hess_cb_builder(obj) -> callback, matching inner_loop_KNITRO_archgeneric's convention (see c13_bench_hessian_archs.jl)

    θ_full = CS.reconstruct_full(x_free, ctx.m)
    t0 = time()
    K_hard, inner_x, nStatus, n_fg, n_hess = inner_loop_internal_archgeneric(obj_cm, θ_full;
        hess_cb_builder = hess_builder, hvp = false)
    telapsed = time() - t0

    solved = nStatus in (0, -100, -101, -103)
    if !solved
        return (nStatus = nStatus, ok = false, telapsed = telapsed)
    end

    W = size(ctx.U, 1)
    K = zeros(W); G = zeros(W, obj_cm.d)
    obj_cm.moments!(K, G, θ_full, ctx.U, obj_cm)
    ncon = obj_cm.d - obj_cm.outer_constr_index + 2
    cbuf = zeros(ncon)
    fval = obj_cm(inner_x, constr = @view(cbuf[1:ncon]))
    Delta_dual = cbuf[1] / 1e10
    m_weights = copy(obj_cm.arg1)
    Delta_primal = primal_divergence(m_weights)
    gamma_focal_prime = θ_full[3 + ctx.D]

    return (nStatus = nStatus, ok = true, telapsed = telapsed, Delta_dual = Delta_dual,
            Delta_primal = Delta_primal, gamma_focal_prime = gamma_focal_prime,
            n_fg = n_fg, n_hess = n_hess, obj_cm = obj_cm, θ_full = θ_full,
            inner_x = inner_x, aug = aug)
end

println("=== Known-mismatch probe: interval basis G + Architecture C (cumulative-basis) Hessian, calibration/L=10 only ===")
let mism = combined_interval_archC(10, x_free_calib)
    @printf("interval+ArchC: status=%d ok=%s (expected FAIL -- confirms basis mismatch, not a wiring bug)\n", mism.nStatus, mism.ok)
end

println("\n=== Combined bundle (CUMULATIVE basis + Architecture C Hessian, the pairing ArchC was actually validated against) vs dense reference (cumulative + Architecture A) ===")
for (label, x_free) in (("calibration", x_free_calib), ("perturbed", x_free_perturbed))
    for L in (10, 20, 50)
        ref = reference_dense(L, x_free)
        comb = combined_cumulative_archC(L, x_free)
        if !(ref.r.inner_status in (0, -100, -101, -103)) || !comb.ok
            @printf("[%-12s L=%2d] SOLVE MISMATCH: ref_status=%d comb_status=%d\n",
                    label, L, ref.r.inner_status, comb.nStatus)
            continue
        end
        ddiff = abs(ref.r.Delta_dual - comb.Delta_dual)
        pdiff = abs(ref.r.Delta_primal - comb.Delta_primal)
        gdiff = abs(ref.r.gamma_focal_prime - comb.gamma_focal_prime)
        speedup = ref.r.elapsed.total / comb.telapsed
        @printf("[%-12s L=%2d] ref: status=%-4d D=%.6f t=%.3fs | comb: status=%-4d D=%.6f t=%.3fs | |Ddual diff|=%.2e |Dprimal diff|=%.2e |gp diff|=%.2e  speedup=%.2fx\n",
                label, L, ref.r.inner_status, ref.r.Delta_dual, ref.r.elapsed.total,
                comb.nStatus, comb.Delta_dual, comb.telapsed, ddiff, pdiff, gdiff, speedup)
    end
end
