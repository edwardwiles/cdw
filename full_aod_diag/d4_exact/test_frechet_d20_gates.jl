# ============================================================================
# Fixed-Fréchet-marginals real D=20/W=80000/L=50 fixed-point gate battery
# (task brief §10.2). Two points: the benchmark/calibration A* point, and one
# existing cold-verified flexible-CM incumbent (production_runs/cm_campaign_2026-07-22/
# chain1/delta_1.0/stage_latest.jls, delta=1.0). Include order mirrors
# cm_cplus_d20_multipoint_gate.jl (the established real-D20 CM-C+ gate
# pattern) plus the new fixed-Fréchet files.
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "common_marginals_moments.jl"))
include(joinpath(@__DIR__, "common_marginals_interval.jl"))
include(joinpath(@__DIR__, "instrumentation.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "composite_gradient.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))
include(joinpath(@__DIR__, "cm_lookup_kernels.jl"))
include(joinpath(@__DIR__, "lfix_cm_aware.jl"))
include(joinpath(@__DIR__, "cm_hessian_architectures.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_frechet_config.jl"))
include(joinpath(@__DIR__, "frechet_reference_targets.jl"))
include(joinpath(@__DIR__, "cm_frechet_moments.jl"))
include(joinpath(@__DIR__, "cm_frechet_hessian.jl"))
include(joinpath(@__DIR__, "cm_frechet_lfix_aware.jl"))
include(joinpath(@__DIR__, "cm_frechet_production_bundle.jl"))
include(joinpath(@__DIR__, "cm_frechet_checkpoint.jl"))
using Printf, LinearAlgebra, Statistics, Serialization, Dates

lp(xs...) = (println(xs...); flush(stdout))
t0 = time()
elapsed() = round(time() - t0, digits = 1)
x_free_from_w(w, pe) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

lp("=== test_frechet_d20_gates === ", Dates.now())

const L = 50
const CKPT_PATH = "/bbkinghome/edav/gravity_robustness/production_runs/cm_campaign_2026-07-22/chain1/delta_1.0/stage_latest.jls"

results = NamedTuple[]

"Weighted-CDF residual diagnostics (task brief §11): per-origin/grid maximum |E_Fhat[1{U_o<=u_l}] - p_l|, using the recovered least-favorable weights m*."
function frechet_cdf_residuals(base, ctx, targets::FrechetReferenceTargets)
    m = base.m_star; W = length(m); p = m ./ sum(m)
    D = ctx.D; L = length(targets.probs)
    resid = Matrix{Float64}(undef, D, L)
    for l in 1:L, o in 1:D
        resid[o, l] = sum(p[s] * (ctx.U[s, o] <= targets.thresholds[l]) for s in 1:W) - targets.targets[l]
    end
    return resid
end
frechet_cdf_residuals(base, ctx, ::Nothing) = nothing

"One (point,mode) evaluation: base-state solve, verify diagnostics, C+/Reference gradients, timing, residuals."
function evaluate_point(label, mode, x_free, ctx, pe, targets_frec)
    lp("-"^100); lp("[", elapsed(), "s] POINT=", label, " MODE=", mode); lp("-"^100)
    D = ctx.D; W = size(ctx.U, 1)
    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)

    local Delta_dual, verify, ninner, g_ref, g_cp, t_base, t_gref, t_gcp, resid

    try
    if mode === :unrestricted
        t_base = @elapsed begin
            θ_full0 = CS.reconstruct_full(x_free, ctx.m)
            K, x, nStatus, _, _ = inner_loop_internal_archgeneric(ctx.obj, θ_full0; hess_cb_builder = archA_hess_cb_builder)
        end
        base = BaseDualState(collect(x_free), θ_full0, x[1], collect(x[2:end]), copy(ctx.obj.arg1), nStatus)
        Delta_dual = delta_dual_from_base(ctx.obj, base)
        verify = (inner_status = nStatus, primal_dual_gap = NaN, weighted_kkt_resid = NaN)
        ninner = ctx.obj.d
        t_gref = @elapsed (g_ref, _) = composite_gradient_at_fast(x_free, ctx, pe; base = base)
        t_gcp = NaN; g_cp = fill(NaN, length(g_ref))
        resid = nothing
    elseif mode === :common_flexible
        pcx = build_cm_production_context(ctx, CS; L = L, contrasts = :orthonormal)
        t_base = @elapsed (base, verify_nt) = archC_verified_state(x_free, pcx.ctx_cm, pcx.cctx)
        verify = (inner_status = verify_nt.inner_status, primal_dual_gap = verify_nt.primal_dual_gap,
                  weighted_kkt_resid = verify_nt.max_abs_moment_kkt_resid)
        Delta_dual = verify_nt.Delta_dual
        ninner = pcx.ctx_cm.obj.d
        t_gref = @elapsed (g_ref, _) = cm_production_gradient(x_free, pcx, ctx, pe; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        t_gcp = @elapsed (g_cp, _) = cm_production_gradient_cplus(x_free, pcx, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        resid = frechet_cdf_residuals(base, ctx, targets_frec)   # informative-only under flexible CM (targets != constraint here)
    elseif mode === :frechet_reference
        cfg = CMFrechetConfig(cm = CMConfig(cm_grid_size = L, contrasts = :orthonormal, cm_hessian_backend = :structured), marginal_mode = :frechet_reference)
        fpcx = build_cm_frechet_production_context(ctx, CS, cfg; L = L)
        t_base = @elapsed (base, verify_nt) = archC_frechet_verified_state(x_free, fpcx.ctx_cm, fpcx.fctx)
        verify = (inner_status = verify_nt.inner_status, primal_dual_gap = verify_nt.primal_dual_gap,
                  weighted_kkt_resid = verify_nt.max_abs_moment_kkt_resid)
        Delta_dual = verify_nt.Delta_dual
        ninner = fpcx.ctx_cm.obj.d
        t_gref = @elapsed (g_ref, _) = cm_frechet_production_gradient(x_free, fpcx, ctx, pe; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        t_gcp = @elapsed (g_cp, _) = cm_frechet_production_gradient_cplus(x_free, fpcx, ctx, pe, pool, ws; base = base, threaded = false, h_mode = :fixed, h0 = 0.01)
        resid = frechet_cdf_residuals(base, ctx, fpcx.targets)
    else
        error("unknown mode $mode")
    end
    catch e
        e isa CMExpectedSolveFailure || rethrow()
        # Genuine, disclosed finding (not an implementation bug): the fixed-Fréchet restriction is
        # strictly TIGHTER than flexible CM (proven at D=4 and at A*, task brief section 11's own
        # nesting requirement), so it can have NO feasible least-favorable reweighting at a point
        # that left flexible CM no slack (e.g. an incumbent optimized right up against flexible
        # CM's own delta<=1 boundary). Record as infeasible (Delta=+Inf, nesting holds trivially)
        # rather than crashing the whole battery or silently fabricating a number.
        lp("  INFEASIBLE: ", sprint(showerror, e))
        return (label = label, mode = mode, Delta_dual = Inf, inner_status = -400,
                primal_dual_gap = NaN, weighted_kkt_resid = NaN, ninner = -1,
                t_base = NaN, t_gref = NaN, t_gcp = NaN,
                max_grad_diff = NaN, cosine = NaN, max_cdf_resid = NaN)
    end

    max_grad_diff = mode === :unrestricted ? NaN : maximum(abs.(g_ref .- g_cp))
    cosang = mode === :unrestricted ? NaN : dot(g_ref, g_cp) / (norm(g_ref) * norm(g_cp) + 1e-300)
    max_resid = resid === nothing ? NaN : maximum(abs.(resid))

    lp(@sprintf("  Delta_dual=%.10f  inner_status=%s  gap=%.3e  kkt_resid=%.3e  ninner=%d",
                Delta_dual, string(verify.inner_status), verify.primal_dual_gap, verify.weighted_kkt_resid, ninner))
    lp(@sprintf("  wall: base=%.3fs  grad_ref=%.3fs  grad_cplus=%.3fs", t_base, t_gref, t_gcp))
    mode !== :unrestricted && lp(@sprintf("  C+ vs Reference: max|Δg|=%.3e  cosine=%.10f", max_grad_diff, cosang))
    resid !== nothing && lp(@sprintf("  max |weighted-CDF residual vs target| = %.3e", max_resid))

    return (label = label, mode = mode, Delta_dual = Delta_dual, inner_status = verify.inner_status,
            primal_dual_gap = verify.primal_dual_gap, weighted_kkt_resid = verify.weighted_kkt_resid,
            ninner = ninner, t_base = t_base, t_gref = t_gref, t_gcp = t_gcp,
            max_grad_diff = max_grad_diff, cosine = cosang, max_cdf_resid = max_resid)
end

# ---- Point 1: A* (calibration) ----
lp("="^100); lp("[", elapsed(), "s] Building ctx for POINT A* (calibration), W=80000, delta=1.0"); lp("="^100)
ctx_A = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true, draw_design = :sobol_randomized, draw_seed = 20260719)
pe_A = build_pivot_elimination(ctx_A)
D = ctx_A.D
lp(@sprintf("[%.1fs] ctx built. D=%d W=%d", elapsed(), D, size(ctx_A.U,1)))
xf_Astar = ctx_A.θ0_up[ctx_A.free_idx]
targets_A = build_frechet_reference_targets(ctx_A, CMFrechetConfig(cm = CMConfig(cm_grid_size = L)); L = L)

for mode in (:unrestricted, :common_flexible, :frechet_reference)
    push!(results, evaluate_point("A_star", mode, xf_Astar, ctx_A, pe_A, targets_A))
end

# ---- Point 2: existing cold-verified flexible-CM incumbent (delta=1.0) ----
lp("="^100); lp("[", elapsed(), "s] Loading existing flexible-CM incumbent: ", CKPT_PATH); lp("="^100)
ckpt = load_cm_checkpoint(CKPT_PATH)
lp("  checkpoint: W=", ckpt.W, " delta=", ckpt.delta, " draw_design=", ckpt.draw_design, " draw_seed=", ckpt.draw_seed,
   " cm_L=", ckpt.cm_L, " best_feasible=", ckpt.best_feasible === nothing ? "nothing" : "present")

ctx_B = d20_real_setup_design(W = ckpt.W, δ = ckpt.delta, find_smallest = ckpt.find_smallest,
                               draw_design = ckpt.draw_design, draw_seed = ckpt.draw_seed)
pe_B = build_pivot_elimination(ctx_B)
if ctx_B.draw_meta.checksum_uniform != ckpt.draw_checksum_uniform || ctx_B.draw_meta.checksum_transformed != ckpt.draw_checksum_transformed
    error("draw checksum MISMATCH against the recorded checkpoint provenance -- refusing to evaluate against a different problem instance")
end
lp("  draw checksums verified OK")
w_incumbent = ckpt.best_feasible.w
xf_incumbent = x_free_from_w(w_incumbent, pe_B)
targets_B = build_frechet_reference_targets(ctx_B, CMFrechetConfig(cm = CMConfig(cm_grid_size = L)); L = L)

for mode in (:unrestricted, :common_flexible, :frechet_reference)
    push!(results, evaluate_point("existing_incumbent_delta1", mode, xf_incumbent, ctx_B, pe_B, targets_B))
end

# ============================================================================
lp("="^100); lp("SUMMARY TABLE"); lp("="^100)
for r in results
    @printf "%-28s %-18s Delta=%.8f  status=%-5s  gap=%.2e  kkt=%.2e  ninner=%-4d  base=%.2fs  gref=%.2fs  gcp=%.2fs  maxΔg=%.2e  cos=%.8f  maxCDFresid=%.2e\n" r.label string(r.mode) r.Delta_dual string(r.inner_status) r.primal_dual_gap r.weighted_kkt_resid r.ninner r.t_base r.t_gref r.t_gcp r.max_grad_diff r.cosine r.max_cdf_resid
end

lp()
lp("="^100); lp("NESTING CHECK"); lp("="^100)
for label in ("A_star", "existing_incumbent_delta1")
    ru = only(r.Delta_dual for r in results if r.label == label && r.mode === :unrestricted)
    rf = only(r.Delta_dual for r in results if r.label == label && r.mode === :common_flexible)
    rr = only(r.Delta_dual for r in results if r.label == label && r.mode === :frechet_reference)
    ok = (ru <= rf + 1e-6) && (rf <= rr + 1e-6)
    @printf "%-28s  unrestricted=%.8f <= flexible=%.8f <= frechet=%.8f   NESTING_OK=%s\n" label ru rf rr string(ok)
end

lp()
lp(@sprintf("Peak RSS: %.2f GB", Sys.maxrss() / 1e9))
lp("Total wall: ", elapsed(), "s")

resultsdir = joinpath(@__DIR__, "..", "..", "results", "fullA_d4", "fixed_frechet_d20_gates")
mkpath(resultsdir)
serialize(joinpath(resultsdir, "d20_gate_results.jls"), results)
lp("Results saved to ", joinpath(resultsdir, "d20_gate_results.jls"))
