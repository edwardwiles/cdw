# Real D=20 fixed-point gates for the origin-specific-ZC restriction
# (task brief Section 11.2). Two economic outer points (calibrated benchmark
# A*, and the existing cold-verified unrestricted incumbent at delta=1.0,
# production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/chain_A/
# stage3_d1.0), K=1 and K=2 (K_pair==K_mean per task brief Section 2's chosen
# test values). At each (point,K): initialize nu from the actual draw
# moments, profile Delta_dual over eta at fixed economic point (Optim/LBFGS,
# analytic gradient -- cm_originzc_profile.jl), record residuals, compare
# C+ vs Reference gradients, spot-check reoptimized FD derivatives, record
# wall/RSS. Needs a real KNITRO license.
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
include(joinpath(@__DIR__, "cm_production_bundle.jl"))
include(joinpath(@__DIR__, "gradient_workspace.jl"))
include(joinpath(@__DIR__, "lfix_factorized.jl"))
include(joinpath(@__DIR__, "lfix_factorized_workspace.jl"))
include(joinpath(@__DIR__, "lfix_cm_cplus.jl"))
include(joinpath(@__DIR__, "nested_quantile_grids.jl"))
include(joinpath(@__DIR__, "cm_outer_driver.jl"))
include(joinpath(@__DIR__, "cm_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_moments.jl"))
include(joinpath(@__DIR__, "cm_meanzc_config.jl"))
include(joinpath(@__DIR__, "cm_meanzc_production.jl"))
include(joinpath(@__DIR__, "cm_meanzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_target_layout.jl"))
include(joinpath(@__DIR__, "cm_originzc_moments.jl"))
include(joinpath(@__DIR__, "cm_originzc_production.jl"))
include(joinpath(@__DIR__, "cm_originzc_cplus.jl"))
include(joinpath(@__DIR__, "cm_originzc_config.jl"))
include(joinpath(@__DIR__, "cm_originzc_checkpoint.jl"))
include(joinpath(@__DIR__, "cm_originzc_profile.jl"))
include(joinpath(@__DIR__, "direction_bounds.jl"))
using Printf, LinearAlgebra, Statistics, Dates, Serialization

# Minimal local copy of D20Checkpoint (c10_d20_production_driver.jl:128-166) + load_checkpoint
# (c10_d20_production_driver.jl:210-213) -- NOT the full file, which pulls in a
# PsiObjectiveBundleDelta binding that conflicts with this script's own CM/meanzc/originzc
# include chain (confirmed live: including the whole driver file throws an ambiguity
# UndefVarError inside master_prepare_cc/PMM.jl). Only the read-only struct layout + loader are
# needed here (to read one existing checkpoint's incumbent vector); everything else in that
# file (the actual unrestricted D=20 outer-loop driver) is unused by this gate script.
struct D20Checkpoint
    schema::Int; run_id::String; label::String; branch::Symbol; find_smallest::Bool; delta::Float64
    W::Int; draw_seed::Int; g::Float64; zfree::Vector{Float64}; logA_full::Matrix{Float64}
    dual_warm_start::Vector{Float64}; bandwidth_cache::Dict{Int,Float64}; best_feasible::Any
    n_eval::Int; knitro_iter::Int; wall_elapsed::Float64; checkpoint_reason::Symbol
    screen_counts::Any
    verify_Delta_dual::Float64; verify_gravity_value::Float64; verify_max_abs_moment_kkt_resid::Float64
    verify_moment_resid_norm::Float64; solver_state_note::String
    draw_design::Symbol; draw_checksum_uniform::String; draw_checksum_transformed::String
    knitro_version::String
end
const CHECKPOINT_SCHEMA = 3
function load_checkpoint(path::AbstractString)
    ckpt = deserialize(path)::D20Checkpoint
    ckpt.schema == CHECKPOINT_SCHEMA || error("load_checkpoint($path): schema=$(ckpt.schema), expected $(CHECKPOINT_SCHEMA)")
    return ckpt
end

lp(xs...) = (println(xs...); flush(stdout))

function peak_rss_mb()
    for line in eachline("/proc/self/status")
        if startswith(line, "VmHWM:")
            return parse(Float64, split(line)[2]) / 1024
        end
    end
    return NaN
end

"Single-coordinate reoptimized central FD of Delta_dual w.r.t. eta[j] (cheap spot-check, not the full vector)."
function eta_deriv_fd_single(x_free0, νfull, ctx_cm, j::Int; h::Float64 = 1e-4)
    ηp = log.(νfull); ηp[j] += h
    ηm = log.(νfull); ηm[j] -= h
    _, _, vp = cm_originzc_value_verified_from_eta(x_free0, ηp, ctx_cm)
    _, _, vm = cm_originzc_value_verified_from_eta(x_free0, ηm, ctx_cm)
    return (vp.Delta_dual - vm.Delta_dual) / (2h)
end

lp("="^100)
lp("Origin-ZC D=20 fixed-point gates -- ", Dates.now())
lp("="^100)

t_ctx0 = time()
ctx = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
D = ctx.D
t_ctx = time() - t_ctx0
lp("context-build wall = ", round(t_ctx, digits = 2), "s   D=", D, "  W=", size(ctx.U, 1))
lp("draw checksums: uniform=", ctx.draw_meta.checksum_uniform, " transformed=", ctx.draw_meta.checksum_transformed)

# ---- Point A: calibrated benchmark A* ----
gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
zfree0 = pivot_reduce(z0, pe)
wA = vcat(gp0, zfree0)
xfA = x_free_from_w(wA, pe)
lp("Point A (benchmark A*): gp=", gp0)

# ---- Point B: existing cold-verified unrestricted incumbent, D=20/W=80000/delta=1.0 ----
incumbent_path = "/bbkinghome/edav/gravity_robustness/production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/chain_A/stage3_d1.0/polish/chainA_s3_polish_latest.jls"
ckB = load_checkpoint(incumbent_path)
@assert ckB.W == 80000 && ckB.draw_design == :pseudorandom && ckB.draw_seed == 20260719 && ckB.delta == 1.0
wB = ckB.best_feasible.w
xfB = x_free_from_w(wB, pe)
lp("Point B (cold-verified unrestricted incumbent, chain_A/stage3_d1.0): gp=", wB[1],
   " checkpoint-recorded Delta=", ckB.best_feasible.Delta, " kappa=", 1 - wB[1]^(ctx.σ / (ctx.σ - 1)))

# ---- Re-cold-verify BOTH points under the plain unrestricted problem in THIS ctx ----
lp()
lp("-- cold-verifying both economic points under the UNRESTRICTED (no-restriction) problem --")
for (label, xf) in [("A (benchmark)", xfA), ("B (incumbent)", xfB)]
    obj0 = ctx.obj
    θ_full = CS.reconstruct_full(xf, ctx.m)
    _, xu, nStatusU, _, _ = inner_loop_internal_archgeneric(obj0, θ_full; hess_cb_builder = archA_hess_cb_builder)
    G = CS.select_G_from_H(obj0, obj0.H)
    ncon = obj0.d - obj0.outer_constr_index + 2
    cbuf = zeros(ncon)
    obj0(xu, constr = @view(cbuf[1:ncon]))
    Δu = cbuf[1] / 1e10
    lp("  ", label, ": nStatus=", nStatusU, " Delta_unrestricted=", Δu)
end

results = Dict{Tuple{String,Int},Any}()

for (label, xf) in [("B", xfB)], K in (1, 2)
    lp()
    lp("="^100)
    lp("Point ", label, "  K_mean=K_pair=", K)
    lp("="^100)
    layout = OriginByPowerLayout(D, K, K)
    t_pcx0 = time()
    pcx = build_originzc_production_context(ctx, CS, layout)
    t_pcx = time() - t_pcx0
    lp("  inner dimension (outer_constr_index) = ", pcx.ctx_cm.obj.outer_constr_index, "  n_eta=", n_eta(layout),
       "  n_mean=", pcx.aug.n_mean, "  n_pair=", pcx.aug.n_pair, "  pcx-build wall=", round(t_pcx, digits = 2), "s")

    # init nu from actual draw moments
    nu0 = Vector{Float64}(undef, n_eta(layout))
    for k in 1:K
        Uk = ctx.U .^ k
        for o in 1:D
            nu0[target_index(layout, o, k)] = mean(@view Uk[:, o])
        end
    end
    η0 = log.(nu0)
    lp("  nu0 range (level 1): [", minimum(nu0[1:D]), ", ", maximum(nu0[1:D]), "]",
       K >= 2 ? "  (level 2): [$(minimum(nu0[D+1:2D])), $(maximum(nu0[D+1:2D]))]" : "")

    t_profile0 = time()
    res, last = profile_eta_originzc(xf, η0, pcx; iterations = 100, show_trace = false)
    t_profile = time() - t_profile0
    base, verify, νfull_star = last.base, last.verify, last.νfull
    lp("  profiling: converged=", Optim.converged(res), " iterations=", Optim.iterations(res),
       " wall=", round(t_profile, digits = 1), "s")
    lp("  Delta_dual(profiled) = ", verify.Delta_dual, "  m_mean=", verify.m_mean,
       " primal_dual_gap=", verify.primal_dual_gap, " kkt_resid=", verify.max_abs_moment_kkt_resid)

    # mean/pair residuals at the profiled optimum
    for k in 1:K
        Uk = ctx.U .^ k
        νtargets = mean_targets(layout, νfull_star, k, D)
        r = recovered_mean_residuals_origin(base.m_star, Uk, νtargets)
        lp("  level k=", k, "  mean residual max|.|=", maximum(abs.(r)))
    end
    for k in 1:K
        pairs = packed_pair_index(D)
        Zpair = pcx.aug.Zpairraw_all[k]
        νprod = pair_targets(layout, νfull_star, k, D)
        r = recovered_pair_residuals_origin(base.m_star, Zpair, νprod)
        lp("  level k=", k, "  pair residual   max|.|=", maximum(abs.(r)))
    end

    # C+ vs Reference full gradient comparison at the profiled optimum
    t_grad0 = time()
    W = size(ctx.U, 1)
    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)
    g_ref, _ = cm_originzc_production_gradient(xf, νfull_star, pcx, ctx, pe; base = base, verify = verify,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    g_cplus, _ = cm_originzc_production_gradient_cplus(xf, νfull_star, pcx, ctx, pe, pool, ws; base = base, verify = verify,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    t_grad = time() - t_grad0
    D2 = D^2
    econ_ref = g_ref[1:D2]; econ_cplus = g_cplus[1:D2]
    cossim = dot(econ_ref, econ_cplus) / (norm(econ_ref) * norm(econ_cplus))
    maxdiff_econ = maximum(abs.(econ_ref .- econ_cplus))
    eta_ref = g_ref[D2+1:end]; eta_cplus = g_cplus[D2+1:end]
    maxdiff_eta = maximum(abs.(eta_ref .- eta_cplus))
    lp("  C+ vs Reference: econ cosine=", cossim, " econ max|diff|=", maxdiff_econ, " eta max|diff|=", maxdiff_eta,
       "  gradient-compute wall=", round(t_grad, digits = 1), "s")

    # reoptimized FD spot-check: 1 coordinate for K=1, 2 (different origin+power) for K=2
    check_idxs = K == 1 ? [1] : [1, target_index(layout, 2, 2)]
    for j in check_idxs
        g_fd = eta_deriv_fd_single(xf, νfull_star, pcx.ctx_cm, j; h = 1e-4)
        g_an = d_delta_dual_d_eta_origin_vec(base.λstar, pcx.aug, νfull_star; mean_m = verify.m_mean)[j]
        lp("  eta[", j, "]  analytic=", g_an, "  FD(reoptimized)=", g_fd, "  |diff|=", abs(g_an - g_fd))
    end

    rss = peak_rss_mb()
    lp("  peak RSS so far = ", round(rss, digits = 1), " MB")

    results[(label, K)] = (n_inner = pcx.ctx_cm.obj.outer_constr_index, n_eta = n_eta(layout),
        pcx_wall = t_pcx, profile_wall = t_profile, profile_iterations = Optim.iterations(res),
        grad_wall = t_grad, Delta_dual = verify.Delta_dual, m_mean = verify.m_mean,
        cossim_econ = cossim, maxdiff_econ = maxdiff_econ, maxdiff_eta = maxdiff_eta, peak_rss_mb = rss)
end

lp()
lp("="^100)
lp("SUMMARY")
lp("="^100)
for (label, K) in [("B", 1), ("B", 2)]
    r = results[(label, K)]
    @printf "  point=%s K=%d  n_inner=%d n_eta=%d  Delta=%.6f  pcx_wall=%.1fs profile_wall=%.1fs(%d iters) grad_wall=%.1fs  peak_rss=%.0fMB  C+vsRef: cos=%.10f maxdiff_econ=%.2e maxdiff_eta=%.2e\n" label K r.n_inner r.n_eta r.Delta_dual r.pcx_wall r.profile_wall r.profile_iterations r.grad_wall r.peak_rss_mb r.cossim_econ r.maxdiff_econ r.maxdiff_eta
end
lp()
lp("ALL D=20 FIXED-POINT GATES DONE -- ", Dates.now())
