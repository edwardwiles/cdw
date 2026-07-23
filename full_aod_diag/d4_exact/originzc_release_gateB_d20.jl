# ============================================================================
# Release Gate B (ORIGIN_ZC_K12_PRODUCTION_RELEASE_2026-07-23.md, section 8):
# two real D=20 cold points --
#   Point A (calibrated benchmark A*), K=1
#   Point B (existing cold-verified unrestricted incumbent, delta=1.0,
#            production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/
#            chain_A/stage3_d1.0), K=2
# -- cache-disabled cold inner solve (this arm has no exact-point cache at
# all, so a fresh process invocation IS cache-disabled by construction,
# same property cm_cold_verify.jl relies on) + complete C+ vs Reference
# gradient, nu initialized from the actual frozen draw moments (the SAME
# formula the production stage runner uses, task brief Section 7) -- not
# the expensive Optim/LBFGS profile-to-convergence the experiment branch's
# own d20_originzc_fixedpoint_gates.jl ran (that script is diagnostic-only
# and not rerun here; this is a lighter, single-point release check).
#
# Points A and B, and the archived incumbent path, are IDENTICAL to the ones
# tested during the experiment's own D=20 fixed-point gates (integration
# report Section 5) -- no new points are added, per the release brief's
# explicit instruction.
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
include(joinpath(@__DIR__, "direction_bounds.jl"))
using Printf, LinearAlgebra, Statistics, Dates, Serialization

lp(xs...) = (println(xs...); flush(stdout))

function peak_rss_mb()
    for line in eachline("/proc/self/status")
        startswith(line, "VmHWM:") && return parse(Float64, split(line)[2]) / 1024
    end
    return NaN
end

# Minimal local copy of D20Checkpoint, identical to d20_originzc_fixedpoint_gates.jl's own
# (same rationale: the full production driver file is not includable here, ambiguity conflict).
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

lp("="^100)
lp("Release Gate B: origin-ZC D=20 cold points -- ", Dates.now())
lp("="^100)

ctx = d20_real_setup_design(W = 80000, δ = 1.0, find_smallest = true, draw_design = :pseudorandom, draw_seed = 20260719)
pe = build_pivot_elimination(ctx)
D = ctx.D
lp("D=", D, " W=", size(ctx.U, 1), " draw checksums: uniform=", ctx.draw_meta.checksum_uniform,
   " transformed=", ctx.draw_meta.checksum_transformed)

gp0 = frechet_benchmark_gp(ctx)
z0 = log.(reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D))
zfree0 = pivot_reduce(z0, pe)
xfA = x_free_from_w(vcat(gp0, zfree0), pe)
lp("Point A (benchmark A*): gp=", gp0)

incumbent_path = "/bbkinghome/edav/gravity_robustness/production_runs/2026-07-22/fullA_exact_unrestricted_670eac4/chain_A/stage3_d1.0/polish/chainA_s3_polish_latest.jls"
ckB = load_checkpoint(incumbent_path)
@assert ckB.W == 80000 && ckB.draw_design == :pseudorandom && ckB.draw_seed == 20260719 && ckB.delta == 1.0
wB = ckB.best_feasible.w
xfB = x_free_from_w(wB, pe)
lp("Point B (cold-verified unrestricted incumbent, chain_A/stage3_d1.0): gp=", wB[1],
   " checkpoint-recorded Delta=", ckB.best_feasible.Delta)

function run_point(label::String, xf::AbstractVector, K::Int)
    lp()
    lp("="^100)
    lp("Point ", label, "  K_mean=K_pair=", K)
    lp("="^100)
    layout = OriginByPowerLayout(D, K, K)
    pcx = build_originzc_production_context(ctx, CS, layout)
    lp("  inner dimension = ", pcx.ctx_cm.obj.outer_constr_index, "  n_eta=", n_eta(layout))

    nu0 = Vector{Float64}(undef, n_eta(layout))
    for k in 1:K
        Uk = ctx.U .^ k
        for o in 1:D
            nu0[target_index(layout, o, k)] = mean(@view Uk[:, o])
        end
    end

    t0 = time()
    K_, base, verify = cm_originzc_production_value_verified(xf, nu0, pcx)
    t_solve = time() - t0
    lp("  cold inner solve: nStatus=", verify.inner_status, " verified_success=", is_verified_success(verify),
       " Delta_dual=", verify.Delta_dual, " primal_dual_gap=", verify.primal_dual_gap,
       " max_abs_moment_kkt_resid=", verify.max_abs_moment_kkt_resid, " wall=", round(t_solve, digits = 2), "s")

    for k in 1:K
        Uk = ctx.U .^ k
        νtargets = mean_targets(layout, nu0, k, D)
        r = recovered_mean_residuals_origin(base.m_star, Uk, νtargets)
        lp("  level k=", k, "  mean residual max|.|=", maximum(abs.(r)))
    end
    for k in 1:K
        pairs = packed_pair_index(D)
        Zpair = pcx.aug.Zpairraw_all[k]
        νprod = pair_targets(layout, nu0, k, D)
        r = recovered_pair_residuals_origin(base.m_star, Zpair, νprod)
        lp("  level k=", k, "  pair residual   max|.|=", maximum(abs.(r)))
    end

    t_g0 = time()
    W = size(ctx.U, 1)
    pool = build_grad_workspace_pool(W)
    ws = build_lfix_factorized_workspace(D, W)
    g_ref, _ = cm_originzc_production_gradient(xf, nu0, pcx, ctx, pe; base = base, verify = verify,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    g_cplus, _ = cm_originzc_production_gradient_cplus(xf, nu0, pcx, ctx, pe, pool, ws; base = base, verify = verify,
        threaded = true, h_mode = :cached, bandwidth_cache = Dict{Int,Float64}())
    t_grad = time() - t_g0
    D2 = D^2
    econ_ref = g_ref[1:D2]; econ_cplus = g_cplus[1:D2]
    cossim = dot(econ_ref, econ_cplus) / (norm(econ_ref) * norm(econ_cplus))
    maxdiff_econ = maximum(abs.(econ_ref .- econ_cplus))
    eta_ref = g_ref[D2+1:end]; eta_cplus = g_cplus[D2+1:end]
    maxdiff_eta = maximum(abs.(eta_ref .- eta_cplus))
    eta_finite = all(isfinite, eta_cplus) && all(isfinite, eta_ref)
    lp("  C+ vs Reference: econ cosine=", cossim, " econ max|diff|=", maxdiff_econ, " eta max|diff|=", maxdiff_eta,
       "  eta gradient finite=", eta_finite, "  grad wall=", round(t_grad, digits = 2), "s")

    rss = peak_rss_mb()
    lp("  peak RSS so far = ", round(rss, digits = 1), " MB")

    verified_ok = is_verified_success(verify)
    verified_ok || error("Point $label K=$K: cold inner solve did NOT pass is_verified_success")
    eta_finite || error("Point $label K=$K: non-finite eta gradient component(s)")
    maxdiff_econ < 1e-8 || error("Point $label K=$K: C+ vs Reference econ max|diff|=$(maxdiff_econ) exceeds release tolerance")
    return (Delta_dual = verify.Delta_dual, cossim = cossim, maxdiff_econ = maxdiff_econ,
            maxdiff_eta = maxdiff_eta, peak_rss_mb = rss, wall_solve = t_solve, wall_grad = t_grad)
end

resA1 = run_point("A", xfA, 1)
resB2 = run_point("B", xfB, 2)

lp()
lp("="^100)
lp("SUMMARY")
lp("="^100)
@printf "  Point A K=1: Delta_dual=%.6f cos=%.10f maxdiff_econ=%.2e maxdiff_eta=%.2e peak_rss=%.0fMB\n" resA1.Delta_dual resA1.cossim resA1.maxdiff_econ resA1.maxdiff_eta resA1.peak_rss_mb
@printf "  Point B K=2: Delta_dual=%.6f cos=%.10f maxdiff_econ=%.2e maxdiff_eta=%.2e peak_rss=%.0fMB\n" resB2.Delta_dual resB2.cossim resB2.maxdiff_econ resB2.maxdiff_eta resB2.peak_rss_mb
lp()
lp("RELEASE GATE B: PASS")
lp("GATE_B_DONE")
