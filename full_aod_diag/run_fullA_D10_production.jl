# ============================================================================
# Full-A production driver, D configurable (default 10): exact-point cache +
# free-only ForwardDiff envelope gradient (mu genuinely removed from the
# differentiated/optimized vector) + tariff-residualized analytic gravity
# gradient. Same wiring as run_fullA_D4_production.jl, validated there
# (gradient matches dense-Jacobian to relerr 4-9e-16, FD to ~1e-11); this
# file builds its own D=10-scale context directly (setup_context.jl's
# build_ad_context() hardcodes D=4) and supports env-var configuration for
# smoke tests / the batch driver:
#   DVAL            - D (default 10)
#   OUTER_OPT_FILE  - KNITRO outer options file (default csw_outer_25.opt)
#   DELTA_GRID      - comma-separated delta values (default "1.0")
#   BOUND           - "lower", "upper", or "both" (default "both")
#
#   julia --project=. full_aod_diag/run_fullA_D10_production.jl
#   DVAL=10 OUTER_OPT_FILE=full_aod_diag/csw_outer_smoke2.opt BOUND=lower \
#     julia --project=. full_aod_diag/run_fullA_D10_production.jl   # smoke test
# ============================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "setup/include_setup.jl")); include(joinpath(ROOT, "prestep/include_prestep.jl"))
include(joinpath(ROOT, "prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT, "moments/include_moments.jl"))
include(joinpath(ROOT, "cc_algo/include_cc_algo.jl")); include(joinpath(ROOT, "lfd/include_lfd.jl")); include(joinpath(ROOT, "misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity
include(joinpath(@__DIR__, "moments_gammanorm.jl"))
CS.include(joinpath(@__DIR__, "PsiObjectiveBundleImplicitMethodB_fullA.jl"))
include(joinpath(@__DIR__, "gravity_tariff.jl"))
include(joinpath(@__DIR__, "ad_benchmark", "derivative_core.jl"))

const DVAL = parse(Int, get(ENV, "DVAL", "10"))
const OUTER_OPT = get(ENV, "OUTER_OPT_FILE", joinpath(@__DIR__, "csw_outer_25.opt"))
const INNER_OPT = joinpath(@__DIR__, "ek_inner.opt")
const BOUND_ARG = get(ENV, "BOUND", "both")

params = (server=1,user=2,fakeData=1,DFake=DVAL,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so = master_setup(params); up = (; params..., D = so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps = master_prestep(so.data, so.counters, up); pp = master_prepare_cc(so.data, so.counters, ps, up)
@unpack θ_initial, θ_initial_up, U, γ, outer_constr_index, nTotalMoments, complement_index, inequality_index = pp
D = so.D; @assert D == DVAL
bi = params.baseIndex; σ = params.σHat; μHat = γ.μHat
Aod_offset = 3 + D

θ0_up = build_theta_gammanorm(θ_initial_up, D, bi, μHat, σ)
bounds = theoretical_gammaprime_bounds(γ, σ)
θ0_up[3+D] = clamp(θ0_up[3+D], bounds.γp_lo, bounds.γp_hi)

θ_lo = (θ0_up .* 0.0001)[:]; θ_hi = (θ0_up .* 10000)[:]
θ_lo[2] = θ0_up[2]; θ_hi[2] = θ0_up[2]
θ_lo[1] = θ0_up[1]; θ_hi[1] = θ0_up[1]
for d in 1:D
    θ_lo[2+d] = θ0_up[2+d]; θ_hi[2+d] = θ0_up[2+d]
end
θ_lo[3+D] = bounds.γp_lo; θ_hi[3+D] = bounds.γp_hi

println(">>> D=$D  l_full=$(length(θ0_up))  mu=$(θ0_up[1])  sigma=$(θ0_up[2])  OUTER_OPT=$OUTER_OPT")

l_full = length(θ0_up)
free_idx = vcat(3 + D, collect(Aod_offset+1:Aod_offset+D^2))
fixed_idx = vcat(1, 2, collect(3:2+D))
fixed_vals = θ0_up[fixed_idx]
fpmap = CS.FreeParamMap(l_full, free_idx, fixed_idx, fixed_vals)
println(">>> n_free = ", CS.n_free(fpmap), " (expect ", 1 + D^2, ")   mu_in_free_vector=", (1 in free_idx))
@assert CS.n_free(fpmap) == 1 + D^2
@assert CS.round_trip_check(θ0_up, fpmap)
@assert !(1 in free_idx) "mu must NOT be in the free vector"

Aod_free_pos = [1 + (d - 1) * D + o for o in 1:D, d in 1:D]
τ = γ.τ
q_tilde, N_obs = precompute_q_tilde(τ)
@assert N_obs == D^2

function make_fullA_obj(δval, find_smallest)
    # needs_outer_moment_jacobian=false: this driver's gradient path (make_div_grad_fn! below, via
    # envelope_scalar_div_ctx) and gravity_grad_fn! (gravity_tariff.jl) never call this bundle's
    # callable with a nonempty theta, so the legacy dense N x (d+2) x l jac_h tensor
    # (cc_algo/PsiObjectiveBundle.jl) is never populated or read -- confirmed by a runtime counter
    # audit (docs/fullA_jach_audit.md: JAC_H_POPULATE_COUNT/JAC_H_THETA_BRANCH_COUNT both 0 across a
    # full evaluate_fullA + L_fix-incremental + short outer-loop exercise) and validated bit-identical
    # against the jac_h-allocated default at D=4/6/8 (same audit doc). Skipping the allocation saves
    # ~28MB at D=4 scaling to ~300MB+ at D=8-10 (8*Jac_W*(d+2)*l bytes) plus its one-time
    # allocation+zeroing wall time, with zero observed behavior change.
    PsiObjectiveBundleImplicit(δ = δval, find_smallest = find_smallest, γ = γ,
        (moments!) = EK_moments_gammanorm_directgp!, moments_jacobian! = error, d = nTotalMoments,
        outer_constr_index = outer_constr_index, inequality_index = inequality_index,
        complement_index = complement_index, l = l_full, U = U, N = params.Jac_W,
        lower_limit = -50, use_cached_x = true,
        outer_loop_opt = OUTER_OPT, inner_loop_opt = INNER_OPT,
        needs_outer_moment_jacobian = false)
end

function make_div_grad_fn!(obj, m)
    ncon_inner = obj.d - obj.outer_constr_index + 2
    cfg_cache = Ref{Any}(nothing)
    return function (g_free, x_free, θ_full, inner_x)
        obj(inner_x, Float64[], Float64[]; constr = zeros(ncon_inner))
        λ = @view inner_x[2:end]
        ctx = (U = obj.U, γobj = obj.γ, λ = λ, arg1 = obj.arg1, d = obj.d, outer_constr_index = obj.outer_constr_index)
        f = x -> envelope_scalar_div_ctx(reconstruct_full(x, m), ctx)
        if cfg_cache[] === nothing
            cfg_cache[] = ForwardDiff.GradientConfig(f, x_free)
        end
        ForwardDiff.gradient!(g_free, f, x_free, cfg_cache[])
        return g_free
    end
end

function gravity_grad_fn!(g_free, x_free)
    μ = fixed_vals[1]
    gravity_grad_free!(g_free, x_free, D, Aod_free_pos, μ, q_tilde, N_obs)
end

function solve_bound(find_smallest::Bool, δval::Real, θinit)
    obj = make_fullA_obj(δval, find_smallest)
    @assert obj.outer_constr_index == obj.d
    div_grad_fn! = make_div_grad_fn!(obj, fpmap)
    function obj_grad_fn!(g_free, x_free)
        fill!(g_free, 0.0)
        g_free[1] = (-1.0)^find_smallest
    end
    r = CS.outer_loop_cached(obj, fpmap, θ_lo, θ_hi, θinit;
        obj_grad_fn! = obj_grad_fn!, div_grad_fn! = div_grad_fn!,
        gravity_grad_fn! = gravity_grad_fn!, has_gravity = true, gravity_value_scale = -1.0 / N_obs,
        use_cache = true, outer_loop_opt = OUTER_OPT)
    γp = r.θ_min_full[3+D]
    κ = 1 - γp^(σ / (σ - 1))
    return r, γp, κ
end

DELTA_GRID = sort(let s = get(ENV, "DELTA_GRID", "1.0")
    parse.(Float64, split(s, ","))
end)   # ascending: warm-start chain runs small-delta-first, per spec section 11

const OUT_DIR = get(ENV, "OUT_DIR", joinpath(@__DIR__, "batch_out"))
isdir(OUT_DIR) || mkpath(OUT_DIR)

"Per-solve output file: one JLD2 per (method,bound,delta), with a `done` flag written only after
the solve is fully processed (spec section 13's completion-marker requirement -- distinct from a
partial/crashed file, which has no `done=true` key or doesn't exist)."
result_path(bound_name, δval) = joinpath(OUT_DIR, "fullA_$(bound_name)_delta$(δval).jld2")

function load_if_done(path)
    isfile(path) || return nothing
    d = try
        JLD2.load(path)
    catch
        return nothing   # corrupt/partial file from an interrupted write -- treat as not done
    end
    get(d, "done", false) === true ? d : nothing
end

results = Dict{Tuple{Symbol,Float64},Any}()
for (name, fs) in ((:lower, false), (:upper, true))   # kappa DECREASING in gamma'_focal: lower<->maximize(fs=false), upper<->minimize(fs=true) -- matches sequential driver's and run_fullA_D10_methodB.jl's convention; was backwards here, causing a lower>upper ordering bug (found via the section-16 bound-ordering check on the first real batch run)
    BOUND_ARG in ("both", String(name)) || continue
    θcur = copy(θ0_up)   # warm-start chain: each delta starts from the PREVIOUS delta's solution
                          # within this SAME bound direction; never crosses lower<->upper (spec section 11)
    for δval in DELTA_GRID
        path = result_path(name, δval)
        existing = load_if_done(path)
        if existing !== nothing
            @printf("\n=== [full-A PRODUCTION] D=%d delta=%g bound=%s -- ALREADY DONE, skipping (resume) ===\n", D, δval, name)
            θcur = existing["theta_min_full"]
            results[(name, δval)] = (γp = existing["gamma_p"], κ = existing["kappa"], wall = existing["wall"],
                                      nStatus = existing["nStatus"], opt_err = existing["opt_err"], feas_err = existing["feas_err"])
            flush(stdout)
            continue
        end
        @printf("\n=== [full-A PRODUCTION] D=%d delta=%g bound=%s (warm-started from prior delta) ===\n", D, δval, name); flush(stdout)
        t0 = time()
        r, γp, κ = solve_bound(fs, δval, θcur)
        wall = time() - t0
        CS.summarize(r.cache; label = "$name bound, delta=$δval")
        @printf("RESULT: status=%d opt_err=%.4g feas_err=%.4g outer_iters=%d gamma'=%.6f kappa=%.6f wall=%.1fs\n",
                r.nStatus, r.opt_err, r.feas_err, r.outer_iters, γp, κ, wall)
        # save BEFORE marking done, so a crash mid-write never leaves a file that reads as done=true
        JLD2.save(path, Dict(
            "method" => "fullA", "bound" => String(name), "delta" => δval, "D" => D,
            "theta_min_full" => r.θ_min_full, "x_min_free" => r.x_min_free,
            "gamma_p" => γp, "kappa" => κ, "nStatus" => r.nStatus, "opt_err" => r.opt_err,
            "feas_err" => r.feas_err, "outer_iters" => r.outer_iters, "outer_fc" => r.outer_fc,
            "wall" => wall,
            "unique_free_x" => length(Set(rr.x_hash for rr in r.cache.trace)),
            "inner_solves" => r.cache.n_inner_solve, "grad_computations" => r.cache.n_grad_compute,
            "warm_started_inner" => r.cache.n_warm_started, "cold_inner" => r.cache.n_cold,
            "t_inner" => r.cache.t_inner, "t_grad" => r.cache.t_grad,
            "starting_point_source" => δval == DELTA_GRID[1] ? "theta0_up (initial)" : "warm-started from delta=$(DELTA_GRID[findfirst(==(δval), DELTA_GRID)-1])",
            "done" => true))
        θcur = r.θ_min_full
        results[(name, δval)] = (γp = γp, κ = κ, wall = wall, nStatus = r.nStatus, opt_err = r.opt_err, feas_err = r.feas_err)
    end
end
println("FULLA_PRODUCTION_RUN DONE  D=$D")
