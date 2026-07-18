# ============================================================================
# Continuation 8, Wave 2, workstream C ("gamma branches"): shared library for
# the low-g and high-g profile_Delta(g) = min_A Delta(g,A) branch investigations.
#
# NEW FILE (c8_gammabranch_* naming per this workstream's file-ownership
# convention). Does NOT modify gamma_profile.jl / gamma_profile_multistart.jl /
# composite_gradient*.jl / oracle*.jl / compressed_*.jl / winner_certificate.jl
# -- reuses all of them as read-only libraries:
#   - gamma_profile.jl: `profile_delta_at_gamma` structure/KNITRO settings are
#     mirrored here (a small variant, `profile_delta_at_gamma_c8`, is defined
#     because we need an extra `moment_repr` knob gamma_profile.jl does not
#     expose -- everything else, including the composite-gradient callback and
#     KNITRO options, is copied verbatim from that file, not redesigned).
#   - compressed_live.jl: `evaluate_fullA_fast_compressed` / `compressed_base_state`
#     for the compressed moment_repr path.
#   - stationarity_check.jl: NOT `include`d directly (its own top-level
#     `include(context.jl)` etc. would redefine structs like PivotGravityElim
#     already loaded by gamma_profile.jl's includes, breaking `pe`'s type
#     identity -- a real hazard, confirmed by inspection, not assumed). Its
#     ~15-line KKT-residual formula is reproduced verbatim below as
#     `external_stationarity_check_c8`, explicitly attributed here.
#   - phaseF_primal_feasibility_lp.jl: same include-collision hazard (it does
#     `ctx = d4_exact_setup(...)` + `const COMMIT/OUTDIR` at top level). Its
#     LP construction (m_s>=0, mean(m)=1, mean(m.*G_j)=0 all j, HiGHS via
#     JuMP) is reproduced verbatim below as `lp_feasibility_check_c8`,
#     reusing OUR ctx instead of building a second one.
# ============================================================================
ENV["GP_RUN_GRID"] = "0"   # load gamma_profile.jl's functions/ctx/pe without running its own grid
include(joinpath(@__DIR__, "gamma_profile.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
using LinearAlgebra: dot, norm
using JuMP, HiGHS
using Random, Statistics

# ---- reference start points (verbatim from gamma_profile_multistart.jl / candidate_registry.jl) ----
const G_INCUMBENT_C8 = 0.8926359584642946   # upper_lfixcomposite_sr1_60s
const ZFREE_INCUMBENT_C8 = [0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]

# calibration-base A_od in reduced coords (same construction as gamma_profile_multistart.jl)
Aod_theta0_c8 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
const ZFREE_CALIB_C8 = pivot_reduce(log.(Aod_theta0_c8), pe)

# ---- the lower incumbent's own (g,A) point (candidate_registry.jl's w_lower_lfixcomposite_fast,
# label "lower_lfixcomposite_fast_sr1_300s", kappa=0.005428799948779983) -- MANDATORY high-g start
# per this workstream's brief. w[1] is gamma'_focal, w[2:end] is z_free (reduced A-block coords). ----
const W_LOWER_INCUMBENT = [0.9967391744173478, 0.33826763364911505, 0.2756097423805949, 0.3168080759212972, 0.28291467586724917, 1.124214996356822, 1.0586966677239436, 1.0353970705480537, 1.0651065385723435, 0.7972795304932372, 0.7498744164437179, 0.797300832305142, 0.7520482961532734, 1.464240420901755, 1.3955928233005424, 1.4031151818251653]
const G_LOWER_INCUMBENT = W_LOWER_INCUMBENT[1]
const ZFREE_LOWER_INCUMBENT = W_LOWER_INCUMBENT[2:end]

# A_od (16-dim, D^2) from reduced zfree
Aod_vec_c8(zfree) = vec(exp.(pivot_expand(zfree, pe)))

# ============================================================================
# moment_repr-aware single evaluation of Delta_dual(g, A) -- normalizes the
# dense (evaluate_fullA, oracle.jl) vs compressed (evaluate_fullA_fast with
# moment_representation=:compressed, oracle_fast.jl/compressed_live.jl)
# return shapes to the SAME NamedTuple field access pattern gamma_profile.jl's
# cb_F! uses (Delta_dual, inner_status, gravity_value, max_abs_moment_kkt_resid,
# theta_full/zeta/lambda for base-state construction).
# ============================================================================
function eval_F_c8(xf::AbstractVector, ctx; moment_repr::Symbol = :dense, warm::Bool = true)
    if moment_repr === :dense
        return evaluate_fullA(xf, ctx; cache = nothing, warm = warm)
    elseif moment_repr === :compressed
        r, _ = evaluate_fullA_fast(xf, ctx; cache = nothing, warm = warm, moment_representation = :compressed)
        return r
    else
        error("eval_F_c8: moment_repr must be :dense or :compressed, got $moment_repr")
    end
end

"""
    profile_delta_at_gamma_c8(g, zfree_start, ctx, pe; moment_repr, maxtime_real, hessopt_tag, threaded)

Local minimization of Delta_dual(g, A) over the A-block (z_free) only, g held
fixed -- SAME KNITRO settings / callback structure as gamma_profile.jl's
`profile_delta_at_gamma`, with one addition: the F-callback's moment
representation is selectable (`:dense` default, `:compressed` opt-in). The
gradient callback (G) is UNCHANGED from gamma_profile.jl -- composite_gradient_at_fast
has no compressed variant (Wave 1 built compressed only for the F/objective
path, see compressed_live.jl's own header), so the base state used for the
gradient is built via the SAME evaluator as F (dense solve_base_state-style
construction from F's own returned zeta/lambda/theta_full, which both dense
and compressed evaluators populate identically -- verified by inspection of
compressed_live.jl's evaluate_fullA_fast_compressed tail, which calls the
same `obj(inner_x, constr=...)` populating `obj.arg1` the dense path relies on).
"""
function profile_delta_at_gamma_c8(g::Float64, zfree_start::Vector{Float64}, ctx, pe;
        moment_repr::Symbol = :dense,
        maxtime_real::Float64 = 30.0, hessopt_tag::String = "sr1", threaded::Bool = true)
    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_$(hessopt_tag).opt"))
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", maxtime_real)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, n)
    KNITRO.KN_set_var_lobnds_all(kc, fill(-8.0, n))
    KNITRO.KN_set_var_upbnds_all(kc, fill(8.0, n))
    KNITRO.KN_set_var_primal_init_values_all(kc, zfree_start)

    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)
    best = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0)
    trace = NamedTuple[]

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w)
        r = eval_F_c8(xf, ctx; moment_repr = moment_repr, warm = true)
        Δ = isfinite(r.Delta_dual) ? r.Delta_dual : 1e6
        evalResult.obj[1] = Δ
        n_eval[] += 1
        if r.inner_status in (0, -100, -101, -103)
            base = BaseDualState(xf, r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
            last_F_state[] = (w = copy(w), base = base)
            if best[] === nothing || Δ < best[].Delta_dual
                best[] = (zfree = copy(zfree), Delta_dual = Δ, gravity_value = r.gravity_value,
                          max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, inner_status = r.inner_status)
            end
        else
            last_F_state[] = nothing
        end
        push!(trace, (idx = n_eval[], Delta_dual = Δ, inner_status = r.inner_status))
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        zfree = evalRequest.x
        w = vcat(g, zfree)
        xf = x_free_from_w(w)
        shared = last_F_state[]
        base = (shared !== nothing && shared.w == w) ? shared.base : nothing
        local gfull
        try
            gfull, _ = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = threaded, h_mode = :adaptive)
        catch e
            println("  [c8_gammabranch grad fallback] base-state solve failed at g=$g, zfree=$zfree ($e) -- returning zero gradient")
            gfull = zeros(D2)
        end
        evalResult.objGrad .= gfull[2:end]
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, Int32[], cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!)

    t0 = time()
    nStatus = KNITRO.KN_solve(kc)
    wall = time() - t0
    nStatus_code, _, xsol, _ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    b = best[]
    return (g = g, knitro_status = nStatus_code, wall = wall, n_eval = n_eval[],
            zfree_terminal = collect(xsol),
            best_zfree = b === nothing ? nothing : b.zfree,
            best_Delta = b === nothing ? NaN : b.Delta_dual,
            best_gravity = b === nothing ? NaN : b.gravity_value,
            best_kkt = b === nothing ? NaN : b.max_abs_moment_kkt_resid,
            trace = trace)
end

# ============================================================================
# multistart at a fixed g: same 9-start pattern as gamma_profile_multistart.jl's
# `make_starts`, generalized to accept an arbitrary set of "anchor" starts
# (the low-g branch uses {continuation, incumbent, calib}; the high-g branch
# ALSO includes the lower incumbent per the brief's mandatory-start requirement).
# ============================================================================
function make_starts_c8(zf_continuation::Vector{Float64}, rng::AbstractRNG;
        anchors::Vector{Tuple{String,Vector{Float64}}} = [("incumbent", ZFREE_INCUMBENT_C8), ("calib", ZFREE_CALIB_C8)],
        n_perturb_each::Int = 2, perturb_radii::Vector{Float64} = [0.5, 1.5])
    starts = Tuple{String,Vector{Float64}}[]
    push!(starts, ("continuation", copy(zf_continuation)))
    for (label, zf) in anchors
        push!(starts, (label, copy(zf)))
    end
    for (label, zf) in anchors
        for r in perturb_radii
            push!(starts, ("$(label)+$(r)r", zf .+ r .* randn(rng, n)))
        end
    end
    push!(starts, ("uniform_pm2_a", (rand(rng, n) .* 4.0 .- 2.0)))
    push!(starts, ("uniform_pm2_b", (rand(rng, n) .* 4.0 .- 2.0)))
    return starts
end

"""
    multistart_profile_at_g(g, zf_continuation, ctx, pe; anchors, moment_repr, maxtime_per_start, seed)

Runs `make_starts_c8`'s full start set at fixed g, returns (best, second_best,
all per-start results) -- the per-g building block for both branch drivers.
"""
function multistart_profile_at_g(g::Float64, zf_continuation::Vector{Float64}, ctx, pe;
        anchors::Vector{Tuple{String,Vector{Float64}}} = [("incumbent", ZFREE_INCUMBENT_C8), ("calib", ZFREE_CALIB_C8)],
        moment_repr::Symbol = :dense, maxtime_per_start::Float64 = 15.0, seed::Int = 7000 + round(Int, g * 1e6))
    rng = MersenneTwister(seed)
    starts = make_starts_c8(zf_continuation, rng; anchors = anchors)
    results = NamedTuple[]
    for (kind, zf0) in starts
        res = profile_delta_at_gamma_c8(g, zf0, ctx, pe; moment_repr = moment_repr, maxtime_real = maxtime_per_start, hessopt_tag = "sr1")
        has_sol = res.best_zfree !== nothing && isfinite(res.best_Delta)
        push!(results, (kind = kind, Delta = has_sol ? res.best_Delta : NaN, feasible = has_sol,
                         knitro_status = res.knitro_status, n_eval = res.n_eval, wall = res.wall,
                         zfree = has_sol ? copy(res.best_zfree) : nothing,
                         gravity = res.best_gravity, kkt = res.best_kkt))
    end
    feas = filter(r -> r.feasible, results)
    if isempty(feas)
        return (g = g, best = nothing, second_best = nothing, all = results)
    end
    ord = sortperm([r.Delta for r in feas])
    best = feas[ord[1]]
    second = length(ord) >= 2 ? feas[ord[2]] : best
    return (g = g, best = best, second_best = second, all = results)
end

# ============================================================================
# External KKT stationarity check -- verbatim reproduction of
# stationarity_check.jl::external_stationarity_check (see header note above
# for why this is copied rather than `include`d), reusing OUR ctx/pe.
# ============================================================================
function external_stationarity_check_c8(w::AbstractVector, ctx, pe;
        find_smallest::Bool, h::Float64 = 0.01,
        w_lo::Union{Nothing,AbstractVector} = nothing, w_hi::Union{Nothing,AbstractVector} = nothing,
        bound_tol::Float64 = 1e-4, δ::Float64 = 1.0, warm::Bool = false)
    # NOTE: defaults to warm=false (COLD inner-dual start), unlike stationarity_check.jl's own
    # warm=true default -- this function is called AFTER an arbitrary sequence of unrelated
    # profile_delta_at_gamma_c8 sweeps that leave ctx.obj.arg1 (the shared inner-dual warm state) at
    # whatever the LAST unrelated (g,A) point was; reusing that stale warm state here corrupts the
    # central-FD Delta evaluations this diagnostic depends on (caught directly: warm=true gave a
    # visibly wrong Delta at a point independently known to sit almost exactly at Delta=delta).
    nn = length(w)
    function x_free_from_w_local(ww)
        gp = ww[1]; zfree = ww[2:end]
        z = pivot_expand(zfree, pe)
        return vcat(gp, vec(exp.(z)))
    end
    function Delta_of_w(ww)
        r = evaluate_fullA(x_free_from_w_local(ww), ctx; cache = nothing, warm = warm)
        return r.Delta_dual
    end
    Δ0 = Delta_of_w(w)
    grad_f = zeros(nn); grad_f[1] = find_smallest ? 1.0 : -1.0
    grad_Delta = zeros(nn)
    n_nonfinite = 0
    for i in 1:nn
        wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
        Δp = Delta_of_w(wp); Δm = Delta_of_w(wm)
        if isfinite(Δp) && isfinite(Δm)
            grad_Delta[i] = (Δp - Δm) / (2h)
        else
            n_nonfinite += 1
            grad_Delta[i] = 0.0
        end
    end
    active_lo = w_lo === nothing ? falses(nn) : (w .- w_lo .< bound_tol)
    active_hi = w_hi === nothing ? falses(nn) : (w_hi .- w .< bound_tol)
    n_active_bounds = count(active_lo) + count(active_hi)
    eta = -dot(grad_f, grad_Delta) / max(dot(grad_Delta, grad_Delta), 1e-300)
    residual_vec = grad_f .+ eta .* grad_Delta
    residual_norm = norm(residual_vec)
    comp_slack = eta * (Δ0 - δ)
    return (Delta = Δ0, Delta_minus_delta = Δ0 - δ, eta = eta, eta_nonneg = eta >= -1e-8,
            residual_norm = residual_norm, residual_relative = residual_norm / max(norm(grad_f), 1e-12),
            complementary_slackness = comp_slack, n_active_bounds = n_active_bounds,
            n_nonfinite_probes = n_nonfinite, grad_Delta = grad_Delta)
end

# ============================================================================
# Direct LP primal-feasibility certificate -- verbatim reproduction of
# phaseF_primal_feasibility_lp.jl::lp_feasibility_check (see header note
# above), reusing OUR ctx instead of building a second one.
# ============================================================================
function lp_feasibility_check_c8(xf::Vector{Float64}, label::String)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    W = size(ctx.obj.U, 1); d = ctx.obj.d
    K = zeros(W); G = zeros(W, d)
    ctx.obj.moments!(K, G, θ_full, ctx.obj.U, ctx.obj)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, m[1:W] >= 0)
    @constraint(model, mean_m, sum(m) / W == 1)
    @constraint(model, moments[j=1:d], sum(m[s] * G[s, j] for s in 1:W) / W == 0)
    @objective(model, Min, 0)
    optimize!(model)
    status = termination_status(model)
    feasible_exact = status == MOI.OPTIMAL

    phase1_max_resid = NaN
    if !feasible_exact
        model2 = Model(HiGHS.Optimizer)
        set_silent(model2)
        @variable(model2, m2[1:W] >= 0)
        @variable(model2, t >= 0)
        @constraint(model2, sum(m2) / W == 1)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W <= t)
        @constraint(model2, [j=1:d], sum(m2[s] * G[s, j] for s in 1:W) / W >= -t)
        @objective(model2, Min, t)
        optimize!(model2)
        phase1_max_resid = termination_status(model2) == MOI.OPTIMAL ? value(t) : NaN
    end
    classification = if feasible_exact
        "FEASIBLE_LP"
    elseif isfinite(phase1_max_resid) && phase1_max_resid > 1e-6
        "CERTIFIED_INFEASIBLE"
    else
        "LP_INCONCLUSIVE"
    end
    return (label = label, feasible_exact = feasible_exact, phase1_max_resid = phase1_max_resid, classification = classification)
end

println("c8_gammabranch_core.jl loaded: ctx D=$(ctx.D) W=$(size(ctx.U,1)) δ=$(ctx.δ) σ=$(ctx.σ)  bounds=[$(ctx.bounds.γp_lo), $(ctx.bounds.γp_hi)]")
println("G_INCUMBENT_C8=$G_INCUMBENT_C8  G_LOWER_INCUMBENT=$G_LOWER_INCUMBENT (kappa_lower=0.005428799948779983)")
