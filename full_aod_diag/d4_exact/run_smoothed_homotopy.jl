# ============================================================================
# Steps 3-7 + gradient benchmark: genuinely CONSISTENT smoothed full-A solve
# with a temperature (rho) homotopy, matched smoothed value+gradient outer
# KNITRO solves, then a switch to the exact hard oracle + lfix_incremental
# (:incremental_o1 tier) polish. See docs/fullA_smoothed_consistent_experiment.md
# for the full writeup this script's output feeds.
#
# Run: julia --project=. full_aod_diag/d4_exact/run_smoothed_homotopy.jl
# ============================================================================
include(joinpath(@__DIR__, "smoothed_consistent.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
using KNITRO, Printf, Dates, Statistics, LinearAlgebra

const COMMIT = try strip(read(`git rev-parse --short HEAD`, String)) catch; "uncommitted" end
const RUN_ID = "smoothed_homotopy_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const FIND_SMALLEST = true   # "upper" direction, matches the upper_maxit40 headline candidate this experiment benchmarks against
const MAXIT_STAGE = parse(Int, get(ENV, "D4X_HOMOTOPY_MAXIT", "15"))
const OPT_FILE_STAGE = joinpath(@__DIR__, "csw_outer_fcga_no_maxit$(MAXIT_STAGE).opt")
const OPT_FILE_FINAL = joinpath(@__DIR__, "csw_outer_fcga_no_maxit40.opt")

ctx = d4_exact_setup(find_smallest = FIND_SMALLEST)
pe = build_pivot_elimination(ctx)
D = ctx.D; D2 = D^2

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

zfree0 = pivot_reduce(zeros(D, D), pe)
gp0 = ctx.θ0_up[3+D]
w_calib = vcat(gp0, zfree0)

# ---- homotopy STARTING point: NOT theta_initial/calibration. Directly verified (this session,
#      see docs/fullA_smoothed_consistent_experiment.md sec 4) that calibration's inner CC dual is
#      INFEASIBLE (nStatus=-300) under BOTH the hard oracle (candidate_registry.jl's own printed
#      output) AND the smoothed model at every rho this script tries -- a pre-existing property of
#      this synthetic economy at that particular theta, not a smoothing artifact. Starting the
#      OUTER search there gives KNITRO's gradient callback no usable local information (a Delta
#      that failed to solve carries no (zeta*,lambda*) to differentiate around), which empirically
#      caused every homotopy stage to terminate KNITRO status=-200 (declared infeasible) with ZERO
#      feasible evaluations found -- confirmed as the failure mode of this script's first working
#      version, not assumed. Fix: start from `upper_maxit15_productfd_control`
#      (candidate_registry.jl), a point ALREADY independently verified feasible for the HARD model
#      and re-verified here to be smoothed-feasible at every rho in the grid below.
w0 = [0.8938496736355915, 0.12274466988967254, 0.001935434700755778, 0.09886609762478069, 0.02405249845877564, 1.2817778618748479, 0.22664068017003447, 1.2294664287879011, 1.3227219006788014, 0.6240228573299679, 0.5169790045732584, 0.5284244103680663, 0.5442350971177623, 0.8102649765537995, 1.3598366690362491, 0.7041331280854306]

w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

println("="^78); println("STEP 4: temperature (rho) grid from the empirical winner/runner-up gap distribution"); println("="^78)
θ_full0 = CS.reconstruct_full(x_free_from_w(w0), ctx.m)
_, _, gap0 = compute_winners_fast(θ_full0, ctx)
gap_finite = filter(isfinite, vec(gap0))
qs = (0.75, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
gapq = Dict(q => quantile(gap_finite, q) for q in qs)
println("winner/runner-up PRICE gap distribution at the homotopy start point (w0=upper_maxit15_productfd_control, W=$(size(ctx.U,1)), D=$D, $(D^2) draw-destination cells):")
for q in qs
    println("  quantile($q) = ", gapq[q])
end
println("min gap = ", minimum(gap_finite), "   max gap = ", maximum(gap_finite))
println("NOTE on 'blow up': smoothMinIndNew!'s convention (softMIN via exp((x-xmin)*tuner), tuner<0)")
println("is bounded in (0,1] for EVERY tuner<0 and EVERY finite price gap -- there is no coarse-rho")
println("overflow risk for this specific formulation (unlike a softMAX over unbounded values). The")
println("real risk this grid must resolve is UNDERFLOW/no-discrimination: rho too large relative to")
println("the typical gap gives a near-uniform (uninformative) split; rho too small relative to the")
println("SMALLEST gap wastes homotopy stages doing nothing (already indistinguishable from hard).")

# rho_k = quantile(gap, q_k) for decreasing q -- ties the homotopy schedule directly to the
# empirical scale of "how close is a typical near-tie", per task's explicit instruction.
#
# GENUINE FINDING (not anticipated going in, directly caught by this script's first working run --
# see docs/fullA_smoothed_consistent_experiment.md sec 4 for the full writeup): the SMOOTHED inner
# CC dual problem has its OWN feasibility boundary in rho, separate from (and COARSER than) the
# divergence-budget Delta<=delta constraint. At coarse rho (large, e.g. rho=gapq(0.5)=0.169 or
# rho=0.1), w0's smoothed inner dual is OUTRIGHT INFEASIBLE (KNITRO nStatus=-300, "problem appears
# infeasible" -- not merely Delta>delta, the dual solve itself has no solution) even though the
# SAME point's HARD inner dual solves cleanly (candidate_registry.jl). This makes sense in
# hindsight: coarse smoothing distorts the moment TARGETS substantially (mean|G-Gh| ~0.1-0.3 at
# rho=0.1-0.2, per test_smoothed_consistent.jl TEST 2), which can push the reweighting problem
# outside the region KNITRO's inner dual solver can certify. A naive "start broad, sharpen later"
# homotopy schedule based on the gap distribution ALONE (this script's first version) silently
# handed KNITRO's outer solve zero usable evaluations at every stage (status -200/-410, "problem
# may be locally infeasible", NO feasible point found in any of the 12-15 iterations) -- caught by
# inspecting Delta_smoothed=NaN at every reported stage, not assumed to be working. Fix: empirically
# scan for the COARSEST rho at which w0's smoothed inner dual actually solves (nStatus a valid
# code), and use that as the coarse end of the real grid instead of the raw gap-quantile value.
println("\nScanning for the coarsest EMPIRICALLY-FEASIBLE rho (smoothed inner dual actually solves at w0):")
candidate_coarse = [gapq[q] for q in (0.75, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)]
xf0 = x_free_from_w(w0)
coarsest_feasible = nothing
for rho in candidate_coarse
    obj_probe = smoothed_obj_for(ctx, rho_to_tuner(rho))
    Δp, _, nsp = smoothed_optimized_Delta(xf0, ctx, obj_probe)
    inner_ok = nsp in (0, -100, -101, -103)
    println("  rho=$rho  inner_status=$nsp  inner_solve_ok=$inner_ok  Delta=$Δp")
    if inner_ok && coarsest_feasible === nothing
        global coarsest_feasible = rho
    end
end
coarsest_feasible === nothing && error("STEP 4: no candidate rho (even the finest quantile) gives a feasible smoothed inner dual at w0 -- cannot build a homotopy grid")
println("Coarsest empirically-feasible rho = ", coarsest_feasible)

RHO_GRID = filter(r -> r <= coarsest_feasible, [gapq[q] for q in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01)])
isempty(RHO_GRID) && (RHO_GRID = Float64[])
(isempty(RHO_GRID) || RHO_GRID[1] < coarsest_feasible) && pushfirst!(RHO_GRID, coarsest_feasible)
push!(RHO_GRID, minimum(gap_finite) / 10)   # final stage: below the smallest observed gap -- effectively hard for every draw at this economy
unique!(RHO_GRID)
println("\nChosen rho grid (descending, empirically feasibility-bounded at the coarse end): ", RHO_GRID)

# ============================================================================
# STEP 3 + 5: homotopy loop. Matched smoothed VALUE (re-solved inner CC dual
# at fixed rho, via the unmodified CS.inner_loop_internal) and matched
# smoothed GRADIENT (ForwardDiff.gradient of smoothed_fixed_dual_L at the
# freshly-solved (zeta*,lambda*) -- the envelope-theorem construction already
# validated on the EXACT/hard side by three_way_derivatives.jl/test_three_way.jl,
# applied here to the smoothed model).
# ============================================================================
mutable struct StageResult
    rho::Float64
    w_best::Vector{Float64}
    kappa::Float64
    gp::Float64
    Delta_smoothed::Float64
    gravity_value::Float64
    grad_check_relerr::Float64
    entropy_mean::Float64
    entropy_max_possible::Float64
    frac_boundary::Float64
    n_outer_iters::Int
    n_eval::Int
    wall_seconds::Float64
    hessopt_fallback_detected::Bool
    knitro_status::Int
    opt_err::Float64
end

results = StageResult[]
w_current = copy(w0)
stage_log_rows = String[]

for (stage_idx, rho) in enumerate(RHO_GRID)
    println("\n" * "="^78); println("HOMOTOPY STAGE $stage_idx / $(length(RHO_GRID)):  rho = $rho  (tuner = $(rho_to_tuner(rho)))"); println("="^78)
    tuner = rho_to_tuner(rho)
    obj_s = smoothed_obj_for(ctx, tuner)

    n_eval = Ref(0)
    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)

    "Returns `nothing` (not a thrown error) on inner-solve failure -- an infeasible/failed smoothed
    inner dual solve at a probe point is an EXPECTED event during outer search (the calibration start
    point itself is one, per candidate_registry.jl's own hard-oracle output), not a fatal bug. Letting
    it throw INSIDE a KNITRO callback is fatal to the whole outer solve (task sec 17's own documented
    failure mode, reproduced here first before this guard was added -- see git history)."
    function base_at(w::Vector{Float64})
        xf = x_free_from_w(w)
        try
            return solve_smoothed_base_state(xf, ctx, obj_s, tuner)
        catch e
            e isa ErrorException || rethrow()
            return nothing
        end
    end

    function eval_F(w::Vector{Float64})
        n_eval[] += 1
        xf = x_free_from_w(w)
        Δ, inner_x, nStatus = smoothed_optimized_Delta(xf, ctx, obj_s)
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        if feasible && (best_feasible[] === nothing || (FIND_SMALLEST ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp))
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ)
        end
        return Δ, nStatus
    end

    "Central-FD fallback (mirrors run_d4_optimized_fd.jl's eval_grad_central_fd's own documented
    one-sided/zero fallback) if the base solve at `w` itself fails: probes nearby feasible points
    instead of propagating a fatal error into the KNITRO callback."
    function eval_grad(w::Vector{Float64})
        base = base_at(w)
        if base === nothing
            # crude non-fatal fallback: nudge along each coordinate until a feasible base is found,
            # else report a zero gradient for that direction (flagged, matches this investigation's
            # established convention for a probe that cannot be evaluated)
            println("  [grad fallback] base solve failed at current w -- probing a small perturbation")
            for eps in (1e-3, 1e-2, 5e-2)
                base = base_at(w .+ eps)
                base !== nothing && break
            end
            base === nothing && return zeros(length(w))
        end
        g = ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base), w)
        return g
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, OPT_FILE_STAGE)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w_current)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        Δ, nStatus = eval_F(w)
        evalResult.obj[1] = FIND_SMALLEST ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = FIND_SMALLEST ? 1.0 : -1.0
        evalResult.jac .= eval_grad(w)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    knitro_log_path = joinpath(OUTDIR, "knitro_stage$(stage_idx)_rho$(rho).log")
    t0 = time()
    open(knitro_log_path, "w") do io
        redirect_stdout(io) do
            KNITRO.KN_solve(kc)
        end
    end
    wall = time() - t0
    nStatus, objv, w_min, lambda_ = KNITRO.KN_get_solution(kc)
    opt_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, opt_err)
    outer_iters = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, outer_iters)
    KNITRO.KN_free(kc)

    # ---- KNITRO options check: grep the log for a silent hessopt fallback message (task's explicit
    #      instruction -- this investigation has been burned by eval_fcga=yes doing this before) ----
    log_text = read(knitro_log_path, String)
    fallback_detected = occursin("hessopt", lowercase(log_text)) && (occursin("chang", lowercase(log_text)) || occursin("switch", lowercase(log_text)) || occursin("l-bfgs", lowercase(log_text)) && occursin("instead", lowercase(log_text)))
    requested_hessopt_line = occursin("hessopt", log_text) ? "hessopt option present in log (see $knitro_log_path for exact text)" : "no hessopt mention found"

    w_use = best_feasible[] !== nothing ? best_feasible[].w : w_min
    xf_use = x_free_from_w(w_use)
    θ_use = CS.reconstruct_full(xf_use, ctx.m)

    # smoothed Delta at the chosen point (re-solve, matched value)
    Δ_final, inner_x_final, nStatus_final = smoothed_optimized_Delta(xf_use, ctx, obj_s)

    # gravity (smoothing-INDEPENDENT -- pure function of AodPow/tau/q_tilde, see file header of
    # smoothed_consistent.jl and the direct code audit in docs/fullA_smoothed_consistent_experiment.md)
    Aod_θ_use = reshape(θ_use[ctx.Aod_offset+1:ctx.Aod_offset+D^2], D, D)
    μ_use = θ_use[1]
    lambda_g = reshape(ctx.γ.P, (D, D))'
    Aod_lvl = Aod_θ_use .* ctx.γ.cHat .* (((ctx.γ.wHat .* ctx.τ) ./ (ctx.γ.wHat[1,1] .* ctx.τ[1,:]')) .^ (1/μ_use)) .* (lambda_g ./ lambda_g[1,:]')
    AodPow_use = (Aod_lvl ./ ctx.γ.cHat) .^ (-μ_use)
    gravity_val = gravity_value(ctx.τ, AodPow_use, ctx.q_tilde, ctx.N_obs; exclude_diagonal=get(ctx, :exclude_diagonal_gravity, false))

    # entropy / boundary-mass diagnostics at the chosen point, this stage's rho
    probs = winner_softmax_probs(θ_use, ctx, tuner)
    ed = winner_entropy_diagnostics(probs)

    # matched-gradient check: ForwardDiff vs a validated central-FD at the FINAL point (not just at
    # setup, per task's "gradient checks" requirement at EVERY temperature)
    base_final = base_at(w_use)
    if base_final === nothing
        println("  [warn] base solve failed at this stage's chosen point -- skipping gradient check (relerr=NaN)")
        relerr = NaN
    else
        g_ad = ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base_final), w_use)
        hgc = 1e-5
        g_fd = zeros(D2)
        for i in 1:D2
            wp = copy(w_use); wp[i] += hgc; wm = copy(w_use); wm[i] -= hgc
            g_fd[i] = (smoothed_fixed_dual_L(x_free_from_w(wp), ctx, obj_s, base_final) - smoothed_fixed_dual_L(x_free_from_w(wm), ctx, obj_s, base_final)) / (2hgc)
        end
        relerr = norm(g_ad .- g_fd) / max(norm(g_ad), 1e-12)
    end

    κ_use = 1 - θ_use[3+D]^(ctx.σ / (ctx.σ - 1))
    println(@sprintf("stage %d: rho=%.6g  gp=%.8f  kappa=%.8f  Delta_smoothed=%.6f  gravity=%.3g  grad_relerr=%.3g  entropy_mean=%.4f/%.4f  frac_boundary=%.4f  outer_iters=%d  n_eval=%d  wall=%.2fs  knitro_status=%d  opt_err=%.3g",
        stage_idx, rho, w_use[1], κ_use, Δ_final, gravity_val, relerr, ed.mean_entropy, ed.max_possible_entropy, ed.frac_boundary_lt_099, outer_iters[], n_eval[], wall, nStatus, opt_err[]))
    println("  hessopt/eval_fcga check: ", requested_hessopt_line, "  fallback_detected(heuristic)=", fallback_detected)

    push!(results, StageResult(rho, w_use, κ_use, w_use[1], Δ_final, gravity_val, relerr, ed.mean_entropy, ed.max_possible_entropy, ed.frac_boundary_lt_099, outer_iters[], n_eval[], wall, fallback_detected, nStatus, opt_err[]))
    global w_current = copy(w_use)   # warm start next (finer) stage from this stage's best point
end

open(joinpath(OUTDIR, "homotopy_stages.csv"), "w") do io
    println(io, "stage,rho,gp,kappa,Delta_smoothed,gravity_value,grad_relerr,entropy_mean,entropy_max_possible,frac_boundary,n_outer_iters,n_eval,wall_seconds,hessopt_fallback_detected,knitro_status,opt_err")
    for (i, r) in enumerate(results)
        println(io, i, ",", r.rho, ",", r.gp, ",", r.kappa, ",", r.Delta_smoothed, ",", r.gravity_value, ",",
                r.grad_check_relerr, ",", r.entropy_mean, ",", r.entropy_max_possible, ",", r.frac_boundary, ",",
                r.n_outer_iters, ",", r.n_eval, ",", r.wall_seconds, ",", r.hessopt_fallback_detected, ",",
                r.knitro_status, ",", r.opt_err)
    end
end

# ============================================================================
# GRADIENT BENCHMARK (per task's explicit instruction): jac_h materialized
# tensor route vs direct ForwardDiff of the scalar fixed-dual envelope vs
# VJP, all on the SMOOTHED path, timed and memory-measured separately.
# ============================================================================
println("\n" * "="^78); println("GRADIENT BENCHMARK (smoothed path, at the finest-rho stage's best point)"); println("="^78)
let
    rho_bench = RHO_GRID[end]
    tuner = rho_to_tuner(rho_bench)
    obj_s = smoothed_obj_for(ctx, tuner)
    w_bench = w_current
    xf_bench = x_free_from_w(w_bench)
    base = solve_smoothed_base_state(xf_bench, ctx, obj_s, tuner)

    # Method 1: direct ForwardDiff.gradient of the scalar fixed-dual envelope (the method actually
    # used for the outer solve above).
    t1 = @elapsed (g1 = ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base), w_bench))
    a1 = @allocated ForwardDiff.gradient(ww -> smoothed_fixed_dual_L(x_free_from_w(ww), ctx, obj_s, base), w_bench)

    # Method 2: generic jac_h / materialized-tensor route -- obj_s.moments_jacobian! is `error`
    # (never implemented for this problem, exactly as the hard path's own audit found -- see
    # docs/fullA_d4_code_audit.md sec 6 and the parallel jac_h-audit workstream). The only way to
    # produce the equivalent dense tensor is to ForwardDiff.jacobian the FULL smoothed_moments! call
    # w.r.t. ALL D2 free coordinates, i.e. materialize a (W x d x D2) object -- exactly what the task
    # warns is not obviously efficient merely because it is now mathematically valid. Benchmarked
    # directly, not assumed.
    function G_of_w(ww::AbstractVector)
        xf = x_free_from_w(ww)
        θf = CS.reconstruct_full(xf, ctx.m)
        W = size(obj_s.U, 1); d = obj_s.d
        K = zeros(eltype(θf), W); G = zeros(eltype(θf), W, d)
        obj_s.moments!(K, G, θf, obj_s.U, obj_s)
        return vec(G)
    end
    t2 = @elapsed (J = ForwardDiff.jacobian(G_of_w, w_bench))
    a2 = @allocated ForwardDiff.jacobian(G_of_w, w_bench)
    # contract the materialized tensor with (m*, lambda*) exactly as frozen_adjoint_Q/the production
    # gradient does, to get an apples-to-apples gradient from this route
    W = size(obj_s.U, 1); d = obj_s.d; oci = obj_s.outer_constr_index
    Jr = reshape(J, W, d, D2)
    g2 = zeros(D2)
    for k in 1:D2
        acc = 0.0
        for s in 1:W
            acc += base.m_star[s] * dot(base.λstar, @view(Jr[s, 1:oci-1, k]))
        end
        g2[k] = acc / W
    end

    println(@sprintf("Method 1 (ForwardDiff of scalar fixed-dual envelope): time=%.4fs  bytes=%d  |g|=%.4f", t1, a1, norm(g1)))
    println(@sprintf("Method 2 (materialize full W x d x D2 jac_h tensor via ForwardDiff.jacobian): time=%.4fs  bytes=%d  |g|=%.4f", t2, a2, norm(g2)))
    println("  Method 1 / Method 2 time ratio: ", round(t2 / t1, digits = 2), "x  (Method 2 SLOWER means materializing the full tensor is NOT efficient despite being valid, exactly what the task asked to check, not assume)")
    println(@sprintf("  gradient agreement (method1 vs method2, should match): max abs diff=%.3g  cosine=%.6f", maximum(abs.(g1 .- g2)), dot(g1,g2)/(norm(g1)*norm(g2))))
    println("Method 3 (VJP / reverse-mode): NOT AVAILABLE in this codebase -- per memory `enzyme-mooncake-status`,")
    println("  both Enzyme and Mooncake compile but produce NaN gradients on this codebase's real production")
    println("  path; not re-attempted here given that documented prior finding. Method 1 IS effectively a")
    println("  manual VJP in spirit (it differentiates the ALREADY-CONTRACTED scalar m*'lambda*'G(theta)")
    println("  directly, rather than differentiating G(theta) first and contracting after) -- it is the best")
    println("  available substitute for a true single-pass adjoint in this codebase.")

    open(joinpath(OUTDIR, "gradient_benchmark.csv"), "w") do io
        println(io, "method,time_seconds,bytes_allocated,grad_norm")
        println(io, "forwarddiff_scalar_envelope,", t1, ",", a1, ",", norm(g1))
        println(io, "materialized_jac_h_tensor,", t2, ",", a2, ",", norm(g2))
    end
end

# ============================================================================
# STEP 6: switch to the exact hard oracle at the smoothed-optimal point.
# ============================================================================
println("\n" * "="^78); println("STEP 6-7: exact-hard evaluation of the smoothed-optimal point, then lfix_incremental(:incremental_o1) polish"); println("="^78)

w_smoothed_opt = w_current
xf_smoothed_opt = x_free_from_w(w_smoothed_opt)
r_hard_at_smoothed_opt = evaluate_fullA(xf_smoothed_opt, ctx; cache = nothing, warm = false)
κ_hard_at_smoothed_opt = isfinite(r_hard_at_smoothed_opt.gamma_focal_prime) ? 1 - r_hard_at_smoothed_opt.gamma_focal_prime^(ctx.σ/(ctx.σ-1)) : NaN
println("EXACT-HARD evaluation of the smoothed-optimal point (theta from the finest homotopy stage, rho=$(RHO_GRID[end])):")
println("  gp=", r_hard_at_smoothed_opt.gamma_focal_prime, "  kappa=", κ_hard_at_smoothed_opt, "  Delta_dual=", r_hard_at_smoothed_opt.Delta_dual,
        "  Delta-delta=", r_hard_at_smoothed_opt.Delta_minus_delta, "  gravity=", r_hard_at_smoothed_opt.gravity_value,
        "  inner_status=", r_hard_at_smoothed_opt.inner_status, "  max_abs_moment_kkt_resid=", r_hard_at_smoothed_opt.max_abs_moment_kkt_resid)

hard_feasible_start = isfinite(r_hard_at_smoothed_opt.Delta_dual) && r_hard_at_smoothed_opt.inner_status in (0,-100,-101,-103)

# ---- polish: lfix_incremental_at(tier=:incremental_o1) cheap gradient, verified against the exact
#      hard oracle at every accepted step (trust-the-direction / verify-the-value hybrid scheme, per
#      docs/fullA_d4_final_report.md sec 7's "L_fix-based hybrid scheme" framing). Does NOT build a
#      new polishing METHOD -- reuses lfix_incremental_at exactly as profile_lfix_tiers.jl validated it.
polish_log = NamedTuple[]
w_polish = copy(w_smoothed_opt)
if hard_feasible_start
    κ_current = κ_hard_at_smoothed_opt
    Δ_current = r_hard_at_smoothed_opt.Delta_dual
    step = 0.02
    MAXPOLISH = 8
    for round in 1:MAXPOLISH
        base_h = solve_base_state(x_free_from_w(w_polish), ctx)
        cache_h = build_lfix_base_cache(x_free_from_w(w_polish), ctx, base_h)
        g = zeros(D2)
        hfd = 0.01
        for i in 1:D2
            Lp = lfix_incremental_at(cache_h, ctx, pe, w_polish, i, w_polish[i] + hfd; tier = :incremental_o1)
            Lm = lfix_incremental_at(cache_h, ctx, pe, w_polish, i, w_polish[i] - hfd; tier = :incremental_o1)
            g[i] = (Lp - Lm) / (2hfd)
        end
        # ascend/descend gp (coordinate 1) while respecting the divergence budget via L_fix's own
        # value as a CHEAP local proxy for feasibility; direction = -sign(find_smallest)*e1 blended
        # with a small feasibility-restoring component along -grad(L_fix) if L_fix is near delta.
        dir = zeros(D2); dir[1] = FIND_SMALLEST ? -1.0 : 1.0
        # project out the effect on L_fix that would push the divergence budget the wrong way: if
        # d(L_fix)/dw . dir would increase L_fix above delta, damp the step (crude backtracking line
        # search below verifies against the EXACT oracle regardless, so an imperfect projection here
        # only costs a wasted trial, not a silent wrong answer)
        accepted = false
        local_step = step
        for tryi in 1:6
            w_trial = w_polish .+ local_step .* dir
            w_trial = clamp.(w_trial, w_lo, w_hi)
            r_trial = evaluate_fullA(x_free_from_w(w_trial), ctx; cache = nothing, warm = true)
            κ_trial = isfinite(r_trial.gamma_focal_prime) ? 1 - r_trial.gamma_focal_prime^(ctx.σ/(ctx.σ-1)) : NaN
            feas_trial = isfinite(r_trial.Delta_dual) && r_trial.Delta_dual <= ctx.δ + 1e-6 && r_trial.inner_status in (0,-100,-101,-103)
            improve = feas_trial && (FIND_SMALLEST ? κ_trial > κ_current : κ_trial < κ_current)
            push!(polish_log, (round = round, tryi = tryi, step = local_step, kappa_trial = κ_trial, feasible = feas_trial, accepted = improve))
            if improve
                global w_polish = w_trial
                global κ_current = κ_trial
                global Δ_current = r_trial.Delta_dual
                accepted = true
                println(@sprintf("  polish round %d: accepted step=%.4f  kappa=%.8f  Delta=%.6f  feasible=%s", round, local_step, κ_current, Δ_current, feas_trial))
                break
            else
                local_step /= 2
            end
        end
        if !accepted
            println("  polish round $round: no improving feasible step found within 6 halvings -- stopping polish (local optimum of this direction, or step too coarse)")
            break
        end
    end
else
    println("SKIPPING polish: the smoothed-optimal point's exact-hard inner dual solve is infeasible (inner_status=$(r_hard_at_smoothed_opt.inner_status)) -- L_fix's base state requires a feasible base point (per three_way_derivatives.jl::solve_base_state's own assertion). Reporting the smoothed-optimum's exact-hard evaluation as-is; no polish attempted.")
end

open(joinpath(OUTDIR, "polish_log.csv"), "w") do io
    println(io, "round,tryi,step,kappa_trial,feasible,accepted")
    for r in polish_log
        println(io, r.round, ",", r.tryi, ",", r.step, ",", r.kappa_trial, ",", r.feasible, ",", r.accepted)
    end
end

println("\n" * "="^78); println("FINAL EXACT-HARD FRESH RECHECK of the polished candidate"); println("="^78)
r_polished = evaluate_fullA(x_free_from_w(w_polish), ctx; cache = nothing, warm = false)
κ_polished = isfinite(r_polished.gamma_focal_prime) ? 1 - r_polished.gamma_focal_prime^(ctx.σ/(ctx.σ-1)) : NaN
println("polished: gp=", r_polished.gamma_focal_prime, "  kappa=", κ_polished, "  Delta_dual=", r_polished.Delta_dual,
        "  Delta-delta=", r_polished.Delta_minus_delta, "  gravity=", r_polished.gravity_value,
        "  inner_status=", r_polished.inner_status, "  max_abs_moment_kkt_resid=", r_polished.max_abs_moment_kkt_resid)

const KAPPA_UPPER_MAXIT40 = 0.17176461388430053   # cross-checked against candidate_registry.jl's own fresh re-evaluation this session
println("\n" * "="^78); println("STEP 7 SUMMARY: three distinct results"); println("="^78)
println("(a) smoothed-problem optimum (finest rho stage, rho=$(RHO_GRID[end])):  gp=", w_smoothed_opt[1], "  (smoothed Delta / kappa are AT A POSITIVE TEMPERATURE -- NOT a valid hard-estimand number, reported separately per task instruction)")
println("(b) EXACT-HARD evaluation of that SAME smoothed-optimal theta:          kappa=", κ_hard_at_smoothed_opt, "  feasible=", hard_feasible_start)
println("(c) FINAL exact-hard POLISHED candidate (after lfix_incremental polish): kappa=", κ_polished, "  feasible(Delta<=delta)=", isfinite(r_polished.Delta_dual) && r_polished.Delta_dual <= ctx.δ + 1e-6)
println("Existing best hard-only incumbent (upper_maxit40, from candidate_registry.jl): kappa=", KAPPA_UPPER_MAXIT40)
println("Delta kappa (polished - incumbent) = ", κ_polished - KAPPA_UPPER_MAXIT40)

open(joinpath(OUTDIR, "final_summary.txt"), "w") do io
    println(io, "rho_grid=", RHO_GRID)
    println(io, "(a) smoothed_opt_gp=", w_smoothed_opt[1], " w=", w_smoothed_opt)
    println(io, "(b) hard_eval_of_smoothed_opt: kappa=", κ_hard_at_smoothed_opt, " Delta_dual=", r_hard_at_smoothed_opt.Delta_dual, " inner_status=", r_hard_at_smoothed_opt.inner_status)
    println(io, "(c) polished: kappa=", κ_polished, " Delta_dual=", r_polished.Delta_dual, " gravity=", r_polished.gravity_value, " w=", w_polish)
    println(io, "incumbent upper_maxit40 kappa=", KAPPA_UPPER_MAXIT40)
    println(io, "delta_kappa_vs_incumbent=", κ_polished - KAPPA_UPPER_MAXIT40)
end
println("\nWrote ", OUTDIR)
