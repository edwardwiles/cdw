# ============================================================================
# Continuation 8, Section 9 (nested-W). Main grid driver.
#
# For each W in {8000, 20000, 80000} (nested prefixes of ONE common W=80000
# draw pool, see c8_nestedw_context.jl) and each candidate in
# {upper_lfixcomposite_sr1_60s, lower_v2} (candidate_registry.jl's current
# headline incumbents):
#   (a) RE-EVALUATE the candidate's EXISTING (g,A) at this W's draws (no
#       optimizer involved) -- "does a fixed point's value/feasibility move
#       with more/fewer draws";
#   (b) RE-OPTIMIZE (warm-started from the candidate's own w, same driver
#       class that produced it: run_d4_optimized_fd.jl's
#       lfix_composite/lfix_composite_fast + hessopt=sr1 dispatch, reproduced
#       here rather than editing that file per this workstream's
#       file-ownership convention) -- "does the candidate improve once the
#       optimizer actually sees this W's draws".
# These are two DIFFERENT checks (explicitly not collapsed here).
#
# Run: source .knitro_env.sh (plain source) then
#   JULIA_NUM_THREADS=20 julia --project=. full_aod_diag/d4_exact/c8_nestedw_run_grid.jl
# ============================================================================
include(joinpath(@__DIR__, "c8_nestedw_context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "composite_gradient_fast.jl"))   # -> composite_gradient_at, composite_gradient_at_fast, BaseDualState (via three_way_derivatives.jl)
using KNITRO, JLD2, Printf, Dates

const COMMIT = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const RUN_ID = "c8_nestedw_grid_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, RUN_ID)
mkpath(OUTDIR)
const CSV_PATH = joinpath(OUTDIR, "c8_nestedw_results.csv")
const A_SOLUTIONS_PATH = joinpath(OUTDIR, "c8_nestedw_a_solutions.jld2")
println(">>> c8_nestedw_run_grid.jl  commit=$COMMIT  outdir=$OUTDIR"); flush(stdout)

const W_GRID = [8000, 20000, 80000]

const w_upper = [0.8926359584642946, 0.16935885803984474, 0.037584283338539824, 0.12216855636925181, 0.1378438125793156, 1.4358658074709083, 0.21920931970835777, 1.2241096016138515, 1.3275683327810723, 0.7077536256287202, 0.5456384492177044, 0.5135010576503252, 0.6960276724200908, 0.9333803510577844, 1.4851233069076464, 0.3113004424480845]
const w_lower = [0.9973649883022927, 0.4191333995096165, 0.3278704261228879, 0.34822377242086583, 0.3266848028818515, 1.1414170966875377, 1.299758470774316, 1.005176411943463, 1.0769193031619044, 0.8872003122885062, 0.7853545670122154, 0.8376830845864721, 0.7594387919364166, 1.7457822520932191, 1.4007037442409944, 1.5130549509169442]

const REOPT_BUDGET = Dict(
    "upper" => Dict(8000 => 60.0, 20000 => 90.0, 80000 => 400.0),
    "lower" => Dict(8000 => 120.0, 20000 => 150.0, 80000 => 450.0),
)
const CANDIDATES = (
    upper = (label = "upper", find_smallest = true, w0 = w_upper, gradient_method = :lfix_composite),
    lower = (label = "lower", find_smallest = false, w0 = w_lower, gradient_method = :lfix_composite_fast),
)

const OPT_FILE_SR1 = joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt")
isfile(OPT_FILE_SR1) || error("missing $OPT_FILE_SR1")

# ---- CSV setup (write incrementally, flush every row -- survives a mid-grid crash with partial results) ----
CSV_COLS = ["W", "candidate", "phase", "gamma_focal_prime", "kappa", "Delta_dual", "Delta_primal",
            "gravity_value", "runtime_seconds", "feasible", "inner_status", "knitro_status",
            "h_diagnostic", "a_solution_key", "note"]
open(CSV_PATH, "w") do io
    println(io, join(CSV_COLS, ","))
end
function write_row!(row::Dict)
    open(CSV_PATH, "a") do io
        println(io, join((get(row, c, "") for c in CSV_COLS), ","))
    end
end

jld2_store = Dict{String,Any}()
function save_solution!(key, w, xf)
    jld2_store[key] = (w = collect(w), x_free = collect(xf))
    # NOTE: jldsave(path; dict...) requires Symbol keys (kwarg splat) -- our keys are Strings
    # (candidate/W/phase-composed labels), so use jldopen + explicit String-indexed assignment instead.
    # Rewrite whole file each time -- small (16-dim vectors), safe against a mid-grid crash.
    jldopen(A_SOLUTIONS_PATH, "w") do file
        for (k, v) in jld2_store
            file[k] = v
        end
    end
end

# ---- pivot elimination is structural (depends only on ctx.fixed_vals/q_tilde/N_obs, not U or find_smallest) --
# build once per W from a throwaway ctx, reuse for both candidates/phases at that W (verified equal to a
# find_smallest=false build too, mirroring c8_finalverify_lower_v2_opt.jl's own pe_prod==pe assertion).
function pivot_for_W(W)
    ctx_probe = build_nested_ctx(W; find_smallest = true)
    return build_pivot_elimination(ctx_probe)
end

# ---- one adaptive-h diagnostic per (W, candidate): h_sweep.jl's threshold-crossing rule, evaluated at
# the candidate's OWN point along the raw x_free A-block's first coordinate direction (a representative,
# reproducible probe direction -- NOT a full per-coordinate bandwidth selection, documented as such).
# min_switch_mass defaults to 1/W inside adaptive_h_candidate, so this genuinely shrinks as W grows,
# per the task's explicit requirement not to reuse a W=8000-tuned h unchecked at larger W.
include(joinpath(@__DIR__, "winner_switching.jl"))   # -> v_free_to_Amat, exact_tie_thresholds (adaptive_h_candidate's deps)
include(joinpath(@__DIR__, "h_sweep.jl"))
function h_diagnostic(ctx, xf, D2)
    # v_free here must be length(ctx.free_idx) == D2+1 (index 1 = gamma'_focal, SKIPPED internally by
    # v_free_to_Amat; indices 2:D2+1 = the D^2 raw A_od block) -- NOT the D2-length reduced-w vector.
    n_free = length(ctx.free_idx)
    v = zeros(n_free); v[2] = 1.0   # unit step in the first raw A_od coordinate (a representative probe direction)
    return adaptive_h_candidate(xf, v, ctx)
end

# ============================================================================
# Phase (a): re-evaluation at fixed (g,A), draws swapped to this W's prefix
# ============================================================================
println("\n" * "="^78); println("PHASE A: RE-EVALUATION (fixed candidate point, draws swapped per W)"); println("="^78); flush(stdout)
for W in W_GRID
    pe = pivot_for_W(W)
    D2 = length(w_upper)
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    ctx = build_nested_ctx(W; find_smallest = true)   # Delta_dual/kappa/gravity are all independent of find_smallest -- reused for both candidates
    for cand in (CANDIDATES.upper, CANDIDATES.lower)
        xf = x_free_from_w(cand.w0)
        t0 = time()
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = false)   # cold
        dt = time() - t0
        κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
        feasible = isfinite(r.Delta_dual) && r.Delta_dual <= ctx.δ + 1e-6
        h = h_diagnostic(ctx, xf, D2)
        key = "$(cand.label)_W$(W)_reeval"
        save_solution!(key, cand.w0, xf)
        @printf("[reeval W=%d %s] kappa=%.10f Delta_dual=%.8f Delta-delta=%+.4e gravity=%.3e feasible=%s inner_status=%d h=%.5f runtime=%.3fs\n",
            W, cand.label, κ, r.Delta_dual, r.Delta_dual - ctx.δ, r.gravity_value, feasible, r.inner_status, h, dt)
        write_row!(Dict("W" => W, "candidate" => cand.label, "phase" => "reeval",
            "gamma_focal_prime" => r.gamma_focal_prime, "kappa" => κ, "Delta_dual" => r.Delta_dual,
            "Delta_primal" => r.Delta_primal, "gravity_value" => r.gravity_value, "runtime_seconds" => dt,
            "feasible" => feasible, "inner_status" => r.inner_status, "knitro_status" => "",
            "h_diagnostic" => h, "a_solution_key" => key, "note" => "cold_reeval_at_candidate_w"))
        flush(stdout)
    end
end

# ============================================================================
# Phase (b): re-optimization warm-started from the candidate's own w
# ============================================================================
println("\n" * "="^78); println("PHASE B: RE-OPTIMIZATION (warm-started from candidate's own w)"); println("="^78); flush(stdout)

function run_reopt(W::Int, cand)
    pe = pivot_for_W(W)
    D2 = length(cand.w0)
    x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
    ctx = build_nested_ctx(W; find_smallest = cand.find_smallest, outer_loop_opt = OPT_FILE_SR1)
    pe2 = build_pivot_elimination(ctx)
    @assert pe2.pivot_lin == pe.pivot_lin && pe2.c == pe.c && pe2.g0 == pe.g0 "pivot elimination differs across find_smallest -- investigate before trusting w0"

    budget = REOPT_BUDGET[cand.label][W]
    w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
    w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))
    @assert all(w_lo .<= cand.w0 .<= w_hi) "candidate w0 violates production box bounds at W=$W"

    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
    n_eval = Ref(0)
    last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)

    function record!(w, Δ, r)
        n_eval[] += 1
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        better = best_feasible[] === nothing || (cand.find_smallest ? w[1] < best_feasible[].gp : w[1] > best_feasible[].gp)
        if feasible && better
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ)
        end
    end

    function eval_F(w::Vector{Float64})
        xf = x_free_from_w(w)
        r = evaluate_fullA(xf, ctx; cache = nothing, warm = true)
        record!(w, r.Delta_dual, r)
        if cand.gradient_method == :lfix_composite_fast
            if r.inner_status in (0, -100, -101, -103)
                base = BaseDualState(xf, r.θ_full, r.zeta, r.lambda, copy(ctx.obj.arg1), r.inner_status)
                last_F_state[] = (w = copy(w), base = base)
            else
                last_F_state[] = nothing
            end
        end
        return r.Delta_dual, r
    end

    function eval_grad_dispatch(w::Vector{Float64})
        xf = x_free_from_w(w)
        if cand.gradient_method == :lfix_composite
            g, meta = composite_gradient_at(xf, ctx, pe)
            return g
        else   # :lfix_composite_fast
            shared = last_F_state[]
            base = (shared !== nothing && shared.w == w) ? shared.base : nothing
            g, meta = composite_gradient_at_fast(xf, ctx, pe; base = base, threaded = true, h_mode = :adaptive)
            return g
        end
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, OPT_FILE_SR1)
    KNITRO.KN_set_param_by_name(kc, "maxtime_real", budget)
    KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, cand.w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        Δ, r = eval_F(w)
        evalResult.obj[1] = cand.find_smallest ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = cand.find_smallest ? 1.0 : -1.0
        evalResult.jac .= eval_grad_dispatch(w)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    println(">>> reopt W=$W candidate=$(cand.label) gradient_method=$(cand.gradient_method) hessopt=sr1 budget=$(budget)s")
    flush(stdout)
    t0 = time()
    open(joinpath(OUTDIR, "knitro_$(cand.label)_W$(W).log"), "w") do io
        redirect_stdout(io) do
            KNITRO.KN_solve(kc)
        end
    end
    wall = time() - t0
    nStatus, objv, w_min, lambda_ = KNITRO.KN_get_solution(kc)
    KNITRO.KN_free(kc)

    @printf("    knitro_status=%d wall=%.1fs n_eval=%d best_feasible_found=%s\n",
        nStatus, wall, n_eval[], best_feasible[] !== nothing)
    flush(stdout)

    # exact fresh cold feasibility recheck of the tracked best-feasible point (never trust the raw
    # terminal iterate, per this workstream's standing convention)
    if best_feasible[] === nothing
        return (W = W, cand = cand, nStatus = nStatus, wall = wall, n_eval = n_eval[], ctx = ctx, pe = pe,
                w_final = nothing, r = nothing, converged = nStatus in (-100, -101, -102, -103))
    end
    xf_best = x_free_from_w(best_feasible[].w)
    r = evaluate_fullA(xf_best, ctx; cache = nothing, warm = false)
    converged = nStatus in (-100, -101, -102, -103)
    return (W = W, cand = cand, nStatus = nStatus, wall = wall, n_eval = n_eval[], ctx = ctx, pe = pe,
            w_final = best_feasible[].w, r = r, converged = converged)
end

for cand in (CANDIDATES.upper, CANDIDATES.lower)
    for W in W_GRID
        res = run_reopt(W, cand)
        D2 = length(cand.w0)
        if res.w_final === nothing
            @printf("[reopt W=%d %s] NO FEASIBLE POINT FOUND (knitro_status=%d, converged=%s)\n", W, cand.label, res.nStatus, res.converged)
            write_row!(Dict("W" => W, "candidate" => cand.label, "phase" => "reopt",
                "gamma_focal_prime" => "", "kappa" => "", "Delta_dual" => "", "Delta_primal" => "",
                "gravity_value" => "", "runtime_seconds" => res.wall, "feasible" => false,
                "inner_status" => "", "knitro_status" => res.nStatus, "h_diagnostic" => "",
                "a_solution_key" => "", "note" => (res.converged ? "converged_but_no_feasible_iterate" : "best_feasible_stalled_no_point")))
        else
            r = res.r; ctx = res.ctx
            κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
            feasible = isfinite(r.Delta_dual) && r.Delta_dual <= ctx.δ + 1e-6
            xf_best = vcat(res.w_final[1], vec(exp.(pivot_expand(res.w_final[2:end], res.pe))))
            h = h_diagnostic(ctx, xf_best, D2)
            key = "$(cand.label)_W$(W)_reopt"
            save_solution!(key, res.w_final, xf_best)
            note = res.converged ? "converged_best_feasible_tracked_cold_recheck" : "best_feasible_stalled_at_budget_cold_recheck"
            @printf("[reopt W=%d %s] kappa=%.10f Delta_dual=%.8f Delta-delta=%+.4e gravity=%.3e feasible=%s inner_status=%d h=%.5f knitro_status=%d wall=%.1fs note=%s\n",
                W, cand.label, κ, r.Delta_dual, r.Delta_dual - ctx.δ, r.gravity_value, feasible, r.inner_status, h, res.nStatus, res.wall, note)
            write_row!(Dict("W" => W, "candidate" => cand.label, "phase" => "reopt",
                "gamma_focal_prime" => r.gamma_focal_prime, "kappa" => κ, "Delta_dual" => r.Delta_dual,
                "Delta_primal" => r.Delta_primal, "gravity_value" => r.gravity_value, "runtime_seconds" => res.wall,
                "feasible" => feasible, "inner_status" => r.inner_status, "knitro_status" => res.nStatus,
                "h_diagnostic" => h, "a_solution_key" => key, "note" => note))
        end
        flush(stdout)
    end
end

println("\nWrote ", CSV_PATH)
println("Wrote ", A_SOLUTIONS_PATH)
println("DONE")
