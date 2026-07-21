# ============================================================================
# Negative-cache audit (integration/fullA-negative-cache-audit), live D=20
# experiment. Tasks 2(supplement)/3/4 of the audit brief:
#
#   A. Capture fresh, CURRENT-architecture organic -300 points at D=20,
#      W=80000, delta=5, real production draw_seed=20260719 -- via the same
#      deterministic perturbation-sweep methodology
#      c9_infscreen_d20_workloads.jl's "Class 4 search" used (that file's own
#      run predates this branch's later screens and used an UNSEEDED ctx.U,
#      so its exact points cannot be reproduced here -- this rebuilds fresh
#      under this worktree's current screen stack + explicit seeding).
#   B. At each captured organic point: 7 start-policy variants + repeats.
#   C. Independent LP feasibility certificate (HiGHS, ported from
#      phaseF_primal_feasibility_lp.jl) at D=20/W=80000 scale.
#   D. Dual-bank rescue simulation (task 4): does a later, better-populated
#      DualBank ever rescue a point that failed under a weak start?
#
# All results written to CSV/JSON under results/fullA_d4/<commit>/negcache_audit/.
# ============================================================================
include(joinpath(@__DIR__, "draw_design.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
include(joinpath(@__DIR__, "three_way_derivatives.jl"))
include(joinpath(@__DIR__, "lfix_incremental.jl"))
include(joinpath(@__DIR__, "compressed_moments.jl"))
include(joinpath(@__DIR__, "structured_moment_build.jl"))
include(joinpath(@__DIR__, "compressed_cc_inner.jl"))
include(joinpath(@__DIR__, "oracle_fast.jl"))
include(joinpath(@__DIR__, "compressed_live.jl"))
include(joinpath(@__DIR__, "fast_range_screen.jl"))
include(joinpath(@__DIR__, "dual_bank.jl"))
using Random, Printf, Dates, LinearAlgebra, Statistics, Serialization
using JuMP, HiGHS

const FEASIBLE_CODES = (0, -100, -101, -103)
const SCREEN_INFEAS = (:pairwise_certified_infeasible, :witness_certified_infeasible,
                        :winner_scan_infeasible, :EXACT_INFEASIBLE_PREWINNER_ENVELOPE,
                        :EXACT_INFEASIBLE_WINNING_RANGE, :EXACT_INFEASIBLE_MOMENT_RANGE)

COMMIT = strip(read(`git rev-parse --short HEAD`, String))
OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "negcache_audit")
mkpath(OUTDIR)
LOGIO = open(joinpath(OUTDIR, "log.txt"), "w")
lp(xs...) = (println(xs...); println(LOGIO, xs...); flush(stdout); flush(LOGIO))

lp("="^90); lp("negcache_audit_experiment.jl starting ", now(), "  commit=", COMMIT); lp("="^90)

# ---- A. context, real production defaults ----
const W = 80000
const DELTA = 5.0
const DRAW_SEED = 20260719
t0 = time()
ctx = d20_real_setup_design(W = W, δ = DELTA, find_smallest = true, draw_design = :pseudorandom, draw_seed = DRAW_SEED)
lp("ctx build: wall=", round(time() - t0, digits = 2), "s  draw_checksum=(", ctx.draw_meta.checksum_uniform, ",", ctx.draw_meta.checksum_transformed, ")")
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
gp0 = ctx.θ0_up[3+D]
xf_nat = ctx.θ0_up[ctx.free_idx]
zfree0 = pivot_reduce(log.(reshape(xf_nat[2:end], D, D)), pe)
lp("D=", D, "  n_free_A=", D2 - 1, "  envelope_supported=", rsc.envelope !== nothing)

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))

function full_eval(xf::Vector{Float64}; warm::Bool = true, cache = nothing)
    return evaluate_fullA_screened_ranged(xf, ctx, rsc; moment_representation = :compressed,
        cache = cache, use_cache = cache !== nothing, warm = warm, tag = "",
        pairwise = ctx.pairwise, witness = ctx.witness, use_witness = ctx.witness !== nothing)
end

# seed the compressed warm-start cache with a cold solve first (matches production's own
# documented need -- a fresh ctx's very first warm=true call has no prior state).
r_seed, _ = full_eval(x_free_from_w(vcat(gp0, zfree0)); warm = false)
lp("seed (calibration, cold): status=", r_seed.inner_status, " Delta=", r_seed.Delta_dual)
r_seed.inner_status in FEASIBLE_CODES || error("calibration point itself not feasible -- cannot proceed")

# ============================================================================
# STEP A: deterministic perturbation sweep for organic screen-passing -300
# ============================================================================
Random.seed!(777001)
STEPS = [1.0, 2.0, 4.0, 8.0, 16.0, 24.0]
N_PER_STEP = 8
organic = NamedTuple[]
bank_feed = NamedTuple[]   # (order, zfree, x_solved) for feasible trials -- used to build DualBank later
trial_rows = NamedTuple[]
neval = 0
for step in STEPS
    for i in 1:N_PER_STEP
        global neval
        dir = randn(D2 - 1); dir ./= norm(dir)
        zfree_t = zfree0 .+ step .* dir
        xf = x_free_from_w(vcat(gp0, zfree_t))
        t0i = time()
        r, meta = full_eval(xf; warm = false)   # cold: worst case, matches class4 methodology
        twall = time() - t0i
        neval += 1
        screen_infeas = meta.screen_status in SCREEN_INFEAS
        push!(trial_rows, (neval = neval, step = step, i = i, screen_status = meta.screen_status,
                            inner_status = r.inner_status, Delta_dual = r.Delta_dual, wall = twall))
        if !screen_infeas && r.inner_status in FEASIBLE_CODES
            push!(bank_feed, (order = neval, zfree = copy(zfree_t), x_solved = vcat(r.zeta, r.lambda)))
        end
        if !screen_infeas && !(r.inner_status in FEASIBLE_CODES)
            lp(">>> ORGANIC CANDIDATE: step=", step, " i=", i, " inner_status=", r.inner_status,
               " screen_status=", meta.screen_status, " wall=", round(twall, digits = 2), "s")
            push!(organic, (step = step, i = i, gp = gp0, zfree = copy(zfree_t), xf = copy(xf),
                             cold_status = r.inner_status, cold_wall = twall))
        end
    end
    lp("step=", step, " done: organic_so_far=", length(organic), " bank_feed_so_far=", length(bank_feed))
    length(organic) >= 4 && break
end

open(joinpath(OUTDIR, "sweepA_trials.csv"), "w") do io
    println(io, "neval,step,i,screen_status,inner_status,Delta_dual,wall")
    for r in trial_rows
        println(io, r.neval, ",", r.step, ",", r.i, ",", r.screen_status, ",", r.inner_status, ",", r.Delta_dual, ",", r.wall)
    end
end
lp("\nSTEP A complete: ", length(organic), " organic (screen-pass, KNITRO-fail) candidate(s) found among ", neval, " trials.")

if isempty(organic)
    lp("NO organic -300 candidates found in this sweep -- widen STEPS/N_PER_STEP in a follow-up. Stopping here.")
    close(LOGIO)
    exit(0)
end

# save organic candidates' exact vectors (task's explicit requirement: recover exact outer vectors)
for (k, o) in enumerate(organic)
    serialize(joinpath(OUTDIR, "organic_candidate_$(k).jls"),
        (gp = o.gp, zfree = o.zfree, xf = o.xf, step = o.step, i = o.i, cold_status = o.cold_status,
         W = W, delta = DELTA, draw_seed = DRAW_SEED,
         draw_checksum_uniform = ctx.draw_meta.checksum_uniform, draw_checksum_transformed = ctx.draw_meta.checksum_transformed))
end

# ============================================================================
# STEP B: 7 start-policy variants + repeats, at each organic candidate
# ============================================================================
lp("\n", "="^90); lp("STEP B: multi-start variant test at each organic candidate"); lp("="^90)

# small DualBank fed from the sweep's own successful trials (chronological)
bank = DualBank(8)
for bf in bank_feed
    record_success!(bank, bf.order, bf.zfree, bf.x_solved)
end
lp("DualBank populated with ", length(bank.history), " successful trial(s) from the sweep.")

nvar_inner = ctx.obj.outer_constr_index   # ζ + λ's, matches zeros(obj.outer_constr_index) convention (dual_bank.jl)

variant_rows = NamedTuple[]
for (k, o) in enumerate(organic)
    lp("\n--- organic candidate $k (step=$(o.step), i=$(o.i)) ---")
    xf = o.xf; zfree_t = o.zfree

    # (1) neutral dual (zeros) -- warm=false forces obj.x .= NaN -> falls back to zeros
    for rep in 1:3
        t0v = time()
        r, _ = full_eval(xf; warm = false)
        tw = time() - t0v
        push!(variant_rows, (candidate = k, variant = "1_neutral_rep$(rep)", inner_status = r.inner_status,
                              Delta_dual = r.Delta_dual, wall = tw, dual_norm = NaN))
        lp("  [1_neutral rep$rep] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2))
    end

    # (2) "original production warm start": whatever obj.x currently holds after the neutral
    # reps above (obj.x .= NaN on failure -- this IS what a real production run's NEXT callback
    # call would see immediately after an organic -300, i.e. the realistic worst case)
    ctx.obj.x .= NaN
    t0v = time(); r, _ = full_eval(xf; warm = true); tw = time() - t0v
    push!(variant_rows, (candidate = k, variant = "2_poisoned_prod_slot", inner_status = r.inner_status,
                          Delta_dual = r.Delta_dual, wall = tw, dual_norm = NaN))
    lp("  [2_poisoned_prod_slot] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2))

    # (3) last accepted successful dual (bank's most recent entry)
    if !isempty(bank.history)
        x0 = bank.history[end].x_solved
        ctx.obj.x .= x0
        t0v = time(); r, _ = full_eval(xf; warm = true); tw = time() - t0v
        push!(variant_rows, (candidate = k, variant = "3_last_accepted", inner_status = r.inner_status,
                              Delta_dual = r.Delta_dual, wall = tw, dual_norm = norm(x0)))
        lp("  [3_last_accepted] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2), " |x0|=", round(norm(x0), digits = 3))
    end

    # (4) nearest successful dual from the bank (by scaled zfree distance -- reuses select_warm_start)
    cf_score = try
        θ_full_score = CS.reconstruct_full(xf, ctx.m)
        build_compressed_factual(θ_full_score, ctx; check_ties = true)
    catch e
        e isa TiedWinnerError ? nothing : rethrow()
    end
    if cf_score !== nothing && !isempty(bank.history)
        x0, label = select_warm_start(bank, ctx.obj, cf_score, zfree_t)
        ctx.obj.x .= x0
        t0v = time(); r, _ = full_eval(xf; warm = true); tw = time() - t0v
        push!(variant_rows, (candidate = k, variant = "4_bank_selected($label)", inner_status = r.inner_status,
                              Delta_dual = r.Delta_dual, wall = tw, dual_norm = norm(x0)))
        lp("  [4_bank_selected($label)] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2))
    end

    # (5) deliberately damped dual: 0.1x and 0.5x of the nearest bank candidate
    if !isempty(bank.history)
        base_x = bank.history[end].x_solved
        for frac in (0.1, 0.5)
            ctx.obj.x .= frac .* base_x
            t0v = time(); r, _ = full_eval(xf; warm = true); tw = time() - t0v
            push!(variant_rows, (candidate = k, variant = "5_damped_$(frac)x", inner_status = r.inner_status,
                                  Delta_dual = r.Delta_dual, wall = tw, dual_norm = norm(frac .* base_x)))
            lp("  [5_damped_$(frac)x] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2))
        end
    end

    # (6) "fresh KNITRO context" -- structurally ALWAYS true in this codebase: inner_loop_KNITRO
    # calls KNITRO.KN_new() on every single call (cc_algo/inner_loop_functions.jl:55), so every
    # variant above (and every real production call) already uses a brand-new context each time.
    # Recorded here as an explicit repeat of the neutral-start case to document that fact plainly
    # (not a separate code path -- there is none to test).
    t0v = time(); r, _ = full_eval(xf; warm = false); tw = time() - t0v
    push!(variant_rows, (candidate = k, variant = "6_fresh_context_confirms_always_true", inner_status = r.inner_status,
                          Delta_dual = r.Delta_dual, wall = tw, dual_norm = NaN))
    lp("  [6_fresh_context (structurally always true)] status=", r.inner_status, " Delta=", r.Delta_dual)

    # (7) larger iteration budget: temp .opt file with maxit raised 100 -> 5000
    loose_opt = joinpath(OUTDIR, "ek_inner_maxit5000.opt")
    if !isfile(loose_opt)
        base_opt = read(joinpath(D4X_ROOT, "full_aod_diag", "ek_inner.opt"), String)
        open(loose_opt, "w") do io
            for line in split(base_opt, '\n')
                if startswith(line, "maxit ") || startswith(line, "maxit\t")
                    println(io, "maxit        5000")
                else
                    println(io, line)
                end
            end
        end
    end
    old_opt = ctx.obj.inner_loop_opt
    ctx.obj.inner_loop_opt = loose_opt
    ctx.obj.x .= NaN
    t0v = time(); r, _ = full_eval(xf; warm = false); tw = time() - t0v
    ctx.obj.inner_loop_opt = old_opt
    push!(variant_rows, (candidate = k, variant = "7_maxit5000_cold", inner_status = r.inner_status,
                          Delta_dual = r.Delta_dual, wall = tw, dual_norm = NaN))
    lp("  [7_maxit5000_cold] status=", r.inner_status, " Delta=", r.Delta_dual, " wall=", round(tw, digits = 2))
end

open(joinpath(OUTDIR, "sweepB_variants.csv"), "w") do io
    println(io, "candidate,variant,inner_status,Delta_dual,wall,dual_norm")
    for r in variant_rows
        println(io, r.candidate, ",", r.variant, ",", r.inner_status, ",", r.Delta_dual, ",", r.wall, ",", r.dual_norm)
    end
end
n_rescued = count(r -> !(r.candidate == 0) && r.inner_status in FEASIBLE_CODES, variant_rows)
lp("\nSTEP B complete: ", n_rescued, " / ", length(variant_rows), " variant attempts returned a FEASIBLE status ",
   "(any >0 among candidates' variants means at least one alternative start RESCUED an organic failure).")

# ============================================================================
# STEP C: independent LP feasibility certificate (HiGHS), at organic candidates
# ============================================================================
lp("\n", "="^90); lp("STEP C: independent LP feasibility certificate (HiGHS)"); lp("="^90)
Wn = size(ctx.obj.U, 1); dmom = ctx.obj.d

function lp_feasibility_check(xf::Vector{Float64}, label::String)
    θ_full = CS.reconstruct_full(xf, ctx.m)
    K = zeros(Wn); G = zeros(Wn, dmom)
    ctx.obj.moments!(K, G, θ_full, ctx.obj.U, ctx.obj)
    t0lp = time()
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, m[1:Wn] >= 0)
    @constraint(model, sum(m) / Wn == 1)
    @constraint(model, moments[j = 1:dmom], sum(m[s] * G[s, j] for s in 1:Wn) / Wn == 0)
    @objective(model, Min, 0)
    optimize!(model)
    status = termination_status(model)
    feasible_exact = status == MOI.OPTIMAL
    phase1_max_resid = NaN
    if !feasible_exact
        model2 = Model(HiGHS.Optimizer); set_silent(model2)
        @variable(model2, m2[1:Wn] >= 0)
        @variable(model2, t >= 0)
        @constraint(model2, sum(m2) / Wn == 1)
        @constraint(model2, [j = 1:dmom], sum(m2[s] * G[s, j] for s in 1:Wn) / Wn <= t)
        @constraint(model2, [j = 1:dmom], sum(m2[s] * G[s, j] for s in 1:Wn) / Wn >= -t)
        @objective(model2, Min, t)
        optimize!(model2)
        phase1_max_resid = termination_status(model2) == MOI.OPTIMAL ? value(t) : NaN
    end
    tlp = time() - t0lp
    classification = if feasible_exact
        "FEASIBLE_LP"
    elseif isfinite(phase1_max_resid) && phase1_max_resid > 1e-6
        "CERTIFIED_INFEASIBLE"
    elseif isfinite(phase1_max_resid)
        "NUMERICALLY_BORDERLINE"
    else
        "NUMERICALLY_UNRESOLVED"
    end
    lp("  [", label, "] lp1_status=", status, " classification=", classification,
       " phase1_max_resid=", phase1_max_resid, " wall=", round(tlp, digits = 2), "s")
    return (label = label, lp1_status = string(status), feasible_exact = feasible_exact,
            phase1_max_resid = phase1_max_resid, classification = classification, wall = tlp)
end

lp_rows = NamedTuple[]
# sanity check on the LP machinery: calibration point must be FEASIBLE_LP
push!(lp_rows, lp_feasibility_check(x_free_from_w(vcat(gp0, zfree0)), "calibration_SANITY_CHECK"))
for (k, o) in enumerate(organic)
    push!(lp_rows, lp_feasibility_check(o.xf, "organic_candidate_$(k)"))
end
open(joinpath(OUTDIR, "sweepC_lp_results.csv"), "w") do io
    println(io, "label,lp1_status,feasible_exact,phase1_max_resid,classification,wall")
    for r in lp_rows
        println(io, r.label, ",", r.lp1_status, ",", r.feasible_exact, ",", r.phase1_max_resid, ",", r.classification, ",", r.wall)
    end
end

# ============================================================================
# STEP D: dual-bank rescue simulation (task 4's central scenario)
# ============================================================================
lp("\n", "="^90); lp("STEP D: dual-bank rescue simulation"); lp("="^90)
lp("For each organic candidate: (a) weak-start failure already shown in Step B variant 1/2.")
lp("(b) bank already populated from ", length(bank_feed), " nearby successful sweep trials.")
lp("(c)+(d) revisit via select_warm_start's BEST-scoring candidate (already run as variant 4 above).")
lp("Explicit verdict per candidate:")
for (k, o) in enumerate(organic)
    vs = filter(r -> r.candidate == k, variant_rows)
    any_rescued = any(r -> r.inner_status in FEASIBLE_CODES, vs)
    lp("  candidate $k: any variant rescued the point? ", any_rescued,
       any_rescued ? "  <-- FALSIFIES first-failure caching for this point" : "  (consistent with genuine, start-independent infeasibility)")
end

lp("\nnegcache_audit_experiment.jl COMPLETE at ", now())
close(LOGIO)
