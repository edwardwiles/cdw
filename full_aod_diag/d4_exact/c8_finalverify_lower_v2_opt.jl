# ============================================================================
# Continuation 8, Section 8 (final D=4 candidate verification), Part 0.
#
# docs/fullA_gamma_profile_two_branches_c8.md (this session, Wave 2C) found
# that the REGISTERED lower incumbent (lower_lfixcomposite_fast_sr1_300s,
# kappa=0.005428799948779983, g=0.9967391744173478) sits ON the Delta=delta
# boundary using its OWN A, but that A is only a locally-KKT-stationary
# point, NOT the true constrained minimizer A*=argmin_A Delta(g,A) at that g
# -- an independent search found Delta(g_incumbent, A_true_min) ~= 0.892 (11%
# below delta=1) at the SAME g, and separately bracketed the TRUE
# profile_Delta(g)=delta crossing at g ~= 0.997031 (implied kappa ~= 0.004944,
# ~8.9% tighter). That workstream explicitly did NOT push this through to a
# genuine, freshly-optimized (g,A) candidate suitable for registration -- it
# was a multistart/profile finding only. THIS script does that.
#
# Step 1: take the high-g branch's best stored A-solution near the crossing
# (results/fullA_d4/128f260/c8_gammabranch_a_solutions.jld2, row40,
# g=0.99703076171875, the feasible-side bracket edge from
# c8_gammabranch_highg_refine.jl's own bisection -- see
# results/fullA_d4/128f260/c8_gammabranch_highg_refine/highg_bisection_trace.csv)
# and RE-REFINE it with a real per-g constrained-A local solve (reusing
# c8_gammabranch_core.jl's `profile_delta_at_gamma_c8`, read-only, exactly as
# c8_gammabranch_highg_refine.jl's own `best_of_three` does), from TWO
# anchors (the stored row40 A, and the registered lower incumbent's own A),
# to get a solid, well-converged local A-minimizer at this g -- NOT trusting
# the stored JLD2 point blindly (the profile table's own documented caveat:
# the stored a_solution_key entries come from a separate single-chain
# re-derivation pass that can land in a different, non-best basin).
#
# Step 2: use the result as the WARM START for a genuine candidate-quality
# joint (g,A) outer-loop optimization -- the SAME class of run that produced
# the registered lower incumbent (run_d4_optimized_fd.jl direction=lower:
# gradient_method=lfix_composite_fast, hessopt=sr1, eval_fcga=no via
# csw_outer_wallclock_sr1.opt, wall-clock budget the binding stop criterion).
# This script reproduces that driver's exact KNITRO wiring (NOT by editing
# run_d4_optimized_fd.jl -- new file, per this workstream's file-ownership
# convention) but starting from the refined near-crossing point instead of
# the calibration default.
#
# Run: source .knitro_env.sh (plain source, NOT in a pipe) then
#   JULIA_NUM_THREADS=20 D4X_MAXTIME_REAL=300 julia --project=. \
#     full_aod_diag/d4_exact/c8_finalverify_lower_v2_opt.jl
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))   # ctx(find_smallest=true), pe, D, D2, profile_delta_at_gamma_c8,
                                                          # x_free_from_w, ZFREE_LOWER_INCUMBENT, W_LOWER_INCUMBENT,
                                                          # lp_feasibility_check_c8, composite_gradient_at_fast (via gamma_profile.jl)
using KNITRO, JLD2, Printf, Dates

const COMMIT_C8 = strip(read(`git -C $(D4X_ROOT) rev-parse --short HEAD`, String))
const RUN_ID = "c8_finalverify_lower_v2_opt_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, RUN_ID)
mkpath(OUTDIR)
println(">>> c8_finalverify_lower_v2_opt.jl  commit=$COMMIT_C8  outdir=$OUTDIR")
flush(stdout)

# ============================================================================
# Step 1: refine the high-g branch's near-crossing A at g=0.99703076171875
# ============================================================================
const G_TARGET = 0.99703076171875   # highg_refine's feasible-side bracket edge (width 4.9e-7 to the infeasible side)
const A_SOLUTIONS_PATH = joinpath(D4X_ROOT, "results", "fullA_d4", "128f260", "c8_gammabranch_a_solutions.jld2")
const ROW40_KEY = "row40_g0.99703076"

println("="^78); println("STEP 1: re-refine A at g=$G_TARGET from 2 anchors"); println("="^78); flush(stdout)

Aod_row40 = load(A_SOLUTIONS_PATH)["Aod_jld2"][ROW40_KEY]
zfree_row40 = pivot_reduce(log.(reshape(Aod_row40, D, D)), pe)
println("loaded row40 stored A (g=0.99703076171875 row, single-chain re-derivation) as anchor 1")
println("using registered lower incumbent's own A (at its own g=0.9967391744173478) as anchor 2 (cold-started at G_TARGET)")

refine_row40  = profile_delta_at_gamma_c8(G_TARGET, zfree_row40, ctx, pe; moment_repr = :compressed, maxtime_real = 60.0, hessopt_tag = "sr1")
refine_lowinc = profile_delta_at_gamma_c8(G_TARGET, ZFREE_LOWER_INCUMBENT, ctx, pe; moment_repr = :compressed, maxtime_real = 60.0, hessopt_tag = "sr1")

cands = [("row40_anchor", refine_row40), ("lower_incumbent_anchor", refine_lowinc)]
for (kind, r) in cands
    @printf("  [%s] best_Delta=%s  knitro_status=%d  n_eval=%d  wall=%.1fs\n",
        kind, r.best_zfree === nothing ? "INFEASIBLE" : @sprintf("%.6e", r.best_Delta), r.knitro_status, r.n_eval, r.wall)
end
feas_cands = filter(c -> c[2].best_zfree !== nothing && isfinite(c[2].best_Delta), cands)
isempty(feas_cands) && error("STEP 1 FAILED: neither anchor produced a feasible A at g=$G_TARGET -- cannot proceed to step 2")
best_kind, best_refine = feas_cands[argmin([c[2].best_Delta for c in feas_cands])]
zfree_refined = copy(best_refine.best_zfree)
Delta_refined = best_refine.best_Delta
@printf("\nSTEP 1 result: best anchor=%s, Delta(G_TARGET, A_refined)=%.8f (Delta-delta=%+.4e)\n",
    best_kind, Delta_refined, Delta_refined - ctx.δ)
kappa_step1 = 1 - G_TARGET^(ctx.σ / (ctx.σ - 1))
println("(implied kappa purely from G_TARGET, before any further outer-loop movement of g itself: ", kappa_step1, ")")

w0 = vcat(G_TARGET, zfree_refined)

# double-check with a fully independent, cold, DENSE evaluate_fullA call (not the compressed/warm profile solve)
let xf = x_free_from_w(w0)
    r_cold = evaluate_fullA(xf, ctx; cache = nothing, warm = false)
    @printf("STEP 1 cold dense recheck: Delta_dual=%.8f  Delta_primal=%.8f  gravity_value=%.3e  inner_status=%d\n",
        r_cold.Delta_dual, r_cold.Delta_primal, r_cold.gravity_value, r_cold.inner_status)
end
flush(stdout)

# ============================================================================
# Step 2: genuine candidate-quality joint (g,A) outer-loop optimization,
# warm-started from w0, direction=lower (find_smallest=false, matching
# run_d4_optimized_fd.jl's DIRECTION=="lower" convention and
# candidate_registry.jl's own w_lower_lfixcomposite_fast entry), gradient
# lfix_composite_fast + hessopt=sr1 (exact match to how the registered lower
# incumbent was produced -- see docs/fullA_priority4_gamma_profile_and_lower.md).
# ============================================================================
println("\n" * "="^78); println("STEP 2: production joint (g,A) outer-loop optimization from w0, direction=lower"); println("="^78); flush(stdout)

const FIND_SMALLEST_PROD = false   # "lower" direction
const MAXTIME_REAL = parse(Float64, get(ENV, "D4X_MAXTIME_REAL", "300"))
const OPT_FILE_PROD = joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt")
isfile(OPT_FILE_PROD) || error("missing $OPT_FILE_PROD")

ctx_prod = d4_exact_setup(find_smallest = FIND_SMALLEST_PROD, outer_loop_opt = OPT_FILE_PROD)
pe_prod = build_pivot_elimination(ctx_prod)
# sanity: pe_prod must match pe (core.jl's, find_smallest=true) exactly -- gravity coefficients/pivot choice
# are structural (depend on ctx.fixed_vals/q_tilde/N_obs, not on find_smallest), so building two ctx's should
# give byte-identical pivot elimination. Verified here, not assumed.
@assert pe_prod.pivot_lin == pe.pivot_lin && pe_prod.c == pe.c && pe_prod.g0 == pe.g0 "pe_prod != pe -- find_smallest changed the gravity elimination structurally, which should be impossible; investigate before trusting w0's coordinates in ctx_prod"
println("pe_prod == pe verified (gravity elimination structurally independent of find_smallest, as expected)")

D2 = D^2
x_free_from_w_prod(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe_prod))))
function Delta_of_w_prod(w)
    r = evaluate_fullA(x_free_from_w_prod(w), ctx_prod; cache = nothing, warm = true)
    return r.Delta_dual, r
end

w_lo = vcat(ctx_prod.bounds.γp_lo, fill(-8.0, D2 - 1))
w_hi = vcat(ctx_prod.bounds.γp_hi, fill(8.0, D2 - 1))
@assert all(w_lo .<= w0 .<= w_hi) "w0 (refined near-crossing point) violates the production box bounds -- cannot use as primal init"

best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
callback_log = NamedTuple[]
n_eval = Ref(0)
last_F_state = Ref{Union{Nothing,NamedTuple}}(nothing)

function record!(w, Δ, r, kind)
    n_eval[] += 1
    feasible = isfinite(Δ) && Δ <= ctx_prod.δ + 1e-6
    if feasible && (best_feasible[] === nothing || w[1] > best_feasible[].gp)   # find_smallest=false: LARGER w[1] (g) is better
        best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ, gravity_resid = abs(r.gravity_value))
    end
    push!(callback_log, (idx = n_eval[], kind = kind, gp = w[1], Delta = Δ, feasible = feasible,
                          inner_status = r.inner_status, gravity_value = r.gravity_value))
end

function eval_F(w::Vector{Float64})
    Δ, r = Delta_of_w_prod(w)
    record!(w, Δ, r, "F")
    if r.inner_status in (0, -100, -101, -103)
        base = BaseDualState(x_free_from_w_prod(w), r.θ_full, r.zeta, r.lambda, copy(ctx_prod.obj.arg1), r.inner_status)
        last_F_state[] = (w = copy(w), base = base)
    else
        last_F_state[] = nothing
    end
    return Δ, r
end

function eval_grad_dispatch(w::Vector{Float64})
    xf = x_free_from_w_prod(w)
    shared = last_F_state[]
    base = (shared !== nothing && shared.w == w) ? shared.base : nothing
    g, meta = composite_gradient_at_fast(xf, ctx_prod, pe_prod; base = base, threaded = true, h_mode = :adaptive)
    return g
end

kc = KNITRO.KN_new()
KNITRO.KN_load_param_file(kc, OPT_FILE_PROD)
KNITRO.KN_set_param_by_name(kc, "maxtime_real", MAXTIME_REAL)
KNITRO.KN_set_param_by_name(kc, "maxit", 1_000_000)
xIndices = KNITRO.KN_add_vars(kc, D2)
KNITRO.KN_set_var_lobnds_all(kc, w_lo)
KNITRO.KN_set_var_upbnds_all(kc, w_hi)
KNITRO.KN_set_var_primal_init_values_all(kc, w0)
cIndices = KNITRO.KN_add_cons(kc, 1)
KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx_prod.δ)

function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
    w = evalRequest.x
    Δ, r = eval_F(w)
    evalResult.obj[1] = FIND_SMALLEST_PROD ? w[1] : -w[1]
    evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
    return 0
end
function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
    w = evalRequest.x
    evalResult.objGrad .= 0.0; evalResult.objGrad[1] = FIND_SMALLEST_PROD ? 1.0 : -1.0
    evalResult.jac .= eval_grad_dispatch(w)
    return 0
end
cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

println(">>> Running production outer-loop, direction=lower (find_smallest=$FIND_SMALLEST_PROD), gradient=lfix_composite_fast, hessopt=sr1, maxtime_real=$(MAXTIME_REAL)s, warm-started from refined near-crossing point (g=$G_TARGET)")
flush(stdout)
t0 = time()
open(joinpath(OUTDIR, "knitro.log"), "w") do io
    redirect_stdout(io) do
        KNITRO.KN_solve(kc)
    end
end
wall = time() - t0
nStatus, objv, w_min, lambda_ = KNITRO.KN_get_solution(kc)
opt_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_opt_error(kc, opt_err)
feas_err = Ref{Cdouble}(0.0); KNITRO.KN_get_abs_feas_error(kc, feas_err)
outer_iters = Ref{Cint}(0); KNITRO.KN_get_number_iters(kc, outer_iters)
KNITRO.KN_free(kc)

println("KNITRO terminal: status=$nStatus  gamma'_focal=$(w_min[1])  opt_err=$(opt_err[])  feas_err=$(feas_err[])  outer_iters=$(outer_iters[])  wall=$(round(wall,digits=1))s")
println("total Delta(w) evaluations: ", n_eval[])

function full_recheck(w, label)
    xf = x_free_from_w_prod(w)
    r = evaluate_fullA(xf, ctx_prod; cache = nothing, warm = false)   # COLD re-solve
    κ = 1 - r.gamma_focal_prime^(ctx_prod.σ / (ctx_prod.σ - 1))
    println("[$label] gamma'_focal=$(r.gamma_focal_prime)  kappa=$κ  Delta_dual=$(r.Delta_dual)  Delta-delta=$(r.Delta_minus_delta)")
    println("         gravity_value=$(r.gravity_value)  max_abs_moment_kkt_resid=$(r.max_abs_moment_kkt_resid)  mean_m_resid=$(r.mean_m_resid)  inner_status=$(r.inner_status)")
    println("         w = ", w)
    return (label = label, gamma_focal_prime = r.gamma_focal_prime, kappa = κ, Delta_dual = r.Delta_dual,
            Delta_minus_delta = r.Delta_minus_delta, gravity_value = r.gravity_value,
            max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid, mean_m_resid = r.mean_m_resid,
            inner_status = r.inner_status, w = collect(w))
end

println("\n" * "="^78); println("EXACT FRESH FEASIBILITY RECHECK"); println("="^78)
terminal_check = full_recheck(w_min, "raw_terminal")
best_check = best_feasible[] === nothing ? nothing : full_recheck(best_feasible[].w, "best_feasible_tracked")

const KAPPA_INCUMBENT = 0.005428799948779983
const KAPPA_PROFILE_ESTIMATE = 1 - 0.997031^(ctx_prod.σ / (ctx_prod.σ - 1))
println("\nkappa_registered_incumbent = ", KAPPA_INCUMBENT)
println("kappa_profile_estimate (from g~=0.997031, not a real candidate) = ", KAPPA_PROFILE_ESTIMATE)
if best_check !== nothing
    println("kappa_lower_v2 (best_feasible_tracked, cold-rechecked) = ", best_check.kappa)
    beat = best_check.kappa < KAPPA_INCUMBENT
    println(beat ? "lower_v2 BEATS the registered incumbent" : "lower_v2 does NOT beat the registered incumbent")
    println("  improvement: ", (KAPPA_INCUMBENT - best_check.kappa) / KAPPA_INCUMBENT * 100, "%")
end

open(joinpath(OUTDIR, "summary.txt"), "w") do io
    println(io, "direction=lower find_smallest=$FIND_SMALLEST_PROD gradient_method=lfix_composite_fast hessopt_tag=sr1 maxtime_real=$MAXTIME_REAL")
    println(io, "warm_start_g=$G_TARGET warm_start_source=$best_kind warm_start_Delta_at_refine=$Delta_refined")
    println(io, "knitro_status=$nStatus opt_err=$(opt_err[]) feas_err=$(feas_err[]) outer_iters=$(outer_iters[]) wall_seconds=$wall n_eval=$(n_eval[])")
    println(io, "kappa_registered_incumbent=$KAPPA_INCUMBENT")
    println(io, "kappa_profile_estimate=$KAPPA_PROFILE_ESTIMATE")
    println(io, "terminal: ", terminal_check)
    println(io, "best_feasible: ", best_check)
end
open(joinpath(OUTDIR, "callback_trace.csv"), "w") do io
    println(io, "idx,kind,gamma_focal_prime,Delta,feasible,inner_status,gravity_value")
    for r in callback_log
        println(io, r.idx, ",", r.kind, ",", r.gp, ",", r.Delta, ",", r.feasible, ",", r.inner_status, ",", r.gravity_value)
    end
end
println("\nWrote ", OUTDIR)
println("\n*** COPY THIS w VECTOR INTO c8_finalverify_battery.jl AS w_lower_v2 (best_feasible_tracked, cold-rechecked): ***")
println(best_check === nothing ? "NO FEASIBLE POINT FOUND" : best_check.w)
