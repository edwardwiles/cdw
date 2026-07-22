# Diagnostic, round 5 (D=4 minimal reproduction, per user's suggestion -- D=20's ~20-85s context
# build per attempt makes iterating on a threading hypothesis needlessly slow, and the deadlock
# (a nested KN_solve inside another KN_solve's callback, stuck in libiomp5's OpenMP queuing lock)
# should not depend on problem dimension at all if the hypothesis is right).
#
# Minimal, self-contained: a bare outer KN_solve (1 variable, trivial objective/gradient) whose
# CALLBACK invokes ONE real screened_eval (-> evaluate_fullA_screened_ranged ->
# inner_loop_KNITRO_compressed -> a SECOND, nested KN_solve) at a real D=4 point -- the exact
# same nested-solve shape that hung 3 times at D=20, stripped of everything else
# (checkpointing, direction boxes, dual banks, screens beyond the minimum needed).
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))   # ScreenCounters, screened_eval, build_ranged_screen_context, build_pivot_elimination -- all generic over ctx, not D=20-specific themselves (only run_polish_checkpointed/run_profile_checkpointed hardcode d20_real_setup_design internally; we call screened_eval directly, never those). Pulls in context.jl (-> d4_exact_setup) transitively via draw_design.jl -> context_real_d20.jl -> context.jl -- no separate include needed.
using KNITRO

println("== D=4 minimal nested-KN_solve reproduction ==")
println("Threads.nthreads()=", Threads.nthreads(), " maxthreadid()=", Threads.maxthreadid())
flush(stdout)

t0 = time()
ctx = d4_exact_setup(find_smallest = true)
pe = build_pivot_elimination(ctx)
rsc = build_ranged_screen_context(ctx)
D = ctx.D; D2 = D^2
println(">>> D=4 ctx built in ", round(time() - t0, digits = 2), "s")
flush(stdout)

x_free_from_w(w) = vcat(w[1], vec(exp.(pivot_expand(w[2:end], pe))))
Aod_theta_natural = ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2]
zfree0 = pivot_reduce(reshape(log.(Aod_theta_natural), D, D), pe)
gp0 = ctx.θ0_up[3+D]
w0 = vcat(gp0, zfree0)
obj = ctx.obj

θ_full0 = CS.reconstruct_full(x_free_from_w(w0), ctx.m)
obj.x .= NaN   # match evaluate_fullA_fast's own !warm reset before a standalone probe
K_std, inner_x_std, nStatus_std, _, _ = inner_loop_internal_profiled(obj, θ_full0)
println(">>> single standalone inner solve (no nesting) BEFORE outer solve: nStatus=", nStatus_std)
flush(stdout)

n_cb_calls = Ref(0)
function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
    n_cb_calls[] += 1
    println(">>> cb_F! call #", n_cb_calls[], " starting NESTED inner_loop_internal_profiled (this is where D=20 hung)...")
    flush(stdout)
    w = evalRequest.x
    θ_full = CS.reconstruct_full(x_free_from_w(w), ctx.m)
    obj.x .= NaN
    K, inner_x, nStatus, _, _ = inner_loop_internal_profiled(obj, θ_full)
    println(">>> cb_F! call #", n_cb_calls[], " NESTED inner solve RETURNED: nStatus=", nStatus)
    flush(stdout)
    evalResult.obj[1] = isfinite(K) ? K : 1e6
    evalResult.c[1] = 0.0
    return 0
end

kc = KNITRO.KN_new()
KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, "csw_outer_wallclock_sr1.opt"))
KNITRO.KN_set_param_by_name(kc, "maxtime_real", 15.0)
xIndices = KNITRO.KN_add_vars(kc, D2)
KNITRO.KN_set_var_lobnds_all(kc, w0 .- 1.0)
KNITRO.KN_set_var_upbnds_all(kc, w0 .+ 1.0)
KNITRO.KN_set_var_primal_init_values_all(kc, w0)
cIndices = KNITRO.KN_add_cons(kc, 1)
KNITRO.KN_set_con_upbnd(kc, cIndices[1], 100.0)   # slack constraint, never binding
cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)

println(">>> starting OUTER KN_solve (maxtime_real=15s)...")
flush(stdout)
t1 = time()
KNITRO.KN_solve(kc)
println(">>> OUTER KN_solve RETURNED after ", round(time() - t1, digits = 2), "s, n_cb_calls=", n_cb_calls[])
nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)
println("DONE. nStatus=", nStatus, " objSol=", objSol)
KNITRO.KN_free(kc)
