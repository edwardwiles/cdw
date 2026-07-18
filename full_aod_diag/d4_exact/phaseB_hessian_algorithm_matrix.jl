# ============================================================================
# Continuation-session Phase B: controlled KNITRO Hessian-mode / algorithm
# comparison. Per the continuation prompt's correction #1: hessopt=4 is
# PRODUCT FINITE-DIFFERENCE Hessian-vector, not BFGS (hessopt=2=dense BFGS,
# hessopt=3=dense SR1, hessopt=6=L-BFGS -- verified directly from this
# repo's own csw_outer_25.opt comment block, not assumed). Every prior run in
# this investigation's history that used eval_fcga=no (the "restores BFGS"
# claim in docs/fullA_d4_final_report.md sec 6/7) actually ran hessopt=4
# (product-findiff), honored (not silently downgraded) once eval_fcga=no --
# never genuine BFGS.
#
# Same problem, same start (w0, the calibration-anchored initial point --
# NOT the maxit40 candidate, since the point of this comparison is which
# Hessian mode gets FURTHEST from a common start, not a re-check of an
# already-found point), same derivative method (central FD, h=0.01), same
# bounds, same maxit=15, same opttol, across:
#   auto_bfgs        : algorithm=auto, hessopt=2
#   auto_sr1         : algorithm=auto, hessopt=3
#   auto_lbfgs       : algorithm=auto, hessopt=6
#   auto_productfd   : algorithm=auto, hessopt=4  (CONTROL -- reuses the
#                        already-completed results/fullA_d4/9e03706/
#                        optfd_upper_20260717_182444 run: identical opt file)
#   sqp_bfgs         : algorithm=sqp (active-set/SQP), hessopt=2
#   direct_bfgs      : algorithm=direct (barrier/interior), hessopt=2
# ============================================================================
include(joinpath(@__DIR__, "context.jl"))
include(joinpath(@__DIR__, "winners.jl"))
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, "gravity_elimination.jl"))
using KNITRO, Printf, Dates

const COMMIT = strip(read(`git rev-parse --short HEAD`, String))
const OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT, "phaseB_hessian_matrix")
mkpath(OUTDIR)

const CONFIGS = [
    ("auto_bfgs",      "csw_outer_phaseB_auto_bfgs_maxit15.opt"),
    ("auto_sr1",       "csw_outer_phaseB_auto_sr1_maxit15.opt"),
    ("auto_lbfgs",     "csw_outer_phaseB_auto_lbfgs_maxit15.opt"),
    ("sqp_bfgs",       "csw_outer_phaseB_sqp_bfgs_maxit15.opt"),
    ("direct_bfgs",    "csw_outer_phaseB_direct_bfgs_maxit15.opt"),
]
const FIXED_H = 0.01
const FIND_SMALLEST = true   # upper direction, matching the existing baseline runs

function run_one_config(cfg_name::String, opt_file::String)
    println("="^78); println("CONFIG: $cfg_name  (opt file: $opt_file)"); println("="^78); flush(stdout)
    ctx = d4_exact_setup(find_smallest = FIND_SMALLEST, outer_loop_opt = joinpath(@__DIR__, opt_file))
    pe = build_pivot_elimination(ctx)
    D = ctx.D; D2 = D^2

    Aod_theta0 = reshape(ctx.θ0_up[ctx.Aod_offset+1:ctx.Aod_offset+D2], D, D)
    z0 = log.(Aod_theta0)
    zfree0 = pivot_reduce(z0, pe)
    gp0 = ctx.θ0_up[3+D]
    w0 = vcat(gp0, zfree0)

    x_free_from_w(w) = (z = pivot_expand(w[2:end], pe); vcat(w[1], vec(exp.(z))))
    function Delta_of_w(w)
        r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = true)
        return r.Delta_dual, r
    end

    w_lo = vcat(ctx.bounds.γp_lo, fill(-8.0, D2 - 1))
    w_hi = vcat(ctx.bounds.γp_hi, fill(8.0, D2 - 1))

    n_eval = Ref(0); n_grad_calls = Ref(0); n_inner_solves = Ref(0)
    best_feasible = Ref{Union{Nothing,NamedTuple}}(nothing)
    function record!(w, Δ, feasible)
        n_eval[] += 1
        if feasible && (best_feasible[] === nothing || w[1] < best_feasible[].gp)
            best_feasible[] = (gp = w[1], w = copy(w), Delta = Δ)
        end
    end
    function eval_F(w)
        Δ, r = Delta_of_w(w); n_inner_solves[] += 1
        feasible = isfinite(Δ) && Δ <= ctx.δ + 1e-6
        record!(w, Δ, feasible)
        return Δ, r
    end
    function eval_grad_central_fd(w, h = FIXED_H)
        n_grad_calls[] += 1
        n = length(w); g = zeros(n)
        for i in 1:n
            wp = copy(w); wp[i] += h; wm = copy(w); wm[i] -= h
            Δp, _ = Delta_of_w(wp); n_inner_solves[] += 1
            Δm, _ = Delta_of_w(wm); n_inner_solves[] += 1
            if isfinite(Δp) && isfinite(Δm)
                g[i] = (Δp - Δm) / (2h)
            elseif isfinite(Δp)
                Δ0, _ = Delta_of_w(w); n_inner_solves[] += 1
                g[i] = (Δp - Δ0) / h
            elseif isfinite(Δm)
                Δ0, _ = Delta_of_w(w); n_inner_solves[] += 1
                g[i] = (Δ0 - Δm) / h
            else
                g[i] = 0.0
            end
        end
        return g
    end

    kc = KNITRO.KN_new()
    KNITRO.KN_load_param_file(kc, joinpath(@__DIR__, opt_file))
    xIndices = KNITRO.KN_add_vars(kc, D2)
    KNITRO.KN_set_var_lobnds_all(kc, w_lo)
    KNITRO.KN_set_var_upbnds_all(kc, w_hi)
    KNITRO.KN_set_var_primal_init_values_all(kc, w0)
    cIndices = KNITRO.KN_add_cons(kc, 1)
    KNITRO.KN_set_con_upbnd(kc, cIndices[1], ctx.δ)

    function cb_F!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        Δ, r = eval_F(w)
        evalResult.obj[1] = FIND_SMALLEST ? w[1] : -w[1]
        evalResult.c[1] = isfinite(Δ) ? Δ : 1e6
        return 0
    end
    function cb_G!(kc2, cb, evalRequest, evalResult, userParams)
        w = evalRequest.x
        evalResult.objGrad .= 0.0; evalResult.objGrad[1] = FIND_SMALLEST ? 1.0 : -1.0
        evalResult.jac .= eval_grad_central_fd(w)
        return 0
    end
    cb = KNITRO.KN_add_eval_callback(kc, true, cIndices, cb_F!)
    KNITRO.KN_set_cb_grad(kc, cb, cb_G!, jacIndexCons = fill(cIndices[1], D2), jacIndexVars = xIndices)

    t0 = time()
    logpath = joinpath(OUTDIR, "$(cfg_name)_knitro.log")
    open(logpath, "w") do io
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

    logtext = read(logpath, String)
    fallback_lines = filter(l -> occursin("hessopt", lowercase(l)) || occursin("changing", lowercase(l)) ||
                                  occursin("not valid", lowercase(l)) || occursin("algorithm", lowercase(l)),
                             split(logtext, '\n'))
    algo_line = filter(l -> occursin("Knitro selects algorithm", l) || occursin("algorithm:", lowercase(l)), split(logtext, '\n'))

    # fresh cold recheck of both terminal and best-feasible
    function recheck(w, label)
        r = evaluate_fullA(x_free_from_w(w), ctx; cache = nothing, warm = false)
        κ = 1 - r.gamma_focal_prime^(ctx.σ / (ctx.σ - 1))
        return (label = label, gamma_focal_prime = r.gamma_focal_prime, kappa = κ,
                Delta_dual = r.Delta_dual, Delta_minus_delta = r.Delta_minus_delta,
                gravity_value = r.gravity_value, max_abs_moment_kkt_resid = r.max_abs_moment_kkt_resid,
                inner_status = r.inner_status)
    end
    terminal_check = recheck(w_min, "raw_terminal")
    best_check = best_feasible[] === nothing ? nothing : recheck(best_feasible[].w, "best_feasible")

    summary = (config = cfg_name, opt_file = opt_file, knitro_status = nStatus, opt_err = opt_err[],
               feas_err = feas_err[], outer_iters = outer_iters[], wall_seconds = wall,
               n_eval_calls = n_eval[], n_grad_calls = n_grad_calls[], n_inner_solves = n_inner_solves[],
               terminal_kappa = terminal_check.kappa, terminal_Delta_minus_delta = terminal_check.Delta_minus_delta,
               best_feasible_kappa = best_check === nothing ? NaN : best_check.kappa,
               best_feasible_Delta_minus_delta = best_check === nothing ? NaN : best_check.Delta_minus_delta,
               best_feasible_gravity = best_check === nothing ? NaN : best_check.gravity_value,
               best_feasible_kkt = best_check === nothing ? NaN : best_check.max_abs_moment_kkt_resid,
               fallback_detected = any(occursin("not valid", l) || occursin("Changing", l) for l in fallback_lines))

    println("RESULT [$cfg_name]: status=$nStatus opt_err=$(opt_err[]) outer_iters=$(outer_iters[]) wall=$(round(wall,digits=1))s n_inner_solves=$(n_inner_solves[])")
    println("  terminal kappa=$(terminal_check.kappa)  best_feasible kappa=$(summary.best_feasible_kappa)")
    println("  fallback/override lines in log: ", isempty(fallback_lines) ? "(none)" : fallback_lines)
    flush(stdout)

    open(joinpath(OUTDIR, "$(cfg_name)_summary.txt"), "w") do io
        println(io, summary)
        println(io, "terminal: ", terminal_check)
        println(io, "best_feasible: ", best_check)
        println(io, "fallback_lines: ", fallback_lines)
        println(io, "algo_lines: ", algo_line)
    end
    return summary
end

results = NamedTuple[]
for (cfg_name, opt_file) in CONFIGS
    push!(results, run_one_config(cfg_name, opt_file))
end

# ---- fold in the already-completed control (auto_productfd = existing csw_outer_fcga_no_maxit15.opt run) ----
control_summary_path = joinpath(D4X_ROOT, "results", "fullA_d4", "9e03706", "optfd_upper_20260717_182444", "summary.txt")
println("\n(auto_productfd CONTROL reuses existing run: $control_summary_path -- not re-executed, see docs/fullA_d4_resume_audit.md)")

open(joinpath(OUTDIR, "phaseB_comparison_table.csv"), "w") do io
    cols = keys(results[1])
    println(io, join(cols, ","))
    for r in results
        println(io, join((r[c] for c in cols), ","))
    end
end
println("\nWrote comparison table to ", joinpath(OUTDIR, "phaseB_comparison_table.csv"))
