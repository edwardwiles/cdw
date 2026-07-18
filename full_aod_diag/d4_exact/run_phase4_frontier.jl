# ============================================================================
# Continuation 4, Phase 4 (minimum bar): fair wall-clock-matched comparison
# between the historical optimized-value-FD + product-FD-Hessian control and
# the new composite-hybrid gradient (this session's Phase 3 deliverable),
# at a common starting point, common exact settings (upper direction, w0),
# common wall-clock budget(s). Orchestrates run_d4_optimized_fd.jl as
# separate subprocesses (one Julia process per config -- the driver script
# rebuilds ctx from scratch each time, which is cheap relative to the
# wall-clock budgets tested here) so each config gets a clean KNITRO/obj
# state, then parses each config's summary.txt into one comparison table.
#
# Run: julia run_phase4_frontier.jl [budget_seconds_csv]
#   e.g. julia run_phase4_frontier.jl 60        (single 60s budget)
#        julia run_phase4_frontier.jl 30,60,180 (three budgets)
# ============================================================================
using Dates

const HERE = @__DIR__
const BUDGETS = length(ARGS) >= 1 ? parse.(Float64, split(ARGS[1], ",")) : [60.0]
const DIRECTION = length(ARGS) >= 2 ? ARGS[2] : "upper"
const OUT_CSV = joinpath(HERE, "..", "..", "results", "fullA_d4", "9e03706",
                          "phase4_frontier_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
mkpath(dirname(OUT_CSV))

# Minimum bar (task's own words): "AT LEAST ONE wall-clock-matched comparison between the existing
# product-FD/optimized-value control and hybrid-L_fix+SR1-or-LBFGS". Configs 1/4/5 below are exactly
# that; 2/3 (SR1/LBFGS with the OLD expensive gradient) are included as free intermediate points since
# they cost nothing extra to run once the harness exists, per the task's full Phase 4 wishlist.
#
# IMPORTANT, found this session: `hybrid` (gradient SOURCE switching between the exact delta_fd
# refresh and the cheap composite gradient across outer iterates) terminates KNITRO prematurely
# (status -102 after only 2 outer iterations, vs the expected iteration-limit status) when paired
# with a quasi-Newton Hessian approximation (SR1/L-BFGS/BFGS) -- plausible root cause: those methods'
# secant/curvature updates assume a CONSISTENT gradient source across consecutive accepted steps, and
# switching sources mid-solve corrupts that assumption. `lfix_composite` (ALWAYS the cheap gradient,
# no source-switching) does not exhibit this and converges normally. `lfix_composite` is therefore
# used as the primary new-method comparator below; `hybrid` configs are included too so the
# instability is visible in the results table rather than hidden, not as the headline comparison.
const CONFIGS = [
    (label = "control_deltafd_productfd", grad = "delta_fd", hess = "productfd"),
    (label = "deltafd_sr1", grad = "delta_fd", hess = "sr1"),
    (label = "deltafd_lbfgs", grad = "delta_fd", hess = "lbfgs"),
    (label = "lfixcomposite_sr1", grad = "lfix_composite", hess = "sr1"),
    (label = "lfixcomposite_lbfgs", grad = "lfix_composite", hess = "lbfgs"),
    (label = "hybrid_sr1", grad = "hybrid", hess = "sr1"),
    (label = "hybrid_lbfgs", grad = "hybrid", hess = "lbfgs"),
]

function parse_summary(path)
    isfile(path) || return nothing
    txt = read(path, String)
    d = Dict{String,String}()
    for line in split(txt, '\n')
        for pair in split(line, ' ')
            kv = split(pair, '=')
            length(kv) == 2 && (d[kv[1]] = kv[2])
        end
    end
    return d
end

results = NamedTuple[]
for budget in BUDGETS, cfg in CONFIGS
    println("="^78); println("RUNNING: budget=$(budget)s  config=$(cfg.label)"); println("="^78); flush(stdout)
    env = copy(ENV)
    env["D4X_GRADIENT_METHOD"] = cfg.grad
    env["D4X_HESSOPT"] = cfg.hess
    env["D4X_MAXTIME_REAL"] = string(budget)
    t0 = time()
    proc = run(pipeline(setenv(`julia --project=. $(joinpath(HERE, "run_d4_optimized_fd.jl")) $DIRECTION`, env),
                        stdout = joinpath(HERE, "..", "..", "results", "phase4_stdout_$(cfg.label)_$(budget).log"),
                        stderr = joinpath(HERE, "..", "..", "results", "phase4_stderr_$(cfg.label)_$(budget).log")),
               wait = false)
    wait(proc)
    wall_wrapper = time() - t0
    # find the most recent matching results dir (run_d4_optimized_fd.jl's RUN_ID embeds grad/hess/timestamp)
    resdir_root = joinpath(HERE, "..", "..", "results", "fullA_d4", "9e03706")
    cands = filter(d -> occursin("optfd_$(DIRECTION)_$(cfg.grad)_$(cfg.hess)_", d), readdir(resdir_root))
    isempty(cands) && (println("  WARNING: no results dir found for $(cfg.label) -- process may have errored, see phase4_std{out,err}_$(cfg.label)_$(budget).log"); continue)
    latest = maximum(cands)
    summ = parse_summary(joinpath(resdir_root, latest, "summary.txt"))
    summ === nothing && continue
    push!(results, (budget = budget, label = cfg.label, grad = cfg.grad, hess = cfg.hess,
                     resdir = latest, wall_wrapper = wall_wrapper,
                     knitro_status = get(summ, "knitro_status", "?"),
                     outer_iters = get(summ, "outer_iters", "?"),
                     n_eval = get(summ, "n_eval", "?"),
                     gradient_calls = get(summ, "gradient_calls", "?"),
                     n_cheap = get(summ, "n_cheap", "?"),
                     total_inner_solves_in_gradients = get(summ, "total_inner_solves_in_gradients", "?")))
end

open(OUT_CSV, "w") do io
    println(io, "budget,label,grad,hess,resdir,wall_wrapper,knitro_status,outer_iters,n_eval,gradient_calls,n_cheap,total_inner_solves_in_gradients")
    for r in results
        println(io, r.budget, ",", r.label, ",", r.grad, ",", r.hess, ",", r.resdir, ",", round(r.wall_wrapper, digits = 1), ",",
                    r.knitro_status, ",", r.outer_iters, ",", r.n_eval, ",", r.gradient_calls, ",", r.n_cheap, ",", r.total_inner_solves_in_gradients)
    end
end
println("\nWrote ", OUT_CSV)
println("\nNOTE: this CSV has run METADATA only (iters/evals/status). The actual best-exact-feasible")
println("kappa per config must be read from each resdir's summary.txt 'best_feasible' line -- deliberately")
println("not re-parsed into a single float here to avoid silently mis-extracting a NamedTuple field from a")
println("printed Julia value (see this file's own parse_summary, which is line-oriented text, not a real parser).")
