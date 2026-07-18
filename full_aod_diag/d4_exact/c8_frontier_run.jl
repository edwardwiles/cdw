# ============================================================================
# Continuation 8, workstream B: algorithm frontier RERUN post-compressed-live-
# integration + post-winner-accelerator-wiring. Follows run_phase4_frontier.jl's
# exact subprocess-per-config-per-budget pattern (reused, not rebuilt) --
# orchestrates run_d4_optimized_fd_c8.jl (this workstream's own driver copy)
# as a separate Julia subprocess per (budget, config) pair, so each gets a
# clean KNITRO/obj/module state, then parses each run's summary.txt into one
# comparison table.
#
# Run: julia --project=. full_aod_diag/d4_exact/c8_frontier_run.jl [budgets_csv] [direction]
#   e.g. julia c8_frontier_run.jl 15,30,60,120,300 upper
#        julia c8_frontier_run.jl 30,120,300 lower
# ============================================================================
using Dates

const HERE = @__DIR__
const BUDGETS = length(ARGS) >= 1 ? parse.(Float64, split(ARGS[1], ",")) : [15.0, 30.0, 60.0, 120.0, 300.0]
const DIRECTION = length(ARGS) >= 2 ? ARGS[2] : "upper"
const COMMIT = try strip(read(`git rev-parse --short HEAD`, String)) catch; "uncommitted" end
const OUT_CSV = joinpath(HERE, "..", "..", "results", "fullA_d4", COMMIT,
                          "c8_frontier_$(DIRECTION)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv")
mkpath(dirname(OUT_CSV))

# 5 configs, exactly matching the standing continuation-8 brief's Section 6 list:
#   1. dense hard L_fix + SR1
#   2. compressed hard L_fix + SR1 (NEW)
#   3. compressed hard L_fix + L-BFGS (NEW)
#   4. optimized-value FD + L-BFGS (expensive reference control, unchanged/dense)
#   5. consistent smoothed AD + SR1 (this codebase's own smoothed_consistent.jl
#      "Method 1" ForwardDiff-of-fixed-dual-envelope construction, reused, at a
#      single fixed rho picked per-run via the coarsest-empirically-feasible-at-w0
#      scan -- see run_d4_optimized_fd_c8.jl's own header for the exact
#      simplification from the original 5-stage homotopy)
const CONFIGS = [
    (label = "lfixcomposite_sr1_dense",       grad = "lfix_composite", hess = "sr1",   rep = "dense"),
    (label = "lfixcomposite_sr1_compressed",  grad = "lfix_composite", hess = "sr1",   rep = "compressed"),
    (label = "lfixcomposite_lbfgs_compressed",grad = "lfix_composite", hess = "lbfgs", rep = "compressed"),
    (label = "deltafd_lbfgs_dense",           grad = "delta_fd",       hess = "lbfgs", rep = "dense"),
    (label = "smoothed_ad_sr1",               grad = "smoothed_ad",    hess = "sr1",   rep = "dense"),
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

"""
Extract the best-feasible kappa (a Float64, or NaN) from summary.txt's
`best_feasible: (label = ..., ..., kappa = <float>, ...)` printed-NamedTuple
line -- deliberately a narrow regex on ` kappa = ` within that ONE line only
(not the whole file, which also contains `terminal:`'s own kappa field), per
this investigation's own established caution about not silently mis-extracting
a NamedTuple field from printed Julia text (run_phase4_frontier.jl's own
parse_summary docstring flags the same class of risk).
"""
function extract_best_kappa(path)
    isfile(path) || return NaN
    for line in eachline(path)
        startswith(line, "best_feasible: ") || continue
        occursin("nothing", line) && return NaN
        m = match(r"kappa\s*=\s*(-?[0-9.eE+-]+)", line)
        m === nothing && return NaN
        return parse(Float64, m.captures[1])
    end
    return NaN
end

function extract_field(path, prefix_line_startswith, key)
    isfile(path) || return "?"
    for line in eachline(path)
        startswith(line, prefix_line_startswith) || continue
        m = match(Regex(key * "\\s*=\\s*([^,)]+)"), line)
        m === nothing && return "?"
        return strip(m.captures[1])
    end
    return "?"
end

results = NamedTuple[]
for budget in BUDGETS, cfg in CONFIGS
    println("="^78); println("RUNNING: budget=$(budget)s  config=$(cfg.label)  direction=$DIRECTION"); println("="^78); flush(stdout)
    env = copy(ENV)
    env["D4X_GRADIENT_METHOD"] = cfg.grad
    env["D4X_HESSOPT"] = cfg.hess
    env["D4X_MOMENT_REP"] = cfg.rep
    env["D4X_MAXTIME_REAL"] = string(budget)
    t0 = time()
    logdir = joinpath(HERE, "..", "..", "results")
    mkpath(logdir)
    stdout_log = joinpath(logdir, "c8_frontier_stdout_$(DIRECTION)_$(cfg.label)_$(budget).log")
    stderr_log = joinpath(logdir, "c8_frontier_stderr_$(DIRECTION)_$(cfg.label)_$(budget).log")
    proc = run(pipeline(setenv(`julia --project=. $(joinpath(HERE, "run_d4_optimized_fd_c8.jl")) $DIRECTION`, env),
                        stdout = stdout_log, stderr = stderr_log), wait = false)
    wait(proc)
    wall_wrapper = time() - t0
    resdir_root = joinpath(HERE, "..", "..", "results", "fullA_d4", COMMIT)
    cands = isdir(resdir_root) ? filter(d -> occursin("optfdc8_$(DIRECTION)_$(cfg.grad)_$(cfg.hess)_$(cfg.rep)_", d), readdir(resdir_root)) : String[]
    if isempty(cands)
        println("  WARNING: no results dir found for $(cfg.label)@$(budget)s -- process may have errored (exitcode=$(proc.exitcode)), see $stdout_log / $stderr_log")
        push!(results, (budget = budget, label = cfg.label, grad = cfg.grad, hess = cfg.hess, rep = cfg.rep,
                         resdir = "MISSING", wall_wrapper = round(wall_wrapper, digits = 1), exitcode = proc.exitcode,
                         knitro_status = "?", outer_iters = "?", n_eval = "?", gradient_calls = "?", n_cheap = "?",
                         total_inner_solves_in_gradients = "?", compressed_fallback_count = "?",
                         best_kappa = NaN, time_to_first_feasible_s = NaN))
        continue
    end
    latest = maximum(cands)
    summpath = joinpath(resdir_root, latest, "summary.txt")
    summ = parse_summary(summpath)
    best_kappa = extract_best_kappa(summpath)

    # time-to-first-feasible: earliest callback_trace.csv row with feasible=true, converted from
    # eval-index to an approximate wall-clock fraction of this run's own measured wall_seconds
    # (the trace itself is eval-indexed, not individually timestamped -- documented limitation,
    # not silently assumed exact).
    ttf = NaN
    tracepath = joinpath(resdir_root, latest, "callback_trace.csv")
    if isfile(tracepath) && summ !== nothing && haskey(summ, "wall_seconds") && haskey(summ, "n_eval")
        lines = readlines(tracepath)
        n_eval_total = something(tryparse(Int, get(summ, "n_eval", "")), 0)
        wall_seconds = something(tryparse(Float64, get(summ, "wall_seconds", "")), NaN)
        for (i, line) in enumerate(lines[2:end])
            parts = split(line, ",")
            length(parts) >= 5 && parts[4] == "true" && n_eval_total > 0 && begin
                idx = something(tryparse(Int, parts[1]), 0)
                ttf = wall_seconds * idx / n_eval_total
                break
            end
        end
    end

    push!(results, (budget = budget, label = cfg.label, grad = cfg.grad, hess = cfg.hess, rep = cfg.rep,
                     resdir = latest, wall_wrapper = round(wall_wrapper, digits = 1), exitcode = proc.exitcode,
                     knitro_status = get(summ, "knitro_status", "?"),
                     outer_iters = get(summ, "outer_iters", "?"),
                     n_eval = get(summ, "n_eval", "?"),
                     gradient_calls = get(summ, "gradient_calls", "?"),
                     n_cheap = get(summ, "n_cheap", "?"),
                     total_inner_solves_in_gradients = get(summ, "total_inner_solves_in_gradients", "?"),
                     compressed_fallback_count = get(summ, "compressed_fallback_count", "?"),
                     best_kappa = best_kappa, time_to_first_feasible_s = round(ttf, digits = 2)))
    println("  -> best_kappa=$best_kappa  knitro_status=$(get(summ, "knitro_status", "?"))  outer_iters=$(get(summ, "outer_iters", "?"))")
end

open(OUT_CSV, "w") do io
    println(io, "budget,label,grad,hess,rep,resdir,wall_wrapper,exitcode,knitro_status,outer_iters,n_eval,gradient_calls,n_cheap,total_inner_solves_in_gradients,compressed_fallback_count,best_kappa,time_to_first_feasible_s")
    for r in results
        println(io, r.budget, ",", r.label, ",", r.grad, ",", r.hess, ",", r.rep, ",", r.resdir, ",", r.wall_wrapper, ",",
                    r.exitcode, ",", r.knitro_status, ",", r.outer_iters, ",", r.n_eval, ",", r.gradient_calls, ",",
                    r.n_cheap, ",", r.total_inner_solves_in_gradients, ",", r.compressed_fallback_count, ",",
                    r.best_kappa, ",", r.time_to_first_feasible_s)
    end
end
println("\nWrote ", OUT_CSV)
