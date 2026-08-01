# ============================================================================
# Claude Code task 2026-08-01, §15: destination-scale step decomposition for
# the FULL formulation's own outer trajectory. Reads the `:iteration`/
# `:new_best` checkpoints `run_profile_checkpointed` already writes (task
# §14's full arm run), decomposes each accepted step's Delta-log-A into
#   (Delta a_{.,d} - mean_d(Delta a_{.,d}) * 1) + mean_d(Delta a_{.,d}) * 1
# (per-destination COMMON-SCALE component vs RELATIVE component) and reports
# the fraction of step norm / fraction of accepted-move count each accounts
# for. Reuses `D20CheckpointV4`'s own `logA_full` field (already the full
# D x Ddest log-A matrix, gravity-pivot-EXPANDED) -- no new decode logic.
# ============================================================================
include(joinpath(@__DIR__, "c10_d20_production_driver.jl"))
using CSV, DataFrames, Printf, Glob

function load_iteration_checkpoints(ckpt_dir::AbstractString, label::AbstractString)
    files = sort(glob("$(label)_iteration_neval*.jls", ckpt_dir))
    isempty(files) && (files = sort(glob("$(label)_*_neval*.jls", ckpt_dir)))  # fallback: any reason
    ckpts = [load_checkpoint(f) for f in files]
    sort!(ckpts, by = c -> c.n_eval)
    return ckpts
end

function decompose(ckpt_dir::AbstractString, label::AbstractString)
    ckpts = load_iteration_checkpoints(ckpt_dir, label)
    length(ckpts) < 2 && return (n_steps = length(ckpts), rows = NamedTuple[])
    rows = NamedTuple[]
    for i in 2:length(ckpts)
        c0 = ckpts[i-1]; c1 = ckpts[i]
        D = size(c1.logA_full, 1); Ddest = size(c1.logA_full, 2)
        Δa = c1.logA_full .- c0.logA_full   # D x Ddest
        mean_d = vec(sum(Δa, dims = 1)) ./ D   # length Ddest: mean shift per destination
        scale_part = Δa .- 0.0  # placeholder, computed below column-wise
        scale_norm2 = 0.0; relative_norm2 = 0.0
        for d in 1:Ddest
            col = @view Δa[:, d]
            scale_col = fill(mean_d[d], D)
            rel_col = col .- mean_d[d]
            scale_norm2 += sum(abs2, scale_col)
            relative_norm2 += sum(abs2, rel_col)
        end
        total_norm2 = scale_norm2 + relative_norm2
        frac_scale = total_norm2 > 0 ? scale_norm2 / total_norm2 : NaN
        push!(rows, (n_eval_from = c0.n_eval, n_eval_to = c1.n_eval, step_norm = sqrt(total_norm2),
            frac_step_norm_in_scale_directions = frac_scale, frac_step_norm_in_relative_directions = 1 - frac_scale,
            Delta_dual_from = c0.verify_Delta_dual, Delta_dual_to = c1.verify_Delta_dual))
    end
    return (n_steps = length(rows), rows = rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ckpt_dir = joinpath(D4X_ROOT, "results", "profiled_ab_2026-08-01", "full_upper")
    res = decompose(ckpt_dir, "full_upper")
    println("n_steps analyzed: ", res.n_steps)
    if res.n_steps > 0
        df = DataFrame(res.rows)
        outpath = joinpath(D4X_ROOT, "DESTINATION_SCALE_STEP_DECOMPOSITION_2026-08-01.csv")
        CSV.write(outpath, df)
        println("Wrote $outpath")
        println(@sprintf("mean frac_step_norm_in_scale_directions = %.4f", sum(r.frac_step_norm_in_scale_directions for r in res.rows) / length(res.rows)))
    else
        println("No multi-step trajectory found (need >=2 :iteration checkpoints) -- run the full A/B first.")
    end
end
