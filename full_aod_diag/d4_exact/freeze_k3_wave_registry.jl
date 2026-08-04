# freeze_k3_wave_registry.jl -- freezes an immutable, checksummed CSV snapshot of the K=3
# origin_zc/cm_meanzc upper/lower results at a given wave-boundary delta (task section 10/11:
# K3_WAVE1_FINAL_SEED_REGISTRY.csv at delta=0.5, K3_COMPLETE_RESULT_REGISTRY.csv covering the full
# grid after delta=2). Wave 2 may not begin until the wave-1 snapshot this produces is on disk
# (task section 10's own explicit barrier requirement).
#
# Usage: julia --project=. freeze_k3_wave_registry.jl <output_csv_path> <delta1> [delta2 ...]
# Writes one row per (family in {origin_zc,cm_meanzc}) x (direction in {upper,lower}) x each given
# delta, reading each cell's already-finalized report.jls from the POST_VERIFY_FIX campaign_output
# tree. Errors loudly (not silently) if any required report is missing -- the whole point of a
# wave barrier is to refuse to proceed on an incomplete snapshot.
using SHA, Serialization, Dates

const D4E = @__DIR__
isdefined(Main, :OrchestratorRunState) || include(joinpath(D4E, "continuation_polish_orchestrator.jl"))

const CAMPAIGN_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/campaign_output"

function report_path(family, direction, delta)
    joinpath(CAMPAIGN_ROOT, family, direction, "delta_$(delta)", "report.jls")
end

function freeze(out_csv::String, deltas::Vector{Float64})
    rows = NamedTuple[]
    missing_paths = String[]
    for family in ("origin_zc", "cm_meanzc"), direction in ("upper", "lower")
        for delta in deltas
            p = report_path(family, direction, delta)
            if !isfile(p)
                push!(missing_paths, p)
                continue
            end
            st = load_run_state(p)
            r = st.report
            checksum = bytes2hex(sha256(join(string.(r.final_w), ",")))
            push!(rows, (
                family = family, direction = direction, delta = delta,
                GT = r.final_GT, Delta_star = r.final_Delta_star,
                knitro_status = r.knitro_status, result_source = r.result_source,
                algorithm = r.algorithm, n_eval = r.n_eval, wall_s = round(r.wall_s, digits = 1),
                w_checksum_sha256 = checksum, source_report_path = p,
            ))
        end
    end
    if !isempty(missing_paths)
        error("freeze_k3_wave_registry: $(length(missing_paths)) required report(s) missing -- refusing to freeze an incomplete wave snapshot:\n" *
              join(missing_paths, "\n"))
    end
    isdir(dirname(out_csv)) || mkpath(dirname(out_csv))
    header = ["family", "direction", "delta", "GT", "Delta_star", "knitro_status", "result_source",
              "algorithm", "n_eval", "wall_s", "w_checksum_sha256", "source_report_path"]
    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join([getfield(r, Symbol(h)) for h in header], ","))
        end
    end
    chmod(out_csv, 0o444)
    println("Froze ", length(rows), " rows to ", out_csv, " (chmod 444, immutable) at ", Dates.now())
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    out_csv = ARGS[1]
    deltas = parse.(Float64, ARGS[2:end])
    freeze(out_csv, deltas)
end
