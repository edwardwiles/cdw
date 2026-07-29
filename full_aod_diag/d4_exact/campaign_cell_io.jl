# campaign_cell_io.jl -- per-cell status/marker/resume helpers for the five-family overnight
# campaign (2026-07-28, maxtime_real=3600s cap).
#
# The 2026-07-28 shakedown (30s cap, ~90 min total campaign wall time) accumulated all outer/inner
# rows in memory and wrote the family-level aggregate CSVs only once, at the very end of each
# 25-cell family/direction run. That was an acceptable risk at a ~90-minute total campaign wall
# time. It is not acceptable once each cell can run up to 3600s (a single family/direction chain of
# 25 cells can now take up to 25 hours): a mid-chain crash would silently lose every completed
# cell's CSV row, and a restarted process would blindly re-run (and, worse, `rm -rf` and destroy
# the checkpoint of) cells that already completed.
#
# This file adds exactly the two things that gap requires -- nothing else:
#   1. a DONE/FAILED marker + a small JSON status file per cell, written immediately after that
#      cell finishes (so a killed process's completed cells survive and are visible to a
#      supervisor), and read back (not re-solved) if the runner restarts and finds a DONE cell;
#   2. append-only (not rewrite-at-the-end) aggregate CSV writers.

using Dates

cell_done_marker(ckdir) = joinpath(ckdir, "DONE")
cell_failed_marker(ckdir) = joinpath(ckdir, "FAILED")
cell_attempt_file(ckdir) = joinpath(ckdir, "attempt_count.txt")

cell_already_done(ckdir) = isfile(cell_done_marker(ckdir))

function cell_attempt_count(ckdir)
    isfile(cell_attempt_file(ckdir)) || return 0
    return parse(Int, strip(read(cell_attempt_file(ckdir), String)))
end

function bump_cell_attempt!(ckdir)
    n = cell_attempt_count(ckdir) + 1
    mkpath(ckdir)
    open(cell_attempt_file(ckdir), "w") do io
        print(io, n)
    end
    return n
end

# ---- minimal JSON writer (status records only: flat/nested Dicts of Number/String/Bool/Nothing/
# Vector/Dict -- not a general-purpose serializer, mirrors json_lite.jl's read-side scope note) ----
function json_write_value(io::IO, v::AbstractString)
    print(io, '"', replace(v, "\\" => "\\\\", "\"" => "\\\""), '"')
end
json_write_value(io::IO, v::Bool) = print(io, v ? "true" : "false")
json_write_value(io::IO, ::Nothing) = print(io, "null")
function json_write_value(io::IO, v::Real)
    print(io, isfinite(v) ? v : (isnan(v) ? "null" : (v > 0 ? "1e308" : "-1e308")))
end
function json_write_value(io::IO, v::AbstractVector)
    print(io, "[")
    for (i, x) in enumerate(v)
        i > 1 && print(io, ",")
        json_write_value(io, x)
    end
    print(io, "]")
end
function json_write_value(io::IO, v::AbstractDict)
    print(io, "{")
    for (i, (k, x)) in enumerate(v)
        i > 1 && print(io, ",")
        json_write_value(io, string(k))
        print(io, ":")
        json_write_value(io, x)
    end
    print(io, "}")
end
json_write_value(io::IO, v::Missing) = print(io, "null")

function write_json_file(path::AbstractString, d)
    open(path, "w") do io
        json_write_value(io, d)
    end
end

namedtuple_to_dict(nt::NamedTuple) = Dict{String,Any}(string(k) => (v isa NamedTuple ? namedtuple_to_dict(v) : v) for (k, v) in pairs(nt))

"Write config.json, outer_status.json, best_verified_incumbent.json (if any), and the DONE/FAILED
marker for one cell. Called immediately after that cell finishes -- not batched at family end."
function write_cell_status!(ckdir::AbstractString, outer_row::NamedTuple, inner_rows::Vector{<:NamedTuple},
                             best_incumbent, cfg::AbstractDict; failed::Bool = false)
    mkpath(ckdir)
    write_json_file(joinpath(ckdir, "config.json"), cfg)
    write_json_file(joinpath(ckdir, "outer_status.json"), namedtuple_to_dict(outer_row))
    if best_incumbent !== nothing
        write_json_file(joinpath(ckdir, "best_verified_incumbent.json"), namedtuple_to_dict(best_incumbent))
    end
    if !isempty(inner_rows)
        open(joinpath(ckdir, "inner_status_summary.csv"), "w") do io
            ks = string.(keys(inner_rows[1]))
            println(io, join(ks, ","))
            for r in inner_rows
                println(io, join(string.(values(r)), ","))
            end
        end
    end
    rm(cell_failed_marker(ckdir); force = true)
    if failed
        open(cell_failed_marker(ckdir), "w") do io
            println(io, Dates.now())
        end
    else
        open(cell_done_marker(ckdir), "w") do io
            println(io, Dates.now())
        end
    end
end

"Append one row to a family-level aggregate CSV, writing the header only if the file is new.
Opened and closed per call (not held open across the whole family run) so a killed process leaves
a syntactically valid, complete-through-the-last-cell CSV rather than a truncated one."
function append_csv_row!(csv_path::AbstractString, header::AbstractString, row_line::AbstractString)
    is_new = !isfile(csv_path)
    open(csv_path, "a") do io
        is_new && println(io, header)
        println(io, row_line)
    end
end

"Read back a previously-completed cell's outer row (as a plain Dict) from its outer_status.json,
for the case where a restarted runner finds a DONE cell and must fold its row into the in-memory
summary without re-solving it."
function read_cell_outer_status(ckdir::AbstractString)
    path = joinpath(ckdir, "outer_status.json")
    isfile(path) || return nothing
    return json_load(path)
end
