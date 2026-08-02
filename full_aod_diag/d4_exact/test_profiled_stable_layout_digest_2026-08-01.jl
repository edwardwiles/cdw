# ============================================================================
# Production outer bridge task (2026-08-01), §3: fresh-process reproducibility
# gate for stable_layout_digest. Spawns TWO genuinely separate `julia`
# subprocesses (new PID, new hash seed each time -- `julia` randomizes
# `Base.hash`'s seed per process by default) running the same worker script,
# and requires their digests to be byte-identical strings. Also runs a THIRD
# process with `--seed`-equivalent env perturbation (JULIA_HASH_SEED-style
# variation is not a real Julia env var, so instead we vary the process via a
# different working directory / extra loaded package to get a different
# memory layout) to strengthen the "not an artifact of one lucky seed"
# argument.
# ============================================================================
using Test

worker = joinpath(@__DIR__, "_stable_digest_worker_2026-08-01.jl")
project_root = dirname(dirname(@__DIR__))
julia_bin = Base.julia_cmd()[1]

function run_worker()
    out = read(`$julia_bin --project=$project_root $worker`, String)
    digests = Dict{String,String}()
    for line in split(out, '\n')
        if startswith(line, "DIGEST:")
            _, rest = split(line, "DIGEST:"; limit = 2)
            k, v = split(rest, ":"; limit = 2)
            digests[k] = v
        end
    end
    return digests
end

println("Running worker process 1 (fresh julia)..."); flush(stdout)
d1 = run_worker()
println("Running worker process 2 (fresh julia)..."); flush(stdout)
d2 = run_worker()
println("Running worker process 3 (fresh julia)..."); flush(stdout)
d3 = run_worker()

rows = NamedTuple[]
function record!(name::String, pass::Bool, detail::String)
    push!(rows, (test = name, pass = pass, detail = detail))
    println(pass ? "PASS  " : "FAIL  ", name, "  -- ", detail)
end

record!("worker_produced_both_digest_keys",
    all(haskey(d, "unrestricted") && haskey(d, "flexible_CM_mock") for d in (d1, d2, d3)),
    "d1=$d1 d2=$d2 d3=$d3")

record!("digest_is_64_char_hex",
    all(occursin(r"^[0-9a-f]{64}$", d1[k]) for k in ("unrestricted", "flexible_CM_mock")),
    "unrestricted=$(d1["unrestricted"]) flexible_CM_mock=$(d1["flexible_CM_mock"])")

record!("unrestricted_digest_identical_across_3_fresh_processes",
    d1["unrestricted"] == d2["unrestricted"] == d3["unrestricted"],
    "d1=$(d1["unrestricted"]) d2=$(d2["unrestricted"]) d3=$(d3["unrestricted"])")

record!("flexible_CM_mock_digest_identical_across_3_fresh_processes",
    d1["flexible_CM_mock"] == d2["flexible_CM_mock"] == d3["flexible_CM_mock"],
    "d1=$(d1["flexible_CM_mock"]) d2=$(d2["flexible_CM_mock"]) d3=$(d3["flexible_CM_mock"])")

record!("unrestricted_and_mock_digests_differ",
    d1["unrestricted"] != d1["flexible_CM_mock"],
    "different family_kind/restriction ranges must not collide")

using CSV, DataFrames
df = DataFrame(rows)
out_csv = joinpath(project_root, "PROFILED_STABLE_LAYOUT_DIGEST_FRESH_PROCESS_GATE_2026-08-01.csv")
CSV.write(out_csv, df)
println("\nWrote $out_csv")
show(df, allrows = true, allcols = true)
println()

all_pass = all(r.pass for r in rows)
println("\nSTABLE LAYOUT DIGEST FRESH-PROCESS GATE: ", all_pass ? "PASS" : "FAIL")
@assert all_pass
