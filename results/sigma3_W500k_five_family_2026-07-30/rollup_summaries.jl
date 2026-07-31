# rollup_summaries.jl -- sigma3/W500k five-family production campaign (2026-07-30)
#
# Consolidates the per-cell artifacts every cell already writes (campaign_cell_io.jl's
# write_cell_status!: config.json, outer_status.json, best_verified_incumbent.json (if any),
# inner_status_summary.csv, DONE/FAILED markers, plus the <label>_backend_manifest.json each real
# driver writes automatically) into the 8 campaign-wide CSVs the brief requires. Purely a
# read-and-merge script -- writes nothing back to any cell directory, safe to re-run at any time
# (including mid-campaign) to get a fresh snapshot. Designed to be atomic: writes to a .tmp path
# then renames, so a concurrent --status read never sees a half-written CSV.
#
# Usage: julia --project=. results/sigma3_W500k_five_family_2026-07-30/rollup_summaries.jl [outroot]
using Dates

const CAMPAIGN_ROOT = length(ARGS) >= 1 ? ARGS[1] : @__DIR__
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const D4E = joinpath(REPO_ROOT, "full_aod_diag", "d4_exact")
include(joinpath(D4E, "campaign_cell_io.jl"))  # json_write_value pattern reused for reading via json_lite-style parsing below
include(joinpath(D4E, "json_lite.jl"))

const FAMILIES = ["unrestricted", "flexible_cm", "common_frechet", "cm_meanzc", "origin_zc"]
const DIRECTIONS = ["upper", "lower"]

lp(xs...) = (println(xs...); flush(stdout))

function atomic_write_csv(path, header, rows)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, header)
        for r in rows
            println(io, r)
        end
    end
    mv(tmp, path; force = true)
    lp("wrote ", path, " (", length(rows), " rows)")
end

jget(d, k, default = missing) = haskey(d, k) ? d[k] : default
jstr(x) = x === missing ? "" : (x isa AbstractString ? x : string(x))

# ---- discover every cell directory: OUTROOT/<family>/<direction>/delta_<d>/start_<s> ----
cells = NamedTuple[]
for fam in FAMILIES, dir in DIRECTIONS
    famdir = joinpath(CAMPAIGN_ROOT, fam, dir)
    isdir(famdir) || continue
    for deltadir in readdir(famdir)
        m = match(r"^delta_(.+)$", deltadir)
        m === nothing && continue
        for startdir in readdir(joinpath(famdir, deltadir))
            m2 = match(r"^start_(\d+)$", startdir)
            m2 === nothing && continue
            ckdir = joinpath(famdir, deltadir, startdir)
            push!(cells, (family = fam, direction = dir, delta = m.captures[1], start = parse(Int, m2.captures[1]), ckdir = ckdir))
        end
    end
end
lp("discovered ", length(cells), " cell directories under ", CAMPAIGN_ROOT)

# ---- ALL_CELLS.csv ----
all_cells_rows = String[]
best_rows = Dict{Tuple{String,String,String},Any}()          # (family,direction,delta) -> best row
best_start_rows = Dict{Tuple{String,String,String,Int},Any}() # (family,direction,delta,start) -> row
chain_status = Dict{String,Any}()  # family -> (done, failed, pending)
inner_audit_rows = String[]
outer_audit_rows = String[]
backend_audit_rows = String[]
resource_rows = String[]

for c in cells
    status = cell_already_done(c.ckdir) ? "DONE" : (isfile(joinpath(c.ckdir, "FAILED")) ? "FAILED" : "PENDING")
    cfg_path = joinpath(c.ckdir, "config.json")
    outer_path = joinpath(c.ckdir, "outer_status.json")
    incumbent_path = joinpath(c.ckdir, "best_verified_incumbent.json")
    cfg = isfile(cfg_path) ? json_load(cfg_path) : Dict{String,Any}()
    outer = isfile(outer_path) ? json_load(outer_path) : Dict{String,Any}()
    incumbent = isfile(incumbent_path) ? json_load(incumbent_path) : nothing

    best_gp = incumbent === nothing ? "" : jstr(jget(incumbent, "gp"))
    best_delta_star = incumbent === nothing ? "" : jstr(jget(incumbent, "Delta"))
    outer_status_str = jstr(jget(outer, "outer_status", "unknown"))
    push!(all_cells_rows, join([c.family, c.direction, c.delta, c.start, status, outer_status_str,
        jstr(jget(outer, "elapsed_s")), jstr(jget(outer, "n_inner_solves")), jstr(jget(outer, "n_verified")),
        best_gp, best_delta_star, jstr(jget(outer, "error"))], ","))

    key3 = (c.family, c.direction, c.delta)
    if incumbent !== nothing
        d = something(tryparse(Float64, best_delta_star), Inf)
        prior = get(best_rows, key3, nothing)
        if prior === nothing || d < prior[1]
            best_rows[key3] = (d, c.start, best_gp, best_delta_star, status)
        end
        best_start_rows[(c.family, c.direction, c.delta, c.start)] = (best_gp, best_delta_star, status)
    end

    cs = get!(chain_status, c.family, Dict("DONE" => 0, "FAILED" => 0, "PENDING" => 0))
    cs[status] += 1

    inner_csv = joinpath(c.ckdir, "inner_status_summary.csv")
    if isfile(inner_csv)
        lines = readlines(inner_csv)
        for line in (length(lines) > 1 ? lines[2:end] : String[])
            push!(inner_audit_rows, string(c.family, ",", c.direction, ",", c.delta, ",", c.start, ",", line))
        end
    end

    push!(outer_audit_rows, join([c.family, c.direction, c.delta, c.start, outer_status_str,
        jstr(jget(outer, "outer_evals")), jstr(jget(outer, "budget_residual")), jstr(jget(outer, "error"))], ","))

    backend_path = joinpath(c.ckdir, "$(c.family)_$(c.direction)_d$(c.delta)_s$(c.start)_backend_manifest.json")
    if !isfile(backend_path)
        # fall back to a glob -- the exact <label> naming has varied historically (see
        # CACHE_NAMESPACE_NOTE.md); find any *_backend_manifest.json in this cell dir.
        cands = filter(f -> endswith(f, "_backend_manifest.json"), readdir(c.ckdir))
        backend_path = isempty(cands) ? "" : joinpath(c.ckdir, cands[1])
    end
    if !isempty(backend_path) && isfile(backend_path)
        bm = json_load(backend_path)
        structural = jget(bm, "structural", Dict{String,Any}())
        push!(backend_audit_rows, join([c.family, c.direction, c.delta, c.start,
            jstr(jget(bm, "bundle_invariant_pass")), jstr(jget(structural, "bundle_type"))], ","))
    else
        push!(backend_audit_rows, join([c.family, c.direction, c.delta, c.start, "MISSING", "MISSING"], ","))
    end

    rusage_path = joinpath(dirname(dirname(dirname(dirname(c.ckdir)))), "logs", "cells")  # OUTROOT/logs/cells
    # RESOURCE_USAGE.csv: parsed from run_family_chain_sigma3.sh's `/usr/bin/time -v` sidecar
    # files (<celllog>.rusage), NOT from anything Julia-side -- OS-level RSS/wall-time, matching
    # the brief's "expected RSS" resource-plan requirement rather than an in-process estimate.
    rss_kb = ""; wall_s = ""
    if isdir(rusage_path)
        cand = filter(f -> occursin("$(c.family)_$(c.direction)_d$(c.delta)_s$(c.start)_", f) && endswith(f, ".rusage"), readdir(rusage_path))
        if !isempty(cand)
            txt = read(joinpath(rusage_path, cand[end]), String)
            m_rss = match(r"Maximum resident set size \(kbytes\): (\d+)", txt)
            m_wall = match(r"Elapsed \(wall clock\) time.*: (.+)", txt)
            rss_kb = m_rss === nothing ? "" : m_rss.captures[1]
            wall_s = m_wall === nothing ? "" : strip(m_wall.captures[1])
        end
    end
    push!(resource_rows, join([c.family, c.direction, c.delta, c.start, rss_kb, jstr(wall_s)], ","))
end

atomic_write_csv(joinpath(CAMPAIGN_ROOT, "ALL_CELLS.csv"),
    "family,direction,delta,start,status,outer_status,elapsed_s,n_inner_solves,n_verified,best_gp,best_Delta_star,error",
    all_cells_rows)

best_fdd_rows = [join([k[1], k[2], k[3], v[2], v[3], v[4], v[5]], ",") for (k, v) in best_rows]
atomic_write_csv(joinpath(CAMPAIGN_ROOT, "BEST_BY_FAMILY_DIRECTION_DELTA.csv"),
    "family,direction,delta,best_start,best_gp,best_Delta_star,status", best_fdd_rows)

best_fdds_rows = [join([k[1], k[2], k[3], k[4], v[1], v[2], v[3]], ",") for (k, v) in best_start_rows]
atomic_write_csv(joinpath(CAMPAIGN_ROOT, "BEST_BY_FAMILY_DIRECTION_DELTA_START.csv"),
    "family,direction,delta,start,gp,Delta_star,status", best_fdds_rows)

chain_rows = [join([fam, get(cs, "DONE", 0), get(cs, "FAILED", 0), get(cs, "PENDING", 0)], ",") for (fam, cs) in chain_status]
atomic_write_csv(joinpath(CAMPAIGN_ROOT, "FAMILY_CHAIN_STATUS.csv"), "family,n_done,n_failed,n_pending", chain_rows)

atomic_write_csv(joinpath(CAMPAIGN_ROOT, "INNER_STATUS_AUDIT.csv"),
    "family,direction,delta,start," * (isempty(inner_audit_rows) ? "eval_idx,t_s,gp,Delta_star,feasible,verified_success,classification_proxy,final_incumbent" : "raw_inner_row"),
    inner_audit_rows)

atomic_write_csv(joinpath(CAMPAIGN_ROOT, "OUTER_TERMINATION_AUDIT.csv"),
    "family,direction,delta,start,outer_status,outer_evals,budget_residual,error", outer_audit_rows)

atomic_write_csv(joinpath(CAMPAIGN_ROOT, "BACKEND_INVARIANT_AUDIT.csv"),
    "family,direction,delta,start,bundle_invariant_pass,bundle_type", backend_audit_rows)

atomic_write_csv(joinpath(CAMPAIGN_ROOT, "RESOURCE_USAGE.csv"),
    "family,direction,delta,start,max_rss_kb,wall_clock", resource_rows)

lp("="^100)
lp("ROLLUP COMPLETE -- ", now(), " -- ", length(cells), " cells scanned")
lp("="^100)
