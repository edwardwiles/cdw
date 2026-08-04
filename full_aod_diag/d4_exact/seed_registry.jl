# seed_registry.jl -- immutable, content-addressed result/seed registry for the completed K=1
# continuation-polish campaign, built per task TARGETED_K3_EXTENSIONS_2026-08-04 Section 2.
#
# Freezes every `report.jls` under campaign_output{,_w250k}/ (mutable "latest" paths written by
# continuation_campaign_cell_driver.jl / continuation_campaign_w_extension_driver.jl) to a
# content-addressed, read-only path keyed by sha256(final_w), plus writes a CSV index with the
# fields the task requires: family, direction, delta, W, K_mean, K_pair, manifest hash, point
# digest, best_feasible.w path/hash, gp, GT, Delta-star, verification status, source commit.
#
# This is the mechanism that prevents the K=1 -> K=3 stale-seed race the task warns about
# (Section 2): a downstream K=3 wave reads ONLY from this frozen registry, never from the mutable
# campaign_output/<family>/<direction>/delta_<delta>/report.jls "latest" path, which a concurrent
# same-family refinement round could still be overwriting.
#
# Usage: julia --project=. seed_registry.jl [output_csv_path]
# Requires the real driver chain to already be include-d (same chain
# continuation_campaign_cell_driver.jl uses) so `OrchestratorRunState`/`FinalCellReport` deserialize
# correctly, plus w100k_manifest.jl and continuation_polish_orchestrator.jl for SHA/Serialization.

using Dates, SHA, Serialization

const D4E = @__DIR__
include(joinpath(D4E, "w100k_manifest.jl"))
# continuation_polish_orchestrator.jl defines OrchestratorRunState/FinalCellReport/load_run_state.
# Caller must include the real driver chain (c10_d20_production_driver*.jl etc.) BEFORE this file
# if not already loaded, exactly as continuation_campaign_cell_driver.jl does; guarded so this file
# can also be `include`-d twice harmlessly from an already-warm REPL/driver session.
isdefined(Main, :OrchestratorRunState) ||
    include(joinpath(D4E, "continuation_polish_orchestrator.jl"))

const CAMPAIGN_OUTPUT_ROOTS = [
    ("W100k", "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/campaign_output"),
    ("W250k", "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/campaign_output_w250k"),
]

const FROZEN_SEEDS_ROOT = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1"

const K_TABLE = Dict(  # family -> (K_mean, K_pair) at K=1 campaign scope
    "unrestricted"    => (0, 0),
    "flexible_cm"      => (0, 0),
    "common_frechet"   => (0, 0),
    "cm_meanzc"        => (1, 1),
    "origin_zc"        => (1, 1),
)

function branch_head_sha(worktree_dir::AbstractString)
    try
        return strip(read(`git -C $worktree_dir rev-parse HEAD`, String))
    catch
        return "unknown"
    end
end

"""
    freeze_report!(w_tag, root_dir, family, direction, delta_dirname, report_path, rows)

Deserializes one `report.jls`, computes sha256(final_w), copies the ENTIRE state to a
content-addressed, read-only (chmod 444) path under FROZEN_SEEDS_ROOT, and appends one row to
`rows`. Idempotent: if the content-addressed path already exists, does not rewrite it (an
existing frozen file is never mutated -- that is the whole point).
"""
function freeze_report!(w_tag::String, family::String, direction::String, delta_dirname::String,
                         report_path::String, rows::Vector, source_commit::AbstractString)
    state = deserialize(report_path)  # OrchestratorRunState
    report = state.report
    report === nothing && return nothing
    w = report.final_w
    digest = bytes2hex(sha256(join(string.(w), ",")))
    K_mean, K_pair = get(K_TABLE, family, (0, 0))
    manifest = MANIFEST_K1
    mhash = MANIFEST_K1_HASH

    dest_dir = joinpath(FROZEN_SEEDS_ROOT, mhash, w_tag, family, direction, delta_dirname)
    isdir(dest_dir) || mkpath(dest_dir)
    dest_path = joinpath(dest_dir, "$(digest[1:16]).jls")
    if !isfile(dest_path)
        serialize(dest_path, state)
        chmod(dest_path, 0o444)
    end

    # delta_dirname is normally "delta_<float>", but a same-delta refinement round's own
    # record-keeping OUTPUT_DIR copy is named "delta_<float>_round2" etc. -- extract the leading
    # float either way so it lands under the correct delta, not NaN.
    m = match(r"delta_([0-9.]+)", delta_dirname)
    delta_val = m === nothing ? NaN : parse(Float64, m.captures[1])

    push!(rows, (
        family = family, direction = direction, delta = delta_val, W = manifest.W,
        K_mean = K_mean, K_pair = K_pair, manifest_hash = mhash, point_digest = digest,
        frozen_path = dest_path, source_report_path = report_path,
        gp = w[1], GT = report.final_GT, Delta_star = report.final_Delta_star,
        result_source = report.result_source, knitro_status = report.knitro_status,
        verification_status = isfinite(report.final_GT) && !isempty(w) ? "verified" : "UNVERIFIED",
        source_commit = source_commit, w_scale = w_tag, source_dirname = delta_dirname,
    ))
    return dest_path
end

function build_registry(out_csv::String)
    source_commit = branch_head_sha(D4E)
    rows = Vector{Any}()
    for (w_tag, root) in CAMPAIGN_OUTPUT_ROOTS
        isdir(root) || continue
        for family in readdir(root)
            fam_dir = joinpath(root, family)
            isdir(fam_dir) || continue
            for direction in readdir(fam_dir)
                dir_dir = joinpath(fam_dir, direction)
                isdir(dir_dir) || continue
                for delta_dirname in readdir(dir_dir)
                    report_path = joinpath(dir_dir, delta_dirname, "report.jls")
                    isfile(report_path) || continue
                    try
                        freeze_report!(w_tag, family, direction, delta_dirname, report_path, rows, source_commit)
                    catch e
                        @warn "freeze_report! failed" report_path exception=e
                    end
                end
            end
        end
    end

    header = ["family", "direction", "delta", "W", "K_mean", "K_pair", "manifest_hash",
              "point_digest", "frozen_path", "source_report_path", "gp", "GT", "Delta_star",
              "result_source", "knitro_status", "verification_status", "source_commit", "w_scale",
              "source_dirname"]
    # Dedup exact content duplicates (same family/direction/delta/W/digest) -- a same-delta
    # refinement round's OUTPUT_DIR record copy is byte-identical to the canonical path once the
    # round improved on itself; keep only the canonical (non-"_round"-suffixed) source_dirname.
    seen = Set{Tuple}()
    dedup_rows = Any[]
    sorted_rows = sort(rows, by = r -> occursin("_round", r.source_dirname) ? 1 : 0)
    for r in sorted_rows
        key = (r.family, r.direction, r.delta, r.W, r.point_digest)
        key in seen && continue
        push!(seen, key)
        push!(dedup_rows, r)
    end
    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in dedup_rows
            println(io, join([getfield_or(r, Symbol(h)) for h in header], ","))
        end
    end
    chmod(out_csv, 0o444)
    println("Wrote ", length(dedup_rows), " rows (", length(rows) - length(dedup_rows),
            " exact-content duplicates collapsed) to ", out_csv, " (chmod 444, immutable)")
    return dedup_rows
end

getfield_or(nt::NamedTuple, s::Symbol) = haskey(nt, s) ? nt[s] : ""

if abspath(PROGRAM_FILE) == @__FILE__
    out_csv = length(ARGS) >= 1 ? ARGS[1] :
        "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/IMMUTABLE_SEED_REGISTRY_K1_2026-08-04.csv"
    build_registry(out_csv)
end
