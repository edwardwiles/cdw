# Top-level command for the paper_upper_v1 protocol.
#
#   julia run_paper_upper_bounds.jl --protocol protocols/paper_upper_v1.toml [--campaign-root DIR]
#
# Reruns are idempotent: each stage/cell checks its own resume_bundle.jls (protocol_sha-matched)
# before doing any work, and never overwrites a completed one (family_start_chain.jl's own
# run_delta_cell). This top-level script's job is just sequencing: seeds -> Phase I waves (in
# order, never overlapping two waves) -> [Phase II / Phase III -- still under construction, see
# ORCHESTRATOR_STATUS.md].
#
# Also supports:
#   julia run_paper_upper_bounds.jl --protocol <toml> --continue-cell FAMILY:START:delta_D --extra-hours H
#       (writes to convergence_extensions/, never touches the base phase1_discovery/ result --
#        NOT YET IMPLEMENTED, see ORCHESTRATOR_STATUS.md)

using TOML

function parse_args(argv)
    d = Dict{String,String}()
    i = 1
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            key = a[3:end]
            val = (i < length(argv) && !startswith(argv[i+1], "--")) ? argv[i+1] : "true"
            d[key] = val
            i += (val == "true" && (i == length(argv) || startswith(argv[i+1], "--"))) ? 1 : 2
        else
            i += 1
        end
    end
    return d
end

const ARGD = parse_args(ARGS)
haskey(ARGD, "protocol") || error("usage: julia run_paper_upper_bounds.jl --protocol protocols/paper_upper_v1.toml [--campaign-root DIR]")
const PROTOCOL_TOML = abspath(ARGD["protocol"])
const MANIFEST = TOML.parsefile(PROTOCOL_TOML)
const CAMPAIGN_ROOT = get(ARGD, "campaign-root", MANIFEST["output"]["root"])
const SRC_DIR = @__DIR__

lp(xs...) = (println(xs...); flush(stdout))

if haskey(ARGD, "continue-cell") || haskey(ARGD, "continue-status")
    error("run_paper_upper_bounds.jl: --continue-cell / --continue-status (convergence extension " *
          "launcher) is not implemented yet in this session's build -- see ORCHESTRATOR_STATUS.md. " *
          "Do not hand-roll a workaround that writes into phase1_discovery/ directly; wait for the " *
          "real extension command (writes under convergence_extensions/ only).")
end

MANIFEST["source"]["protocol_sha"] != "PENDING_COMMIT" ||
    error("run_paper_upper_bounds.jl: protocols/paper_upper_v1.toml still has protocol_sha=PENDING_COMMIT -- " *
          "commit the multistart_seed_generator.jl / cm_checkpoint.jl / cm_originzc_checkpoint.jl extensions " *
          "to the protocol branch first and record the real commit SHA in the manifest before launching.")

mkpath(CAMPAIGN_ROOT)
mkpath(joinpath(CAMPAIGN_ROOT, "protocol"))
cp(PROTOCOL_TOML, joinpath(CAMPAIGN_ROOT, "protocol", "paper_upper_v1.toml"); force = false)

lp("="^100)
lp("paper_upper_v1 launch: protocol=", MANIFEST["protocol"]["name"], " campaign_root=", CAMPAIGN_ROOT)
lp("protocol_sha=", MANIFEST["source"]["protocol_sha"])
lp("="^100)

# ---- Phase 0: seeds ----
seeds_manifest_path = joinpath(CAMPAIGN_ROOT, "seeds", "manifest.jls")
if isfile(seeds_manifest_path)
    lp("Seeds already generated (", seeds_manifest_path, " exists) -- SKIPPING seed generation.")
else
    MANIFEST["seeds"]["A_scale"] isa Number ||
        error("run_paper_upper_bounds.jl: [seeds].A_scale not frozen -- run paper_upper_v1_orchestrator/scan_seed_scales.jl first.")
    lp("Generating seeds...")
    run(`julia --project=$SRC_DIR $(joinpath(SRC_DIR, "paper_upper_v1_orchestrator", "generate_seeds.jl")) $PROTOCOL_TOML $CAMPAIGN_ROOT`)
end

# ---- Phase I: waves ----
n_starts = MANIFEST["seeds"]["number_of_starts"]
starts_per_wave = get(get(MANIFEST, "concurrency", Dict()), "phase1_starts_per_wave", 2)
start_ids = ["S$(i)" for i in 0:(n_starts - 1)]

wave_idx = 0
core_offset = 0
i = 1
while i <= length(start_ids)
    global wave_idx += 1
    wave_starts = start_ids[i:min(i + starts_per_wave - 1, length(start_ids))]
    if length(wave_starts) < starts_per_wave
        lp("Final partial wave: ", wave_starts)
    end
    lp("-"^100); lp("WAVE ", wave_idx, ": ", wave_starts)
    cmd = `bash $(joinpath(SRC_DIR, "paper_upper_v1_orchestrator", "launch_wave.sh")) $PROTOCOL_TOML $CAMPAIGN_ROOT $(wave_starts...) $core_offset`
    run(cmd)
    global core_offset = 0   # each wave reuses the same core range (waves never overlap in time)
    global i += starts_per_wave
end

lp("="^100)
lp("PHASE I COMPLETE. Phase II (cross-family projection) and Phase III (final polish) are not yet")
lp("implemented in this session's build -- see ORCHESTRATOR_STATUS.md for what remains.")
lp("="^100)
