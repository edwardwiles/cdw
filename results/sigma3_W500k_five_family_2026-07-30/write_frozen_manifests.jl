# write_frozen_manifests.jl -- freezes campaign_config.json + campaign_config.sha256 and copies
# the prerequisite manifests (data/calibration/start) into the immutable campaign root, per the
# sigma3/W500k five-family production campaign brief (2026-07-30).
#
# Run this AFTER Phase C's three_starts_search.jl has produced start_manifest.json -- refuses
# (errors) otherwise, since campaign_config.json's own frozen content includes the start
# manifest's checksum.
#
# Usage: julia results/sigma3_W500k_five_family_2026-07-30/write_frozen_manifests.jl
using Dates, SHA

const CAMPAIGN_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUTS_DIR = joinpath(REPO_ROOT, "campaign_inputs", "sigma3_W500k_2026-07-30")
const D4E = joinpath(REPO_ROOT, "full_aod_diag", "d4_exact")
include(joinpath(D4E, "campaign_cell_io.jl"))  # write_json_file

lp(xs...) = (println(xs...); flush(stdout))

start_manifest_path = joinpath(INPUTS_DIR, "start_manifest.json")
isfile(start_manifest_path) || error("write_frozen_manifests: $start_manifest_path does not exist yet -- run three_starts_search.jl first")

data_manifest_path = joinpath(INPUTS_DIR, "data_manifest.json")
calib_manifest_path = joinpath(INPUTS_DIR, "calibration_manifest.json")
isfile(data_manifest_path) || error("write_frozen_manifests: $data_manifest_path missing")
isfile(calib_manifest_path) || error("write_frozen_manifests: $calib_manifest_path missing")

# copy the 3 prerequisite manifests into the campaign root verbatim (content-addressed, so a
# byte-for-byte copy here is exactly as auditable as the originals in campaign_inputs/)
for (src, dst) in [(data_manifest_path, "data_manifest.json"), (calib_manifest_path, "calibration_manifest.json"),
                    (start_manifest_path, "start_manifest.json")]
    cp(src, joinpath(CAMPAIGN_ROOT, dst); force = true)
    lp("copied ", src, " -> ", joinpath(CAMPAIGN_ROOT, dst))
end

production_sha = strip(read(`git -C $REPO_ROOT rev-parse HEAD`, String))

nproc_str = strip(read(`nproc`, String))
nproc_n = parse(Int, nproc_str)

resource_plan = Dict(
    "generated_at" => string(now()),
    "host_logical_cpus" => nproc_n,
    "families" => 5,
    "julia_threads_per_family" => 20,
    "julia_threads_total" => 100,
    "blas_threads_per_process" => 1,   # OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1, set in run_family_chain_sigma3.sh -- avoids oversubscription against the 20 Julia threads
    "oversubscription_check" => "100 Julia threads requested vs $(nproc_n) logical CPUs -- $(100 <= nproc_n ? "OK, no oversubscription" : "OVERSUBSCRIBED")",
    "numa_policy" => "none -- numactl not installed on this host; threads scheduled by the OS across all NUMA nodes without explicit pinning (documented limitation, see plan doc)",
    "expected_rss_per_family_mb" => "~53000 (observed live during three_starts_search.jl context build at W=500,000, all 5 family contexts resident in one process -- a single-family production process builds only ITS OWN context+pcx, so per-process RSS should be a fraction of this; see RESOURCE_USAGE.csv once cells have actually run for the real per-cell figure)",
    "outer_process_watchdog_s" => 12000,
    "outer_knitro_budget_s" => 10800,
)
write_json_file(joinpath(CAMPAIGN_ROOT, "resource_plan.json"), resource_plan)
lp("wrote resource_plan.json")

campaign_config = Dict(
    "campaign" => "sigma3_W500k_five_family_2026-07-30",
    "generated_at" => string(now()),
    "production_sha" => production_sha,
    "families" => ["unrestricted", "flexible_cm", "common_frechet", "origin_zc", "cm_meanzc"],
    "family_family_note" => "cm_meanzc == \"flexible common marginals + ZC\" (cm_extension=:cm_plus_moments); origin_zc == \"ZC-only\"",
    "sigma" => 3.0,
    "W" => 500_000,
    "draw_design" => "sobol_randomized",
    "draw_seed" => 20260719,
    "exclude_diagonal_gravity" => true,
    "destination_sample" => "exclude_row",
    "deltas" => [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    "directions" => ["upper", "lower"],
    "n_starts" => 3,
    "start_seed" => 20260730,
    "cm_L" => 50,
    "meanzc_K_mean" => 2, "meanzc_K_pair" => 2,
    "originzc_K_mean" => 2, "originzc_K_pair" => 2,
    "outer_strategy_default" => "direct_sr1",
    "outer_strategy_available" => ["direct_sr1", "direct_sr1_with_optional_bfgs_polish"],
    "bfgs_polish_enabled_in_default_launch" => false,
    "outer_knitro_budget_s" => 10800,
    "bfgs_polish_budget_s" => 900,
    "process_group_watchdog_s" => 12000,
    "threads_per_family" => 20,
    "n_families_parallel" => 5,
    "n_cells_total" => 5 * 2 * 6 * 3,  # families x directions x deltas x starts = 180
    "data_manifest_sha256" => bytes2hex(sha256(read(joinpath(CAMPAIGN_ROOT, "data_manifest.json")))),
    "calibration_manifest_sha256" => bytes2hex(sha256(read(joinpath(CAMPAIGN_ROOT, "calibration_manifest.json")))),
    "start_manifest_sha256" => bytes2hex(sha256(read(joinpath(CAMPAIGN_ROOT, "start_manifest.json")))),
)
config_path = joinpath(CAMPAIGN_ROOT, "campaign_config.json")
write_json_file(config_path, campaign_config)
lp("wrote ", config_path)

sha_path = joinpath(CAMPAIGN_ROOT, "campaign_config.sha256")
write(sha_path, bytes2hex(sha256(read(config_path))))
lp("wrote ", sha_path, " = ", read(sha_path, String))

lp("="^100)
lp("FROZEN: campaign_config.json + campaign_config.sha256 + data/calibration/start manifests copied into campaign root")
lp("n_cells_total = ", campaign_config["n_cells_total"], " (5 families x 2 directions x 6 deltas x 3 starts)")
lp("="^100)
