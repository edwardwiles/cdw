# build_k3_transplant_seeds.jl -- builds the K=1->K=3 transplant seed for origin_zc and cm_meanzc
# (task §9), using the SAME validated construction the K=3 preflight's own
# test_k3_preflight_w100k_construction_smoke.jl already confirmed works at real W=100,000 scale
# (real K=1 W=100k economic block from the frozen registry + a FRESH K=3 eta_nu tail built from the
# true analytic moment under F* -- nu0_shared/nu0_origin, factorial(k) -- NOT an empirical Monte
# Carlo average, matching the already-validated D4 preflight's own construction too).
#
# Wraps each transplant w0 as an OrchestratorRunState/FinalCellReport so continuation_campaign_
# cell_driver.jl's own `extra_seed:<path>:<role>` mechanism can load it directly as a Seed -- no
# reused/pre-solved dual, no K=1 duals carried forward (task §9's own explicit requirement).
#
# Usage: julia --project=. build_k3_transplant_seeds.jl
using Dates, SHA, Serialization

const D4E = @__DIR__
isdefined(Main, :d20_real_setup) || include(joinpath(D4E, "full_chain_include.jl"))
isdefined(Main, :OrchestratorRunState) || include(joinpath(D4E, "continuation_polish_orchestrator.jl"))

lp(xs...) = (println(xs...); flush(stdout))

const OUT_DIR = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/POST_VERIFY_FIX/k3_transplant_seeds"
isdir(OUT_DIR) || mkpath(OUT_DIR)

const OZ_K1_SEED_PATH = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1/890424b84acebe64/W100k/origin_zc/upper/delta_0.01/cdb537b72ace59c1.jls"
const MZ_K1_SEED_PATH = "/bbkinghome/edav/repo_scratch/fullA-continuation-polish-2026-08-03/frozen_seeds_K1/890424b84acebe64/W100k/cm_meanzc/upper/delta_2.0/96fc0c36dbe4687b.jls"

nu0_shared(K::Int) = [Float64(factorial(k)) for k in 1:K]
nu0_origin(K::Int, D::Int) = vcat([fill(Float64(factorial(k)), D) for k in 1:K]...)

function build_seed(family::String, k1_seed_path::String, D2_econ::Int, nu0::Vector{Float64})
    isfile(k1_seed_path) || error("K=1 frozen seed not found: $k1_seed_path")
    w_a = deserialize(k1_seed_path).report.final_w[1:D2_econ]
    w0 = vcat(w_a, log.(nu0))
    digest = bytes2hex(sha256(join(string.(w0), ",")))

    report = FinalCellReport(
        family, :upper, 0.01, "K1_to_K3_transplant", "K1_to_K3_transplant_unverified",
        nothing, nothing, NaN, w0, NaN,
        "K1_to_K3_transplant_seed_unverified", -999, 0.0, 0,
        k1_seed_path,
    )
    state = OrchestratorRunState(family, :upper, 0.01, EXPLORE_DIRECT_SR1, "K1_to_K3_transplant_unverified",
                                  string(Dates.now()), report)
    dest = joinpath(OUT_DIR, "$(family)_k3_transplant_seed.jls")
    save_run_state(dest, state)
    chmod(dest, 0o444)
    lp("Wrote ", dest, " (w length=", length(w0), ", digest=", digest[1:16], ")")
    return dest
end

lp("Building shared W=100,000 real D=20 context (draws only, for D/D2_econ)...")
# Must match the PRODUCTION gravity mask (exclude_diagonal_gravity=true,
# gravity_exclude_cells=Brazil-Korea) -- these are the ACTUAL DRIVER functions' own defaults
# (run_originzc_upper_checkpointed/run_cm_upper_checkpointed), but the lower-level d20_real_setup
# called directly here still defaults to false/empty (deliberately left there, per this repo's own
# documented CLAUDE.md history -- only the 3 real production driver functions were flipped). Using
# d20_real_setup's own default silently builds a differently-masked context than the real solve
# uses, which is exactly the dimension-mismatch bug this fix addresses.
ctx = d20_real_setup(W = 100_000, δ = 1.0, find_smallest = true, destination_sample = :exclude_row,
    exclude_diagonal_gravity = true, gravity_exclude_cells = default_gravity_exclude_cells_brazil_korea())
D = ctx.D
D2_econ = D * (hasproperty(ctx, :D_dest) ? ctx.D_dest : D)
lp("  D=", D, " D2_econ=", D2_econ)

oz_path = build_seed("origin_zc", OZ_K1_SEED_PATH, D2_econ, nu0_origin(3, D))
mz_path = build_seed("cm_meanzc", MZ_K1_SEED_PATH, D2_econ, nu0_shared(3))

lp("\nDone. Seeds:")
lp("  origin_zc: ", oz_path)
lp("  cm_meanzc: ", mz_path)
