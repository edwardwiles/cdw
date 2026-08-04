module FixedStateInnerABFrozenManifest
# Task "Claude Code task A: all-family fixed-state FULL-versus-REDUCED inner A/B", step 2
# ("Freeze one common scientific manifest"). This file is intentionally the ONLY artifact this
# task produced beyond the branch/worktree themselves: per the task's own stop condition
# ("If the outer task has not completed, stop after step 2. Do not wait by launching background
# work"), execution stopped here. See ../../../docs/audits/profiled-fixed-state-inner-ab-2026-08-04/
# MASTER.md for the live evidence that the concurrent outer-readiness task had not completed.
#
# Read-only with respect to scientific_manifest/*.jl (ScientificManifest, RunManifest,
# ABComparability) -- those are the concurrent second Claude's territory (FamilyRegistry
# capability definitions, coordinate-mode implementation) and are only `include`d here, never
# edited. Same for src outer-gradient/bandwidth-cache/canonical-outer-runner code: not touched.
#
# Purpose: pin down the ONE ScientificManifest per execution mode (Mode A diagnostic,
# Mode B production) that every later FULL and REDUCED RunManifest in this benchmark must share,
# per task section 2's explicit field list. Values are taken verbatim from the real current
# production manifest (`configs/fullA_production_2026-08-03.toml`, confirmed live at HEAD
# `395dec3` == tag `profiled-functional-ready-2026-08-04`), not invented -- only W/julia_threads
# differ between modes, exactly as task section 7 specifies (W=20_000/threads=1 for Mode A;
# W=100_000/production thread policy for Mode B). blas_threads is left at the toml's own
# production value (8) for both modes, since the task does not ask for a second BLAS policy for
# Mode A beyond "BLAS threads=1" -- NOTE this one field is a genuine open point, not silently
# resolved: see MASTER.md's blocker list.

using TOML

include(joinpath(@__DIR__, "..", "..", "..", "scientific_manifest", "ScientificManifest.jl"))
using .ScientificManifestMod: ScientificManifest

export mode_a_scientific_manifest, mode_b_scientific_manifest, FIXED_STATE_LOWER_LIMIT,
    ECONOMIC_PARAMETERIZATION_FULL, ECONOMIC_PARAMETERIZATION_REDUCED

const PRODUCTION_MANIFEST_TOML = joinpath(@__DIR__, "..", "..", "..", "configs",
    "fullA_production_2026-08-03.toml")

const ECONOMIC_PARAMETERIZATION_FULL = :full_gamma_normalized
const ECONOMIC_PARAMETERIZATION_REDUCED = :profiled_destination_scales

"""
Task CLAUDE.md: `lower_limit=-50` (or similar) is the documented, deliberate KNITRO inner-objective
floor that terminates a genuinely unbounded solve fast rather than chasing it to -Inf -- NOT an
unresolved unknown (see the repo's own top-level CLAUDE.md, section on the two recurring wrong
KNITRO explanations). Frozen here at the documented value so both FULL and REDUCED arms use the
identical floor, per task section 2's explicit "require identical: lower_limit" instruction.
"""
const FIXED_STATE_LOWER_LIMIT = -50.0

function _base_fields()
    d = TOML.parsefile(PRODUCTION_MANIFEST_TOML)
    gravity_cells = Tuple{Int,Int}[Tuple(Int.(c)) for c in d["gravity_exclude_cells"]]
    return d, gravity_cells
end

"""
    mode_a_scientific_manifest() -> ScientificManifest

Task section 7 "Mode A: diagnostic parity" -- W=20,000, JULIA_NUM_THREADS=1, BLAS threads=1.
Every other field is the real production value from `configs/fullA_production_2026-08-03.toml`
(dataset/checksum, country order, France focal, sigma=3, gravity mask incl. the Brazil-Korea
exclusion, destination_sample=:exclude_row, sobol_randomized draw design/seed, L/K, inner/outer
KNITRO option checksums).
"""
function mode_a_scientific_manifest()
    d, gravity_cells = _base_fields()
    return ScientificManifest(;
        dataset_version = d["dataset_version"],
        dataset_checksum = d["dataset_checksum"],
        country_order = String.(d["country_order"]),
        focal_country = d["focal_country"],
        sigma = Float64(d["sigma"]),
        exclude_diagonal_gravity = d["exclude_diagonal_gravity"],
        gravity_exclude_cells = gravity_cells,
        destination_sample = Symbol(d["destination_sample"]),
        draw_design = Symbol(d["draw_design"]),
        draw_seed = Int(d["draw_seed"]),
        W = 20_000,
        L = Int(d["L"]),
        K_mean = Int(d["K_mean"]),
        K_pair = Int(d["K_pair"]),
        julia_threads = 1,
        blas_threads = 1,
        inner_opt_checksum = d["inner_opt_checksum"],
        outer_opt_checksum = d["outer_opt_checksum"])
end

"""
    mode_b_scientific_manifest() -> ScientificManifest

Task section 7 "Mode B: production parity" -- W=100,000, JULIA_NUM_THREADS=10, production BLAS
thread policy, fixed disjoint CPU set (the CPU-set pinning itself is a runtime/OS-level concern,
not a ScientificManifest field, and is NOT frozen here -- see MASTER.md open point). All other
fields identical to Mode A, and identical to the live real production manifest verbatim.
"""
function mode_b_scientific_manifest()
    d, gravity_cells = _base_fields()
    return ScientificManifest(;
        dataset_version = d["dataset_version"],
        dataset_checksum = d["dataset_checksum"],
        country_order = String.(d["country_order"]),
        focal_country = d["focal_country"],
        sigma = Float64(d["sigma"]),
        exclude_diagonal_gravity = d["exclude_diagonal_gravity"],
        gravity_exclude_cells = gravity_cells,
        destination_sample = Symbol(d["destination_sample"]),
        draw_design = Symbol(d["draw_design"]),
        draw_seed = Int(d["draw_seed"]),
        W = Int(d["W"]),
        L = Int(d["L"]),
        K_mean = Int(d["K_mean"]),
        K_pair = Int(d["K_pair"]),
        julia_threads = Int(d["julia_threads"]),
        blas_threads = Int(d["blas_threads"]),
        inner_opt_checksum = d["inner_opt_checksum"],
        outer_opt_checksum = d["outer_opt_checksum"])
end

end # module FixedStateInnerABFrozenManifest
