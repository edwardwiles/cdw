# w100k_manifest.jl -- frozen scientific manifest for the W=100k continuation-polish campaign and
# its K=3 extension (task: TARGETED_K3_EXTENSIONS_2026-08-04). Single source of truth for the
# field values every seed/result in the immutable registry must be validated against -- per this
# repo's CLAUDE.md rule against letting a function silently default a scientific parameter, no
# downstream driver may substitute its own copy of these numbers.
#
# Two manifests: MANIFEST_K1 (the existing, completed campaign: MEANZC_K=ORIGINZC_K=1) and
# MANIFEST_K3 (this task's new K_mean=K_pair=3 extension). Everything else is identical between
# them -- K is the only axis that changes.

using SHA

struct FrozenManifest
    formulation::String              # "FULL_gamma_normalized"
    D::Int
    D_dest::Int
    focal::String
    sigma::Float64
    gravity_mask::String              # description of the current theta/mask (own-trade+BR-KR excluded)
    exclude_diagonal_gravity::Bool
    gravity_exclude_brazil_korea::Bool
    destination_sample::Symbol
    draw_design::Symbol
    draw_seed::Int
    W::Int
    A_coordinate_mode::Symbol
    CM_L::Int
    cm_contrasts::Symbol
    MEANZC_K_mean::Int
    MEANZC_K_pair::Int
    ORIGINZC_K_mean::Int
    ORIGINZC_K_pair::Int
end

const MANIFEST_K1 = FrozenManifest(
    "FULL_gamma_normalized", 20, 19, "France", 3.0,
    "current_theta_mask_own_trade_excluded", true, true,
    :exclude_row, :sobol_randomized, 20260719, 100_000, :powered_aspace, 50, :orthonormal,
    1, 1, 1, 1,
)

const MANIFEST_K3 = FrozenManifest(
    "FULL_gamma_normalized", 20, 19, "France", 3.0,
    "current_theta_mask_own_trade_excluded", true, true,
    :exclude_row, :sobol_randomized, 20260719, 100_000, :powered_aspace, 50, :orthonormal,
    3, 3, 3, 3,
)

"""
    manifest_hash(m::FrozenManifest) -> String

Deterministic short hash (first 16 hex chars of SHA256 of the canonical field-by-field string
repr) used as the manifest-hash column in the immutable seed registry and as a namespace key
for K=3 output directories. NOT a substitute for `assert_manifest_compatible`'s field-by-field
check (kept in `continuation_polish_orchestrator.jl`) -- this hash is for indexing/dedup only.
"""
function manifest_hash(m::FrozenManifest)
    parts = [string(getfield(m, f)) for f in fieldnames(FrozenManifest)]
    return bytes2hex(sha256(join(parts, "|")))[1:16]
end

const MANIFEST_K1_HASH = manifest_hash(MANIFEST_K1)
const MANIFEST_K3_HASH = manifest_hash(MANIFEST_K3)

"""
    manifest_as_namedtuple(m) -> NamedTuple

For `assert_manifest_compatible` (field-by-field gate in continuation_polish_orchestrator.jl),
restricted to the fields that function's target/seed manifests actually compare (W, draw_seed,
draw_design, destination_sample, A_coordinate_mode, sigma).
"""
manifest_as_namedtuple(m::FrozenManifest) = (
    W = m.W, draw_seed = m.draw_seed, draw_design = m.draw_design,
    destination_sample = m.destination_sample, A_coordinate_mode = m.A_coordinate_mode,
    sigma = m.sigma,
)

if abspath(PROGRAM_FILE) == @__FILE__
    println("MANIFEST_K1_HASH=", MANIFEST_K1_HASH)
    println("MANIFEST_K3_HASH=", MANIFEST_K3_HASH)
end
