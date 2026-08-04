module RunManifestMod
# RunManifest -- one typed, versioned, serializable record of everything that determines what a
# single FULL-or-REDUCED production/A-B run actually computed: the ScientificManifest fields
# (what economic problem) plus the run/formulation-specific fields task
# profiled-outer-production-readiness-2026-08-03 §3 additionally requires (family, formulation,
# coordinate mode, nu policy, algorithm/budget, cache/bank/warm-start policy, provenance).
#
# Deliberately does NOT duplicate any ScientificManifest field -- this struct wraps one
# `ScientificManifest` instance (field `sci`) rather than re-declaring sigma/W/L/K/draws/etc.
# Deliberately separate from Melitz, same as ScientificManifest itself (user directive,
# 2026-08-03) -- nothing here is `using`/`include`d by src/melitz, scripts/melitz_*, melitz/,
# test/melitz/, and this module includes none of those.
#
# Scope note (2026-08-03): this commit defines and tests the manifest type itself. It does NOT
# yet wire "every run must write run_manifest.json before computation" or "refuse execution from
# a dirty worktree" into any actual runner -- `write_run_manifest_json` and `refuse_if_dirty`
# below are the primitives a future canonical-runner commit (task §4) must call, not evidence
# that a runner already calls them. Nothing in this file makes any production or REDUCED default
# change, and no A-coordinate-mode/free-nu enumeration is asserted as "complete" here -- see
# `A_coordinate_mode`/`nu_policy` field docs below for exactly what is and isn't validated.

using TOML
using SHA
using Dates

include(joinpath(@__DIR__, "ScientificManifest.jl"))
using .ScientificManifestMod: ScientificManifest, to_toml_dict as sci_to_toml_dict,
    from_toml_dict as sci_from_toml_dict, validate_manifest as validate_sci_manifest

export RunManifest, RUN_MANIFEST_SCHEMA_VERSION, VALID_FAMILIES, VALID_ECONOMIC_PARAMETERIZATIONS,
    VALID_NU_POLICIES, to_toml_dict, from_toml_dict, write_manifest_toml, read_manifest_toml,
    write_run_manifest_json, validate_manifest, current_source_sha, source_is_dirty, refuse_if_dirty,
    digest_economic_state

const RUN_MANIFEST_SCHEMA_VERSION = 1

"""
Real family symbols as used throughout `full_aod_diag/d4_exact/production_backend_manifest.jl`
and the family-specific production drivers -- not invented for this task. Task
profiled-outer-production-readiness-2026-08-03's prose labels ("origin_ZC", "CM_plus_ZC",
"flexible_CM") map onto these as `:origin_zc`, `:cm_meanzc`, `:flexible_cm` respectively.
"""
const VALID_FAMILIES = (:unrestricted, :flexible_cm, :common_frechet, :origin_zc, :cm_meanzc)

"""
Real formulation symbols, from `VALID_ECONOMIC_PARAMETERIZATIONS`
(`full_aod_diag/d4_exact/profiled_ab_comparability_and_plumbing_2026-08-01.jl:121`), reproduced
here (not re-`include`d, to avoid pulling that file's full dependency chain into a manifest-only
module) -- kept as the identical tuple; a test in this module's test file cross-checks the two
files have not drifted.
"""
const VALID_ECONOMIC_PARAMETERIZATIONS = (:full_gamma_normalized, :profiled_destination_scales)

"nu is either not searched (`:fixed`, the current behavior of every family per [[full-vs-reduced-forensic-audit-2026-08-03]]: 'REDUCED never threads nu -- held fixed by explicit design') or searched as part of the outer vector (`:free`, not yet implemented anywhere as of this commit -- task §6)."
const VALID_NU_POLICIES = (:fixed, :free)

"""
    RunManifest

Wraps one `ScientificManifest` (what economic problem) with the additional fields that
determine what a specific FULL-or-REDUCED *run* of that problem computed. No field has a
default -- every field must be supplied explicitly by the caller, matching the project rule
that no function may silently default a value that changes what is being solved or how it was
run (CLAUDE.md, and see memory `feedback-no-defaults-on-any-input-or-setting`).

Field notes:
  - `family::Symbol`: one of `VALID_FAMILIES`.
  - `economic_parameterization::Symbol`: one of `VALID_ECONOMIC_PARAMETERIZATIONS`
    (`:full_gamma_normalized` = FULL, `:profiled_destination_scales` = REDUCED).
  - `A_coordinate_mode::Symbol`: free-form, NOT validated against an enumeration here. FULL's
    real modes seen in this codebase are `:legacy_z`/`:powered_aspace`/`:z_space`/
    `:theta_decoupled_aspace`; REDUCED currently has no formalized mode symbol at all (no
    REDUCED runner file sets `A_coordinate_mode` today) -- formalizing that (task §7,
    e.g. `:profiled_log_relative_A`) is explicitly out of scope for this commit. Recording the
    field now, unvalidated, lets later runs be self-describing without this manifest module
    having to be re-released once §7 lands.
  - `nu_policy::Symbol` / `nu_bounds`: one of `VALID_NU_POLICIES`; `nu_bounds` is
    `Union{Nothing,Tuple{Float64,Float64}}`, must be `nothing` when `nu_policy===:fixed`.
  - `draw_checksum_uniform`/`draw_checksum_transformed::String`: mirrors
    `CMCheckpointV11`'s two draw-checksum fields (`full_aod_diag/d4_exact/cm_checkpoint.jl`) --
    NOT already in `ScientificManifest`, which only records `draw_design`/`draw_seed`/`W`.
  - `outer_algorithm::Symbol`, `outer_max_wall_seconds::Float64`, `outer_max_gradients::Int`:
    the outer search's algorithm identity and hard budget.
  - `cache_policy`/`dual_bank_policy`/`warm_start_policy::Symbol`: free-form identifiers for
    which cache/bank/warm-start behavior a run used; not yet cross-checked against any real
    cache implementation (REDUCED currently has none per task §9's own premise).
  - `initial_state_digest::String`: caller-supplied digest of the decoded initial economic
    state (gp/A/nu) the run started from -- see `digest_economic_state` below for the one
    canonical way to compute it, so two manifests can be compared for "same starting point"
    without re-decoding.
  - `source_sha::String`, `source_dirty::Bool`: `current_source_sha()`/`source_is_dirty()`
    below are the canonical way to fill these from the actual repository the run executed in.
"""
struct RunManifest
    schema_version::Int
    sci::ScientificManifest
    family::Symbol
    economic_parameterization::Symbol
    A_coordinate_mode::Symbol
    nu_policy::Symbol
    nu_bounds::Union{Nothing,Tuple{Float64,Float64}}
    draw_checksum_uniform::String
    draw_checksum_transformed::String
    outer_algorithm::Symbol
    outer_max_wall_seconds::Float64
    outer_max_gradients::Int
    cache_policy::Symbol
    dual_bank_policy::Symbol
    warm_start_policy::Symbol
    initial_state_digest::String
    source_sha::String
    source_dirty::Bool
end

function RunManifest(; schema_version::Int = RUN_MANIFEST_SCHEMA_VERSION, sci::ScientificManifest,
        family::Symbol, economic_parameterization::Symbol, A_coordinate_mode::Symbol,
        nu_policy::Symbol, nu_bounds::Union{Nothing,Tuple{<:Real,<:Real}},
        draw_checksum_uniform::AbstractString, draw_checksum_transformed::AbstractString,
        outer_algorithm::Symbol, outer_max_wall_seconds::Real, outer_max_gradients::Int,
        cache_policy::Symbol, dual_bank_policy::Symbol, warm_start_policy::Symbol,
        initial_state_digest::AbstractString, source_sha::AbstractString, source_dirty::Bool)
    nb = nu_bounds === nothing ? nothing : (Float64(nu_bounds[1]), Float64(nu_bounds[2]))
    RunManifest(schema_version, sci, family, economic_parameterization, A_coordinate_mode,
        nu_policy, nb, String(draw_checksum_uniform), String(draw_checksum_transformed),
        outer_algorithm, Float64(outer_max_wall_seconds), outer_max_gradients,
        cache_policy, dual_bank_policy, warm_start_policy, String(initial_state_digest),
        String(source_sha), source_dirty)
end

"""
    digest_economic_state(gp::AbstractVector, logA::AbstractMatrix, nu::AbstractVector) -> String

Canonical sha256 digest of a decoded economic state, so two runs (or a checkpoint and a fresh
decode) can be compared for "same starting point" without re-decoding. Order-sensitive and
representation-sensitive by construction (round-trips through `string(...)`, matching Julia's
own `show` for `Float64`, which is exact/lossless) -- two states that print identically digest
identically, full stop.
"""
function digest_economic_state(gp::AbstractVector{<:Real}, logA::AbstractMatrix{<:Real},
        nu::AbstractVector{<:Real})
    io = IOBuffer()
    print(io, "gp="); print(io, Float64.(gp))
    print(io, ";logA="); print(io, Float64.(logA))
    print(io, ";nu="); print(io, Float64.(nu))
    return bytes2hex(SHA.sha256(String(take!(io))))
end

"""
    current_source_sha(; repo_dir=dirname(dirname(@__DIR__))) -> String

The checked-out commit SHA of the repository containing this file, via `git rev-parse HEAD`.
Errors (does not silently return "") if not run inside a git checkout -- a manifest with no
real source SHA is not a valid manifest.
"""
function current_source_sha(; repo_dir::AbstractString = dirname(@__DIR__))
    out = read(Cmd(`git rev-parse HEAD`; dir = repo_dir), String)
    return strip(out)
end

"""
    source_is_dirty(; repo_dir=dirname(@__DIR__)) -> Bool

True iff `git status --porcelain` in `repo_dir` reports any tracked-file modification or staged
change. Untracked files alone (e.g. this run's own output directory, if not yet `.gitignore`d)
still count as dirty under plain `--porcelain` -- callers who need to exempt a known scratch
output directory should filter before calling `refuse_if_dirty`, not weaken this function.
"""
function source_is_dirty(; repo_dir::AbstractString = dirname(@__DIR__))
    out = read(Cmd(`git status --porcelain`; dir = repo_dir), String)
    return !isempty(strip(out))
end

"""
    refuse_if_dirty(; repo_dir=dirname(@__DIR__))

Throws if the worktree is dirty. Task §3's "Refuse production-like or A/B execution from a
dirty worktree" -- this is the primitive; no runner calls it yet as of this commit (task §4).
"""
function refuse_if_dirty(; repo_dir::AbstractString = dirname(@__DIR__))
    source_is_dirty(repo_dir = repo_dir) &&
        error("refuse_if_dirty: worktree at $repo_dir is dirty -- commit or stash before a " *
              "production-like or A/B run (task profiled-outer-production-readiness-2026-08-03 §3)")
    return nothing
end

function to_toml_dict(m::RunManifest)
    d = Dict{String,Any}(
        "schema_version" => m.schema_version,
        "sci" => sci_to_toml_dict(m.sci),
        "family" => String(m.family),
        "economic_parameterization" => String(m.economic_parameterization),
        "A_coordinate_mode" => String(m.A_coordinate_mode),
        "nu_policy" => String(m.nu_policy),
        "draw_checksum_uniform" => m.draw_checksum_uniform,
        "draw_checksum_transformed" => m.draw_checksum_transformed,
        "outer_algorithm" => String(m.outer_algorithm),
        "outer_max_wall_seconds" => m.outer_max_wall_seconds,
        "outer_max_gradients" => m.outer_max_gradients,
        "cache_policy" => String(m.cache_policy),
        "dual_bank_policy" => String(m.dual_bank_policy),
        "warm_start_policy" => String(m.warm_start_policy),
        "initial_state_digest" => m.initial_state_digest,
        "source_sha" => m.source_sha,
        "source_dirty" => m.source_dirty,
    )
    if m.nu_bounds !== nothing
        d["nu_bounds"] = [m.nu_bounds[1], m.nu_bounds[2]]
    end
    return d
end

function from_toml_dict(d::AbstractDict)
    req(k) = haskey(d, k) ? d[k] : error("RunManifest.from_toml_dict: missing required key \"$k\"")
    nb = haskey(d, "nu_bounds") ? (Float64(d["nu_bounds"][1]), Float64(d["nu_bounds"][2])) : nothing
    return RunManifest(
        schema_version = Int(req("schema_version")),
        sci = sci_from_toml_dict(req("sci")),
        family = Symbol(req("family")),
        economic_parameterization = Symbol(req("economic_parameterization")),
        A_coordinate_mode = Symbol(req("A_coordinate_mode")),
        nu_policy = Symbol(req("nu_policy")),
        nu_bounds = nb,
        draw_checksum_uniform = req("draw_checksum_uniform"),
        draw_checksum_transformed = req("draw_checksum_transformed"),
        outer_algorithm = Symbol(req("outer_algorithm")),
        outer_max_wall_seconds = req("outer_max_wall_seconds"),
        outer_max_gradients = Int(req("outer_max_gradients")),
        cache_policy = Symbol(req("cache_policy")),
        dual_bank_policy = Symbol(req("dual_bank_policy")),
        warm_start_policy = Symbol(req("warm_start_policy")),
        initial_state_digest = req("initial_state_digest"),
        source_sha = req("source_sha"),
        source_dirty = Bool(req("source_dirty")),
    )
end

function write_manifest_toml(path::AbstractString, m::RunManifest)
    open(path, "w") do io
        TOML.print(io, to_toml_dict(m))
    end
    return path
end

read_manifest_toml(path::AbstractString) = from_toml_dict(TOML.parsefile(path))

"""
    write_run_manifest_json(path, m::RunManifest) -> path

Writes `m` as `run_manifest.json` (task §3: "Every run must write run_manifest.json before
computation"). JSON, not TOML, to match the literal filename the task specifies; uses the same
`to_toml_dict` field set (JSON is a strict superset of what TOML.print can already serialize
here -- no separate JSON schema to keep in sync).
"""
function write_run_manifest_json(path::AbstractString, m::RunManifest)
    d = to_toml_dict(m)
    open(path, "w") do io
        _write_json(io, d)
    end
    return path
end

function _write_json(io::IO, x::AbstractDict)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        _write_json(io, String(k))
        print(io, ":")
        _write_json(io, v)
    end
    print(io, "}")
end
_write_json(io::IO, x::AbstractVector) = (print(io, "["); for (i, v) in enumerate(x); i > 1 && print(io, ","); _write_json(io, v); end; print(io, "]"))
_write_json(io::IO, x::AbstractString) = print(io, "\"", replace(String(x), "\"" => "\\\"", "\\" => "\\\\"), "\"")
_write_json(io::IO, x::Bool) = print(io, x ? "true" : "false")
_write_json(io::IO, x::Union{Int,Float64}) = print(io, x)
_write_json(io::IO, x::Tuple) = _write_json(io, collect(x))

"""
    validate_manifest(m::RunManifest; kwargs...) -> Vector{String}

Returns a list of problems (empty = valid), same non-throwing convention as
`ScientificManifest.validate_manifest`. Delegates to `ScientificManifestMod.validate_manifest`
for the wrapped `sci` field, then adds the RunManifest-specific checks. Does NOT check whether
`family`+`economic_parameterization`+`A_coordinate_mode` is an actually-implemented combination
(task §4's family registry, not yet built, is what would answer that) -- only internal
self-consistency of this manifest's own fields.
"""
function validate_manifest(m::RunManifest; data_dir::Union{Nothing,AbstractString} = nothing,
        opt_dir_inner::Union{Nothing,AbstractString} = nothing,
        opt_dir_outer::Union{Nothing,AbstractString} = nothing)
    problems = validate_sci_manifest(m.sci; data_dir = data_dir, opt_dir_inner = opt_dir_inner,
        opt_dir_outer = opt_dir_outer)
    m.schema_version == RUN_MANIFEST_SCHEMA_VERSION ||
        push!(problems, "schema_version=$(m.schema_version) != current $(RUN_MANIFEST_SCHEMA_VERSION)")
    m.family in VALID_FAMILIES ||
        push!(problems, "family=:$(m.family) not in $VALID_FAMILIES")
    m.economic_parameterization in VALID_ECONOMIC_PARAMETERIZATIONS ||
        push!(problems, "economic_parameterization=:$(m.economic_parameterization) not in $VALID_ECONOMIC_PARAMETERIZATIONS")
    m.nu_policy in VALID_NU_POLICIES ||
        push!(problems, "nu_policy=:$(m.nu_policy) not in $VALID_NU_POLICIES")
    if m.nu_policy === :fixed && m.nu_bounds !== nothing
        push!(problems, "nu_policy=:fixed but nu_bounds=$(m.nu_bounds) is set (must be nothing)")
    end
    if m.nu_policy === :free && m.nu_bounds === nothing
        push!(problems, "nu_policy=:free but nu_bounds is nothing (must be set)")
    end
    if m.nu_policy === :free && !(m.family in (:origin_zc, :cm_meanzc))
        push!(problems, "nu_policy=:free but family=:$(m.family) has no nu (only :origin_zc/:cm_meanzc do)")
    end
    m.outer_max_wall_seconds > 0 || push!(problems, "outer_max_wall_seconds=$(m.outer_max_wall_seconds) must be positive")
    m.outer_max_gradients > 0 || push!(problems, "outer_max_gradients=$(m.outer_max_gradients) must be positive")
    isempty(m.source_sha) && push!(problems, "source_sha is empty")
    isempty(m.initial_state_digest) && push!(problems, "initial_state_digest is empty")
    return problems
end

end # module RunManifestMod
