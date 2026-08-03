module ScientificManifestMod
# ScientificManifest -- one typed, versioned, serializable record of the scientific settings
# for the EK/Ricardo full-A_od gravity-elimination model (FULL production / REDUCED prototype).
#
# Deliberately separate from Melitz: this module has zero `using`/`include` of anything under
# src/melitz/, scripts/melitz_*, melitz/, or test/melitz/, and nothing in those paths should
# ever `include` or `using` this module. The two models are architecturally unrelated and are
# kept that way here on purpose (user directive, 2026-08-03).
#
# This is Phase 4 of REPO_CLEANUP_MASTER_2026-08-03: "one source of scientific truth." It does
# NOT yet change any existing runner's behavior -- no existing driver requires a manifest yet
# (that wiring is Phase 6/7, not done in this pass). This module only defines the type, TOML
# round-trip, and a validation check that a manifest is well-formed and internally consistent.

using TOML
using SHA

export ScientificManifest, to_toml_dict, from_toml_dict, write_manifest_toml, read_manifest_toml,
       validate_manifest, sha256_of_file, sha256_of_directory

const CURRENT_SCHEMA_VERSION = 1

"""
    ScientificManifest

One immutable, serializable record of every scientific (as opposed to execution/performance)
setting that determines what economic problem a FULL or REDUCED run actually solves.

Field grounding (2026-08-03, all read directly from the live codebase, not guessed):
  - `focal_country`, `country_order`: `real_data/noah_D20/countries.csv` (20 countries,
    3-letter lowercase codes, "row" = rest-of-world, always last).
  - `sigma`: the production default was silently 2.5 (`AD_PARAMS.σHat`,
    `full_aod_diag/d4_exact/context_real_d20.jl`'s `σHat::Union{Nothing,Float64}=nothing` kwarg)
    until the 2026-08-01 Brazil-Korea defaults flip explicitly set it to 3.0 in three real
    drivers. `d20_real_setup`/`build_ad_context_real_d20` THEMSELVES still default to `nothing`
    (-> 2.5) unless a caller passes `σHat` explicitly -- confirmed live during this session that
    `campaign_cm_family_runner.jl` (the runner behind the live FULL W100k 10x10 campaign) does
    NOT pass `σHat` at its `run_*_checkpointed` call sites. This is exactly the silent-default
    risk this manifest type exists to close; do not assume every current caller already gets
    sigma=3.0 without independently checking that call site.
  - `exclude_diagonal_gravity`, `gravity_exclude_cells`: `default_gravity_exclude_cells_brazil_korea`
    (`country_resolve.jl`) resolves Brazil/Korea from the real country list; only defined/valid
    for `destination_sample=:exclude_row`.
  - `destination_sample`: `d20_real_setup`'s own default and only non-legacy option.
  - `draw_design`, `draw_seed`, `W`, `L` (`CM_L`), `K_mean`/`K_pair`
    (`MEANZC_K`/`ORIGINZC_K`): `campaign_cm_family_runner.jl` constants, the actual live
    10x10 campaign's settings.
  - `inner_opt_checksum`/`outer_opt_checksum`: sha256 of the actual `.opt` files
    `d20_real_setup` defaults to (`full_aod_diag/csw_outer_25.opt`, `full_aod_diag/ek_inner.opt`).
  - `dataset_checksum`: sha256-of-concatenated-sha256sums of every file under
    `real_data/noah_D20/` (see `sha256_of_directory`) -- NOT a checksum of any single file.

Fields intentionally NOT included, and why: the task sketch that motivated this module included
a scalar `gravity_theta::Float64`. There is no such scalar in this codebase -- theta is a
high-dimensional vector (`θ0_up`, length `3+D+...`) that is estimated from data inside
`d20_real_setup`, not supplied as a manifest input. Including a fake scalar field would be
worse than omitting it. `nu_policy` is similarly not included yet: nu (`cm_meanzc_nu`/
`origin_zc_nu`) currently comes from a pre-built 5-start manifest JSON
(`COMMON_FIVE_STARTS_MANIFEST_2026-07-28.json`), not from this struct -- wiring that in is
future work, not invented here.
"""
struct ScientificManifest
    schema_version::Int
    dataset_version::String
    dataset_checksum::String
    country_order::Vector{String}
    focal_country::String
    sigma::Float64
    exclude_diagonal_gravity::Bool
    gravity_exclude_cells::Vector{Tuple{Int,Int}}
    destination_sample::Symbol
    draw_design::Symbol
    draw_seed::Int
    W::Int
    L::Int
    K_mean::Int
    K_pair::Int
    julia_threads::Int
    blas_threads::Int
    inner_opt_checksum::String
    outer_opt_checksum::String
end

function ScientificManifest(; schema_version::Int = CURRENT_SCHEMA_VERSION,
        dataset_version::AbstractString, dataset_checksum::AbstractString,
        country_order::AbstractVector{<:AbstractString}, focal_country::AbstractString,
        sigma::Real, exclude_diagonal_gravity::Bool,
        gravity_exclude_cells::AbstractVector{<:Tuple{Int,Int}} = Tuple{Int,Int}[],
        destination_sample::Symbol, draw_design::Symbol, draw_seed::Int,
        W::Int, L::Int, K_mean::Int, K_pair::Int,
        julia_threads::Int, blas_threads::Int,
        inner_opt_checksum::AbstractString, outer_opt_checksum::AbstractString)
    ScientificManifest(schema_version, String(dataset_version), String(dataset_checksum),
        String.(country_order), String(focal_country), Float64(sigma), exclude_diagonal_gravity,
        collect(gravity_exclude_cells), destination_sample, draw_design, draw_seed,
        W, L, K_mean, K_pair, julia_threads, blas_threads,
        String(inner_opt_checksum), String(outer_opt_checksum))
end

"""
    sha256_of_file(path) -> String

Lowercase-hex sha256 of a single file's raw bytes.
"""
function sha256_of_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

"""
    sha256_of_directory(dir) -> String

Deterministic (sorted-path) sha256-of-concatenated-per-file-sha256sums, one line per file as
`"<sha256>  <relative_path>\\n"`, matching the shell convention `find ... | sort | xargs
sha256sum | sha256sum`. Two directories with identical file contents at identical relative
paths give the identical result regardless of directory mtimes/permissions/traversal order.
"""
function sha256_of_directory(dir::AbstractString)
    files = String[]
    for (root, _, fs) in walkdir(dir)
        for f in fs
            push!(files, relpath(joinpath(root, f), dir))
        end
    end
    sort!(files)
    lines = IOBuffer()
    for f in files
        h = sha256_of_file(joinpath(dir, f))
        print(lines, h, "  ", f, "\n")
    end
    return bytes2hex(SHA.sha256(String(take!(lines))))
end

_cells_to_toml(cells::Vector{Tuple{Int,Int}}) = [[c[1], c[2]] for c in cells]
_cells_from_toml(v) = Tuple{Int,Int}[(Int(c[1]), Int(c[2])) for c in v]

"""
    to_toml_dict(m::ScientificManifest) -> Dict

Serializes every field. Symbols become their `String` name (round-tripped back to `Symbol` by
`from_toml_dict`); tuples become 2-element arrays (TOML has no tuple type).
"""
function to_toml_dict(m::ScientificManifest)
    return Dict{String,Any}(
        "schema_version" => m.schema_version,
        "dataset_version" => m.dataset_version,
        "dataset_checksum" => m.dataset_checksum,
        "country_order" => m.country_order,
        "focal_country" => m.focal_country,
        "sigma" => m.sigma,
        "exclude_diagonal_gravity" => m.exclude_diagonal_gravity,
        "gravity_exclude_cells" => _cells_to_toml(m.gravity_exclude_cells),
        "destination_sample" => String(m.destination_sample),
        "draw_design" => String(m.draw_design),
        "draw_seed" => m.draw_seed,
        "W" => m.W,
        "L" => m.L,
        "K_mean" => m.K_mean,
        "K_pair" => m.K_pair,
        "julia_threads" => m.julia_threads,
        "blas_threads" => m.blas_threads,
        "inner_opt_checksum" => m.inner_opt_checksum,
        "outer_opt_checksum" => m.outer_opt_checksum,
    )
end

"""
    from_toml_dict(d::Dict) -> ScientificManifest

Inverse of `to_toml_dict`. Errors (does not silently default) if a required key is missing --
a manifest with a missing field is not a valid manifest.
"""
function from_toml_dict(d::AbstractDict)
    req(k) = haskey(d, k) ? d[k] : error("ScientificManifest.from_toml_dict: missing required key \"$k\"")
    return ScientificManifest(
        schema_version = Int(req("schema_version")),
        dataset_version = req("dataset_version"),
        dataset_checksum = req("dataset_checksum"),
        country_order = req("country_order"),
        focal_country = req("focal_country"),
        sigma = req("sigma"),
        exclude_diagonal_gravity = req("exclude_diagonal_gravity"),
        gravity_exclude_cells = _cells_from_toml(get(d, "gravity_exclude_cells", [])),
        destination_sample = Symbol(req("destination_sample")),
        draw_design = Symbol(req("draw_design")),
        draw_seed = Int(req("draw_seed")),
        W = Int(req("W")),
        L = Int(req("L")),
        K_mean = Int(req("K_mean")),
        K_pair = Int(req("K_pair")),
        julia_threads = Int(req("julia_threads")),
        blas_threads = Int(req("blas_threads")),
        inner_opt_checksum = req("inner_opt_checksum"),
        outer_opt_checksum = req("outer_opt_checksum"),
    )
end

function write_manifest_toml(path::AbstractString, m::ScientificManifest)
    open(path, "w") do io
        TOML.print(io, to_toml_dict(m))
    end
    return path
end

function read_manifest_toml(path::AbstractString)
    return from_toml_dict(TOML.parsefile(path))
end

"""
    validate_manifest(m::ScientificManifest; data_dir=nothing, opt_dir=nothing) -> Vector{String}

Returns a list of problems (empty = valid). Does NOT throw, so a caller (e.g. a future dirty-tree
/ preflight gate) can decide whether to error or just report. If `data_dir`/`opt_dir` are given,
also re-derives the dataset/opt checksums from disk and flags a mismatch -- this is what catches
"the manifest claims dataset X but the actual data on disk is not X."
"""
function validate_manifest(m::ScientificManifest; data_dir::Union{Nothing,AbstractString} = nothing,
        opt_dir_inner::Union{Nothing,AbstractString} = nothing,
        opt_dir_outer::Union{Nothing,AbstractString} = nothing)
    problems = String[]
    m.schema_version == CURRENT_SCHEMA_VERSION ||
        push!(problems, "schema_version=$(m.schema_version) != current $(CURRENT_SCHEMA_VERSION)")
    m.destination_sample in (:exclude_row, :all_legacy) ||
        push!(problems, "destination_sample must be :exclude_row or :all_legacy, got :$(m.destination_sample)")
    isempty(m.country_order) && push!(problems, "country_order is empty")
    m.focal_country in m.country_order ||
        push!(problems, "focal_country=\"$(m.focal_country)\" not present in country_order")
    m.sigma > 0 || push!(problems, "sigma=$(m.sigma) must be positive")
    m.W > 0 || push!(problems, "W=$(m.W) must be positive")
    m.L > 0 || push!(problems, "L=$(m.L) must be positive")
    m.K_mean > 0 || push!(problems, "K_mean=$(m.K_mean) must be positive")
    m.K_pair > 0 || push!(problems, "K_pair=$(m.K_pair) must be positive")
    m.julia_threads > 0 || push!(problems, "julia_threads=$(m.julia_threads) must be positive")
    m.blas_threads > 0 || push!(problems, "blas_threads=$(m.blas_threads) must be positive")
    if !isempty(m.gravity_exclude_cells) && m.destination_sample != :exclude_row
        push!(problems, "gravity_exclude_cells is non-empty but destination_sample is not :exclude_row " *
                         "(default_gravity_exclude_cells_brazil_korea only supports :exclude_row)")
    end
    if data_dir !== nothing
        actual = sha256_of_directory(data_dir)
        actual == m.dataset_checksum ||
            push!(problems, "dataset_checksum mismatch: manifest says $(m.dataset_checksum), " *
                             "actual $(data_dir) is $(actual)")
    end
    if opt_dir_inner !== nothing
        actual = sha256_of_file(opt_dir_inner)
        actual == m.inner_opt_checksum ||
            push!(problems, "inner_opt_checksum mismatch: manifest says $(m.inner_opt_checksum), " *
                             "actual $(opt_dir_inner) is $(actual)")
    end
    if opt_dir_outer !== nothing
        actual = sha256_of_file(opt_dir_outer)
        actual == m.outer_opt_checksum ||
            push!(problems, "outer_opt_checksum mismatch: manifest says $(m.outer_opt_checksum), " *
                             "actual $(opt_dir_outer) is $(actual)")
    end
    return problems
end

end # module ScientificManifestMod
