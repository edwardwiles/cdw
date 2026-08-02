# ============================================================================
# Country name/ISO3 -> panel-index resolver (2026-07-31, Brazil-Korea gravity-exclusion task).
#
# No such resolver existed anywhere in the repo before this -- the only precedent was an inline
# `findfirst(==("fra"), countries)` (campaign_inputs/sigma3_W500k_2026-07-30/build_and_validate_snapshot.jl).
# This generalizes that idiom with an explicit, audited alias table (so callers never hardcode
# numeric indices) and hard failure on ambiguity or absence, per this task's requirement.
# ============================================================================

using DelimitedFiles: readdlm

"""
Audited alias table: lowercased free-form name/label -> canonical ISO3 code used in
`real_data/noah_D20/countries.csv`. Only entries actually needed by this task are listed; extend
deliberately (not speculatively) if another caller needs another country's aliases.
"""
const COUNTRY_ALIASES = Dict(
    "brazil" => "bra",
    "bra" => "bra",
    "korea" => "kor",
    "south korea" => "kor",
    "korea, rep." => "kor",
    "republic of korea" => "kor",
    "kor" => "kor",
)

"""
    resolve_country_index(countries::AbstractVector{<:AbstractString}, name_or_alias::AbstractString) -> Int

Resolve a free-form country name or ISO3 code to its 1-based row/column index in `countries`
(e.g. `real_data/noah_D20/countries.csv`'s row order, which is the row/column convention every
other data matrix in this economy uses). Looks up `name_or_alias` (case/whitespace-insensitive) in
`COUNTRY_ALIASES` first; if absent from the table, falls back to treating it as a literal ISO3 code
already. Throws `ErrorException` (not a silent `nothing`/`0`) if the resolved ISO3 code does not
appear in `countries` exactly once -- ambiguous or absent resolution is a hard error, never a
best-guess index.
"""
function resolve_country_index(countries::AbstractVector{<:AbstractString}, name_or_alias::AbstractString)
    key = lowercase(strip(name_or_alias))
    iso3 = get(COUNTRY_ALIASES, key, key)
    matches = findall(==(iso3), countries)
    length(matches) == 1 ||
        error("resolve_country_index($(repr(name_or_alias))): expected exactly one match for " *
              "iso3='$iso3' in countries, found $(length(matches))")
    return matches[1]
end

"""
    global_to_dest_slot(global_idx::Int, named_dest::AbstractVector{Int}) -> Int

Map a country's GLOBAL panel index to its destination-SLOT (column) index within a `D x Ddest`
gravity-sample matrix under a given `named_dest` (the same vector `master_prestep.jl`/
`context_real_d20.jl` build under `destination_sample`). Errors if `global_idx` is not itself a
valid destination under `named_dest` (e.g. it IS the omitted ROW destination).
"""
function global_to_dest_slot(global_idx::Int, named_dest::AbstractVector{Int})
    slot = findfirst(==(global_idx), named_dest)
    slot === nothing &&
        error("global_to_dest_slot: global index $global_idx is not a valid destination under " *
              "named_dest=$named_dest (e.g. it may be the omitted ROW destination)")
    return slot
end

"""
    default_gravity_exclude_cells_brazil_korea(; destination_sample::Symbol = :exclude_row,
        data_dir::AbstractString = ...) -> Vector{Tuple{Int,Int}}

The production default for `gravity_exclude_cells` (2026-08-01 default flip, following the
Brazil-Korea gravity-exclusion release). Resolves Brazil's origin index and Korea's
DESTINATION-SLOT index fresh from `countries.csv` every call -- deliberately NOT a hardcoded
`[(3, 14)]` literal, since the destination-slot encoding is coupled to `destination_sample`
(a literal index would silently be wrong under `:all_legacy` or a different data snapshot).
`destination_sample` must be `:exclude_row` (production's own default and the only value this
release was validated under) -- any other value throws rather than silently resolving a slot
index under the wrong convention.
"""
function default_gravity_exclude_cells_brazil_korea(; destination_sample::Symbol = :exclude_row,
        data_dir::AbstractString = get(ENV, "REAL_DATA_DIR", normpath(joinpath(@__DIR__, "..", "..", "real_data", "noah_D20"))))
    destination_sample == :exclude_row ||
        error("default_gravity_exclude_cells_brazil_korea: only destination_sample=:exclude_row is " *
              "supported (got :$destination_sample) -- the Brazil->Korea destination-slot index was " *
              "only ever resolved/validated under :exclude_row; pass gravity_exclude_cells explicitly " *
              "if you need a different destination_sample.")
    countries = vec(readdlm(joinpath(data_dir, "countries.csv"), ',', String))
    bra_idx = resolve_country_index(countries, "Brazil")
    kor_idx = resolve_country_index(countries, "Korea")
    row_idx = findfirst(==("row"), countries)
    row_idx === nothing && error("default_gravity_exclude_cells_brazil_korea: no 'row' entry in $data_dir/countries.csv")
    named_dest = filter(!=(row_idx), 1:length(countries))
    kor_slot = global_to_dest_slot(kor_idx, named_dest)
    return [(bra_idx, kor_slot)]
end
