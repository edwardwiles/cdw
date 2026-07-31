# ============================================================================
# Country name/ISO3 -> panel-index resolver (2026-07-31, Brazil-Korea gravity-exclusion task).
#
# No such resolver existed anywhere in the repo before this -- the only precedent was an inline
# `findfirst(==("fra"), countries)` (campaign_inputs/sigma3_W500k_2026-07-30/build_and_validate_snapshot.jl).
# This generalizes that idiom with an explicit, audited alias table (so callers never hardcode
# numeric indices) and hard failure on ambiguity or absence, per this task's requirement.
# ============================================================================

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
