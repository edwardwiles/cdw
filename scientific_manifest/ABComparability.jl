module ABComparabilityMod
# ABComparability -- task profiled-outer-production-readiness-2026-08-03 section 10: the hard
# A/B comparability gate. "It must fail unless all non-formulation fields match ... Every A/B
# output must contain AB_COMPARABLE=true, manifest hashes, source SHA, decoded-state equivalence
# result. The report generator must refuse to summarize an invalid pair."
#
# Pure logic over two RunManifest instances -- no KNITRO, no full_aod_diag dependency, so this is
# fast and independently testable (same discipline as ScientificManifest/RunManifest/
# FamilyRegistry). Deliberately separate from Melitz.

using SHA

include(joinpath(@__DIR__, "RunManifest.jl"))
using .RunManifestMod: RunManifest

export ABComparabilityResult, ab_comparable, manifest_digest, require_ab_comparable

"""
    manifest_digest(m::RunManifest) -> String

Deterministic sha256 hex digest of `m`'s own `to_toml_dict` (field-sorted, so digest depends only
on values, not `Dict` iteration order). "Manifest hashes" -- one of the four things task §10
requires every A/B output to contain.
"""
function manifest_digest(m::RunManifest)
    d = RunManifestMod.to_toml_dict(m)
    io = IOBuffer()
    for k in sort(collect(keys(d)))
        print(io, k, "="); print(io, d[k]); print(io, ";")
    end
    return bytes2hex(sha256(String(take!(io))))
end

"""
    ABComparabilityResult

`comparable` mirrors `isempty(problems)`. `coordinate_mode_differs` and
`coordinate_mode_experiment_label` record whether/why an `A_coordinate_mode` difference was
allowed through (task §10: "Coordinate modes may differ only when ... the report labels the
coordinate difference"). `full_manifest_digest`/`reduced_manifest_digest`/`full_source_sha`/
`reduced_source_sha`/`decoded_state_equivalent` are the four required-content fields task §10
lists for every A/B output.
"""
struct ABComparabilityResult
    comparable::Bool
    problems::Vector{String}
    coordinate_mode_differs::Bool
    coordinate_mode_experiment_label::Union{Nothing,String}
    full_manifest_digest::String
    reduced_manifest_digest::String
    full_source_sha::String
    reduced_source_sha::String
    decoded_state_equivalent::Bool
end

"""
    ab_comparable(full::RunManifest, reduced::RunManifest;
        allow_coordinate_mode_diff::Bool=false,
        coordinate_mode_experiment_label::Union{Nothing,String}=nothing) -> ABComparabilityResult

Never throws (same non-throwing convention as `ScientificManifest`/`RunManifest`'s own
`validate_manifest`) -- returns a result whose `.comparable`/`.problems` the caller inspects.
Use `require_ab_comparable` for a throwing wrapper.

Checks (task §10's own required-fields list, mapped onto `RunManifest`'s real fields):
- `full` really is the FULL arm and `reduced` really is the REDUCED arm (economic_parameterization).
- `family` matches.
- Every `ScientificManifest` field EXCEPT nothing (all of them: dataset version/checksum,
  country order, focal country, sigma, gravity mask, ROW treatment, draw design/seed, W, L/K,
  threads, inner/outer option checksums).
- `draw_checksum_uniform`/`draw_checksum_transformed`, `nu_policy`/`nu_bounds`, `outer_algorithm`/
  `outer_max_wall_seconds`/`outer_max_gradients`, `cache_policy`/`dual_bank_policy`/
  `warm_start_policy`/`verification_policy`.
- `initial_state_digest` -- task §10's "initial decoded economic state" requirement; this is also
  what makes `decoded_state_equivalent` in the result meaningful (it IS that comparison).
- `source_sha` -- both arms should come from the same checkout of the same repo state for a
  claim of a coherent, simultaneous A/B; a genuine cross-session comparison against a different
  commit is exactly the kind of thing that should be a labeled exception, not silently passed.
- `A_coordinate_mode` is the ONE field allowed to differ, and only when `allow_coordinate_mode_diff=true`
  AND a non-nothing `coordinate_mode_experiment_label` is given (task §10: "the difference is
  explicitly declared as an experiment ... the report labels the coordinate difference"). This
  function does NOT verify "both reconstruct the intended same economic point" for that case --
  that is exactly what `initial_state_digest` equality (checked unconditionally, above) already
  proves when it passes, and this function does not weaken that check for a coordinate-mode
  experiment.
"""
function ab_comparable(full::RunManifest, reduced::RunManifest;
        allow_coordinate_mode_diff::Bool = false,
        coordinate_mode_experiment_label::Union{Nothing,String} = nothing)
    problems = String[]

    full.economic_parameterization == :full_gamma_normalized ||
        push!(problems, "full manifest's economic_parameterization is :$(full.economic_parameterization), expected :full_gamma_normalized -- arms passed in the wrong order?")
    reduced.economic_parameterization == :profiled_destination_scales ||
        push!(problems, "reduced manifest's economic_parameterization is :$(reduced.economic_parameterization), expected :profiled_destination_scales -- arms passed in the wrong order?")

    full.family == reduced.family ||
        push!(problems, "family differs: full=:$(full.family), reduced=:$(reduced.family)")

    fs, rs = full.sci, reduced.sci
    fs.dataset_version == rs.dataset_version || push!(problems, "sci.dataset_version differs: full=$(fs.dataset_version), reduced=$(rs.dataset_version)")
    fs.dataset_checksum == rs.dataset_checksum || push!(problems, "sci.dataset_checksum differs: full=$(fs.dataset_checksum), reduced=$(rs.dataset_checksum)")
    fs.country_order == rs.country_order || push!(problems, "sci.country_order differs")
    fs.focal_country == rs.focal_country || push!(problems, "sci.focal_country differs: full=$(fs.focal_country), reduced=$(rs.focal_country)")
    fs.sigma == rs.sigma || push!(problems, "sci.sigma differs: full=$(fs.sigma), reduced=$(rs.sigma)")
    fs.exclude_diagonal_gravity == rs.exclude_diagonal_gravity || push!(problems, "sci.exclude_diagonal_gravity differs: full=$(fs.exclude_diagonal_gravity), reduced=$(rs.exclude_diagonal_gravity)")
    fs.gravity_exclude_cells == rs.gravity_exclude_cells || push!(problems, "sci.gravity_exclude_cells differs")
    fs.destination_sample == rs.destination_sample || push!(problems, "sci.destination_sample differs: full=:$(fs.destination_sample), reduced=:$(rs.destination_sample)")
    fs.draw_design == rs.draw_design || push!(problems, "sci.draw_design differs: full=:$(fs.draw_design), reduced=:$(rs.draw_design)")
    fs.draw_seed == rs.draw_seed || push!(problems, "sci.draw_seed differs: full=$(fs.draw_seed), reduced=$(rs.draw_seed)")
    fs.W == rs.W || push!(problems, "sci.W differs: full=$(fs.W), reduced=$(rs.W)")
    fs.L == rs.L || push!(problems, "sci.L differs: full=$(fs.L), reduced=$(rs.L)")
    fs.K_mean == rs.K_mean || push!(problems, "sci.K_mean differs: full=$(fs.K_mean), reduced=$(rs.K_mean)")
    fs.K_pair == rs.K_pair || push!(problems, "sci.K_pair differs: full=$(fs.K_pair), reduced=$(rs.K_pair)")
    fs.julia_threads == rs.julia_threads || push!(problems, "sci.julia_threads differs: full=$(fs.julia_threads), reduced=$(rs.julia_threads)")
    fs.blas_threads == rs.blas_threads || push!(problems, "sci.blas_threads differs: full=$(fs.blas_threads), reduced=$(rs.blas_threads)")
    fs.inner_opt_checksum == rs.inner_opt_checksum || push!(problems, "sci.inner_opt_checksum differs (different KNITRO inner solver options)")
    fs.outer_opt_checksum == rs.outer_opt_checksum || push!(problems, "sci.outer_opt_checksum differs (different KNITRO outer solver options)")

    full.draw_checksum_uniform == reduced.draw_checksum_uniform || push!(problems, "draw_checksum_uniform differs -- same draw_design/draw_seed produced different draws")
    full.draw_checksum_transformed == reduced.draw_checksum_transformed || push!(problems, "draw_checksum_transformed differs -- same draw_design/draw_seed produced different draws")
    full.nu_policy == reduced.nu_policy || push!(problems, "nu_policy differs: full=:$(full.nu_policy), reduced=:$(reduced.nu_policy)")
    full.nu_bounds == reduced.nu_bounds || push!(problems, "nu_bounds differs: full=$(full.nu_bounds), reduced=$(reduced.nu_bounds)")
    full.outer_algorithm == reduced.outer_algorithm || push!(problems, "outer_algorithm differs: full=:$(full.outer_algorithm), reduced=:$(reduced.outer_algorithm)")
    full.outer_max_wall_seconds == reduced.outer_max_wall_seconds || push!(problems, "outer_max_wall_seconds differs: full=$(full.outer_max_wall_seconds), reduced=$(reduced.outer_max_wall_seconds)")
    full.outer_max_gradients == reduced.outer_max_gradients || push!(problems, "outer_max_gradients differs: full=$(full.outer_max_gradients), reduced=$(reduced.outer_max_gradients)")
    full.cache_policy == reduced.cache_policy || push!(problems, "cache_policy differs: full=:$(full.cache_policy), reduced=:$(reduced.cache_policy)")
    full.dual_bank_policy == reduced.dual_bank_policy || push!(problems, "dual_bank_policy differs: full=:$(full.dual_bank_policy), reduced=:$(reduced.dual_bank_policy)")
    full.warm_start_policy == reduced.warm_start_policy || push!(problems, "warm_start_policy differs: full=:$(full.warm_start_policy), reduced=:$(reduced.warm_start_policy)")
    full.verification_policy == reduced.verification_policy || push!(problems, "verification_policy differs: full=:$(full.verification_policy), reduced=:$(reduced.verification_policy)")

    decoded_equiv = full.initial_state_digest == reduced.initial_state_digest
    decoded_equiv || push!(problems, "initial_state_digest differs -- the two arms did not start from the same decoded economic state (gp/A/nu)")

    full.source_sha == reduced.source_sha ||
        push!(problems, "source_sha differs: full=$(full.source_sha), reduced=$(reduced.source_sha) -- arms were built from different repository states")
    full.source_dirty && push!(problems, "full manifest's source_dirty=true -- refusing to certify comparability from a dirty worktree")
    reduced.source_dirty && push!(problems, "reduced manifest's source_dirty=true -- refusing to certify comparability from a dirty worktree")

    coord_differs = full.A_coordinate_mode != reduced.A_coordinate_mode
    if coord_differs
        if !allow_coordinate_mode_diff
            push!(problems, "A_coordinate_mode differs (full=:$(full.A_coordinate_mode), reduced=:$(reduced.A_coordinate_mode)) and allow_coordinate_mode_diff=false -- pass allow_coordinate_mode_diff=true only for an explicitly declared, labeled coordinate-mode experiment")
        elseif coordinate_mode_experiment_label === nothing
            push!(problems, "A_coordinate_mode differs and allow_coordinate_mode_diff=true, but coordinate_mode_experiment_label is nothing -- the report must label the coordinate difference")
        end
    end

    return ABComparabilityResult(isempty(problems), problems, coord_differs, coordinate_mode_experiment_label,
        manifest_digest(full), manifest_digest(reduced), full.source_sha, reduced.source_sha, decoded_equiv)
end

"""
    require_ab_comparable(full, reduced; kwargs...) -> ABComparabilityResult

Throws with every problem listed (not just the first) if `ab_comparable(...)` is not comparable.
"The report generator must refuse to summarize an invalid pair" (task §10) -- this is that refusal.
"""
function require_ab_comparable(full::RunManifest, reduced::RunManifest; kwargs...)
    r = ab_comparable(full, reduced; kwargs...)
    r.comparable ||
        error("require_ab_comparable: NOT comparable --\n  " * join(r.problems, "\n  "))
    return r
end

end # module ABComparabilityMod
