# Production-consolidation Phase 9 (2026-07-31): checkpoint code-version enforcement.
#
# Builds on Phase 8's production_launcher.jl (git state / manifest / fingerprints) to answer
# a DIFFERENT question: not "is it safe to LAUNCH a new campaign right now" (that's Phase 8's
# job), but "is it safe to RESUME an EXISTING checkpoint under the code that's running right
# now." A checkpoint written under one commit and blindly resumed under a DIFFERENT commit
# (different economic formulas, different bug fixes/regressions, different numerics) can
# silently continue a search using a stale/incompatible incumbent -- this file makes that an
# explicit, auditable decision instead of an accident.
#
# Consistent with every other file in this directory: flat top-level definitions, no module
# wrapper (see production_launcher.jl's own note on this).

using Serialization: serialize, deserialize
import Dates

struct MelitzCheckpointVersionMismatch <: Exception
    reason::String
end
Base.showerror(io::IO, e::MelitzCheckpointVersionMismatch) = print(io, "MelitzCheckpointVersionMismatch: ", e.reason)

"""
    melitz_tracked_diff_hash(repo_dir) -> String (hex sha256)

Hash of `git diff HEAD` (tracked files only) at checkpoint-write time. Distinct from
`production_launcher.jl`'s `tracked_dirty::Bool` -- that flag says WHETHER the tree was
dirty; this hash says WHAT the dirty diff actually WAS, so two checkpoints written at the
same commit but under different uncommitted local edits (e.g. two diagnostic-override runs)
are still distinguishable from each other, not just from a clean run.
"""
function melitz_tracked_diff_hash(repo_dir::AbstractString)
    diff_text = read(Cmd(String["git", "-C", repo_dir, "diff", "HEAD", "--"]), String)
    return bytes2hex(sha256(codeunits(diff_text)))
end

"""
    MelitzCheckpointVersionInfo

Everything Phase 9 requires every checkpoint to record: exact source commit, a hash of any
uncommitted tracked diff, the full Phase-8 production manifest (git state, toolchain
versions, source locations), and data/options fingerprints (duplicated from the manifest for
convenience -- checkpoints are read far more often in isolation than manifests are).
"""
Base.@kwdef struct MelitzCheckpointVersionInfo
    saved_at::String
    commit_sha::String
    tracked_diff_hash::String
    production_manifest::MelitzProductionManifest
    data_fingerprints::Dict{String,String}
    options_fingerprints::Dict{String,String}
end

"""
    melitz_build_checkpoint_version_info(repo_dir; allow_dirty=false, knitro_module=nothing,
        melitz_ccbundle_instance=nothing, real_data_dir=nothing, option_files=String[]) -> MelitzCheckpointVersionInfo

Runs the Phase-8 preflight (so a checkpoint can never be WRITTEN by a launch that would
itself have been refused, unless the same diagnostic override is explicitly used) and
packages the result together with a tracked-diff hash and duplicated fingerprints into one
`MelitzCheckpointVersionInfo`, ready to attach to a checkpoint payload.
"""
function melitz_build_checkpoint_version_info(repo_dir::AbstractString; allow_dirty::Bool=false,
        knitro_module=nothing, melitz_ccbundle_instance=nothing,
        real_data_dir::Union{Nothing,AbstractString}=nothing,
        option_files::AbstractVector{<:AbstractString}=String[])
    manifest = melitz_production_preflight!(repo_dir; allow_dirty=allow_dirty, knitro_module=knitro_module,
        melitz_ccbundle_instance=melitz_ccbundle_instance, real_data_dir=real_data_dir, option_files=option_files)
    return MelitzCheckpointVersionInfo(;
        saved_at=string(Dates.now()), commit_sha=manifest.commit_sha,
        tracked_diff_hash=melitz_tracked_diff_hash(repo_dir), production_manifest=manifest,
        data_fingerprints=manifest.data_fingerprints, options_fingerprints=manifest.options_fingerprints)
end

"""
    MelitzCheckpoint{T}

Generic checkpoint envelope: `version_info` (this file's own bookkeeping) plus an arbitrary
caller-defined `payload` (e.g. a campaign's own incumbent/state struct -- ContinuationPoint,
MelitzExpandedState, whatever a given campaign script checkpoints).
"""
struct MelitzCheckpoint{T}
    version_info::MelitzCheckpointVersionInfo
    payload::T
end

"""
    melitz_save_checkpoint!(path, payload, repo_dir; kwargs...) -> MelitzCheckpoint

Builds a fresh `MelitzCheckpointVersionInfo` (running the Phase-8 preflight) and serializes
`MelitzCheckpoint(version_info, payload)` to `path`. `kwargs` are forwarded to
`melitz_build_checkpoint_version_info` (allow_dirty, knitro_module, real_data_dir,
option_files, ...).
"""
function melitz_save_checkpoint!(path::AbstractString, payload, repo_dir::AbstractString; kwargs...)
    version_info = melitz_build_checkpoint_version_info(repo_dir; kwargs...)
    ckpt = MelitzCheckpoint(version_info, payload)
    mkpath(dirname(path))
    serialize(path, ckpt)
    return ckpt
end

"""
    melitz_resume_checkpoint(path, repo_dir; migrate::Bool=false, cold_reverify=nothing) -> (payload, info)

Phase 9's resume gate. `path` must have been written by `melitz_save_checkpoint!` (or contain
a serialized `MelitzCheckpoint`). Behavior:

  - **Same commit** as current `repo_dir` HEAD: returns `(payload, checkpoint.version_info)`
    silently (no warning, no special handling) -- the default, expected case.
  - **Different commit**, `migrate=false` (default): throws `MelitzCheckpointVersionMismatch`
    -- resuming under different code without an explicit decision is refused.
  - **Different commit**, `migrate=true`: requires a `cold_reverify` callback,
    `cold_reverify(payload) -> Bool`. Calls it (under the CURRENT code) to confirm the saved
    incumbent is still genuinely valid before continuing. If `cold_reverify` returns `false`
    (or throws), migration FAILS (throws `MelitzCheckpointVersionMismatch`, does not silently
    fall back to trusting the stale checkpoint). If it returns `true`, returns
    `(payload, migration_info)` where `migration_info` is a NamedTuple recording BOTH the old
    (`checkpoint.version_info.commit_sha`) and new (current HEAD) commits, plus the
    cold-reverification outcome and timestamp -- callers should persist this migration record
    (e.g. re-save the checkpoint under the new commit) rather than discard it.
"""
function melitz_resume_checkpoint(path::AbstractString, repo_dir::AbstractString; migrate::Bool=false,
        cold_reverify::Union{Nothing,Function}=nothing)
    isfile(path) || error("melitz_resume_checkpoint: no checkpoint at $path")
    ckpt = deserialize(path)
    ckpt isa MelitzCheckpoint || error("melitz_resume_checkpoint: $path did not deserialize to a MelitzCheckpoint " *
        "(got $(typeof(ckpt))) -- wrong file, or written by something other than melitz_save_checkpoint!")

    current_commit = strip(read(Cmd(String["git", "-C", repo_dir, "rev-parse", "HEAD"]), String))
    saved_commit = ckpt.version_info.commit_sha

    if current_commit == saved_commit
        return ckpt.payload, ckpt.version_info
    end

    if !migrate
        throw(MelitzCheckpointVersionMismatch(
            "checkpoint at $path was saved under commit $saved_commit, but the current " *
            "commit is $current_commit -- refusing to resume under different code without " *
            "migrate=true. If this is a deliberate code update mid-campaign, pass " *
            "migrate=true with a cold_reverify callback that re-verifies the saved incumbent " *
            "under the CURRENT code before continuing."))
    end

    cold_reverify !== nothing || throw(MelitzCheckpointVersionMismatch(
        "migrate=true requires a cold_reverify::Function callback (cold_reverify(payload)::Bool) " *
        "-- refusing to blindly trust a checkpoint saved under a different commit ($saved_commit " *
        "-> $current_commit) with no re-verification."))

    verified = try
        cold_reverify(ckpt.payload)
    catch e
        throw(MelitzCheckpointVersionMismatch(
            "migration cold_reverify callback THREW while re-verifying the checkpoint saved " *
            "under commit $saved_commit against current commit $current_commit: " *
            "$(sprint(showerror, e)) -- migration refused, not silently trusted."))
    end
    verified isa Bool || throw(MelitzCheckpointVersionMismatch(
        "migration cold_reverify callback must return a Bool, got $(typeof(verified)) -- migration refused."))
    verified || throw(MelitzCheckpointVersionMismatch(
        "migration cold_reverify callback returned false for the checkpoint saved under commit " *
        "$saved_commit against current commit $current_commit -- the saved incumbent did NOT " *
        "reverify under current code; migration refused (this is exactly the case this gate " *
        "exists to catch, not a false alarm to work around)."))

    migration_info = (old_commit=saved_commit, new_commit=current_commit,
        cold_reverified=true, migrated_at=string(Dates.now()))
    println("Checkpoint migration: $saved_commit -> $current_commit, cold-reverification PASSED.")
    return ckpt.payload, migration_info
end
