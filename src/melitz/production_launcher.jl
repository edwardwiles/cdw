# Production-consolidation Phase 8 (2026-07-31): the ONE canonical public entry point every
# Melitz production campaign script should call before doing any real work. Distinct from
# `production_structural_gate.jl` (which asserts the RUNTIME OBJECT -- MelitzCCBundle,
# no dense fallback -- is the production-fast type): this file asserts the SOURCE/PROCESS
# state (git ancestry, tree cleanliness, toolchain versions, fingerprints) is what it claims
# to be, and packages that into a manifest recorded alongside every checkpoint/result this
# campaign produces (Phase 9 wires the checkpoint side of that).
#
# This module intentionally shells out to `git` rather than reading `.git` internals
# directly -- git's on-disk format (worktree `.git` files, packed-refs, commondir layout) is
# not a stable API to hand-parse, and every environment this code runs in already has a
# working `git` binary (confirmed: this repo IS a git worktree tree, `git --version` ->
# 2.43.7 at the time of writing).

# NOTE (consistency with this directory's own convention): every other file in `src/melitz/`
# defines its functions/types flat at top level, included directly into whatever scope loads
# `include_melitz.jl` (Main, in every production entry point / test file in this repo) --
# there is no per-file module wrapping anywhere else in this directory. This file follows
# that same convention deliberately (an earlier draft wrapped this in a
# `module MelitzProductionLauncher ... end` block, which would have been the only such module
# in `src/melitz/` and would have forced every caller to `using .MelitzProductionLauncher`
# inconsistently with how every sibling file is consumed -- removed).

using SHA: sha256
using LinearAlgebra: BLAS
import Dates

# ============================================================================
# git state
# ============================================================================

"""
    melitz_git_state(repo_dir::AbstractString) -> NamedTuple

Runs `git` against `repo_dir` and returns (common_dir, worktree_path, branch, commit_sha,
tracked_dirty::Bool, dirty_files::Vector{String}). `tracked_dirty`/`dirty_files` deliberately
EXCLUDE untracked files (`git status --porcelain --untracked-files=no`) -- an untracked output
file sitting in the worktree (a checkpoint, a log, a result CSV) must never by itself block a
launch; only modifications to TRACKED files (source, config, `.opt` files, anything checked
into git) count as "dirty" for launch-refusal purposes. This is the exact split Phase 8's own
requirement #2/"untracked output files should not by themselves block launch" describes.
"""
function melitz_git_state(repo_dir::AbstractString)
    git(args...) = strip(read(Cmd(String["git", "-C", repo_dir, string.(args)...]), String))
    common_dir = git("rev-parse", "--git-common-dir")
    # git-common-dir can be relative to repo_dir (e.g. "../.git") -- normalize to absolute.
    common_dir_abs = isabspath(common_dir) ? common_dir : normpath(joinpath(repo_dir, common_dir))
    worktree_path = git("rev-parse", "--show-toplevel")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit_sha = git("rev-parse", "HEAD")
    porcelain = read(Cmd(String["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"]), String)
    dirty_files = String[l[4:end] for l in split(porcelain, '\n') if !isempty(l)]
    return (common_dir=common_dir_abs, worktree_path=worktree_path, branch=branch,
        commit_sha=commit_sha, tracked_dirty=!isempty(dirty_files), dirty_files=dirty_files)
end

"""
    melitz_check_ancestry(repo_dir, current_sha, approved_base_sha) -> Bool

`git merge-base --is-ancestor approved_base_sha current_sha` -- true iff `current_sha`
descends from (or equals) `approved_base_sha`. Exit code 0 -> true, 1 -> false, anything else
(e.g. unknown commit) -> throws (a config error, not a normal refusal).
"""
function melitz_check_ancestry(repo_dir::AbstractString, current_sha::AbstractString, approved_base_sha::AbstractString)
    cmd = Cmd(Cmd(String["git", "-C", repo_dir, "merge-base", "--is-ancestor", approved_base_sha, current_sha]); ignorestatus=true)
    proc = run(cmd)
    code = proc.exitcode
    code in (0, 1) || error("melitz_check_ancestry: git merge-base returned unexpected exit code $code " *
        "(likely an unknown/unreachable commit $approved_base_sha or $current_sha, not a normal ancestry mismatch)")
    return code == 0
end

"""
    melitz_approved_base_commit(repo_dir) -> String

Reads the approved production integration base commit SHA from a small, git-tracked config
file (`melitz_production_approved_base.txt` at the repo root) rather than a hardcoded
constant in this source file -- exactly the "parameterize this, don't hardcode a single SHA
that will immediately go stale" requirement: whoever approves a NEW production integration
commit updates that one file (a normal, reviewable git commit), and every campaign launched
afterward automatically checks ancestry against the new approved base without any Julia
source change. Can be overridden per-invocation via the `MELITZ_PRODUCTION_APPROVED_BASE`
environment variable (useful for testing this file's own refusal behavior against a
synthetic ancestry, without touching the tracked config file).
"""
function melitz_approved_base_commit(repo_dir::AbstractString)
    env_override = get(ENV, "MELITZ_PRODUCTION_APPROVED_BASE", "")
    !isempty(env_override) && return env_override
    cfg_path = joinpath(repo_dir, "melitz_production_approved_base.txt")
    isfile(cfg_path) || error("melitz_approved_base_commit: no $cfg_path found and no " *
        "MELITZ_PRODUCTION_APPROVED_BASE env override set -- cannot determine the approved " *
        "production base commit to check ancestry against.")
    return strip(read(cfg_path, String))
end

# ============================================================================
# toolchain / environment report
# ============================================================================

function melitz_toolchain_report(; knitro_module=nothing)
    knitro_version = "unknown"
    if knitro_module !== nothing
        try
            buf = Vector{UInt8}(undef, 15)
            Base.invokelatest(knitro_module.KN_get_release, 15, buf)
            knitro_version = strip(String(buf), '\0')
        catch e
            knitro_version = "error: $(sprint(showerror, e))"
        end
    end
    return (julia_version=string(VERSION), knitro_version=knitro_version,
        julia_nthreads=Threads.nthreads(), blas_nthreads=BLAS.get_num_threads())
end

# ============================================================================
# fingerprints
# ============================================================================

"""
    melitz_file_fingerprint(path) -> String (hex sha256)

Content hash of a single file (used for both data files -- real_data/noah_D20/*.csv -- and
option files -- *.opt). Deliberately a hash of CONTENT, not mtime/size, so a file restored
from git history or copied verbatim fingerprints identically regardless of filesystem
metadata.
"""
function melitz_file_fingerprint(path::AbstractString)
    isfile(path) || error("melitz_file_fingerprint: not a file: $path")
    return bytes2hex(sha256(read(path)))
end

"""
    melitz_options_fingerprint(paths) -> String (hex sha256)

Combined fingerprint of a set of KNITRO `.opt` option files (order-independent: paths are
sorted before hashing so the same file set fingerprints identically regardless of the order
callers happen to list them in).
"""
function melitz_options_fingerprint(paths::AbstractVector{<:AbstractString})
    parts = String[]
    for p in sort(collect(paths))
        push!(parts, basename(p), melitz_file_fingerprint(p))
    end
    return bytes2hex(sha256(join(parts, "|")))
end

function melitz_data_fingerprint(real_data_dir::AbstractString)
    isdir(real_data_dir) || error("melitz_data_fingerprint: not a directory: $real_data_dir")
    files = sort(filter(f -> endswith(f, ".csv"), readdir(real_data_dir)))
    isempty(files) && error("melitz_data_fingerprint: no .csv files found in $real_data_dir")
    return melitz_options_fingerprint(joinpath.(real_data_dir, files))
end

# ============================================================================
# source-location manifest (Phase 3 methods()/which() audit, made reusable at runtime)
# ============================================================================

"""
    melitz_source_location_manifest(; melitz_ccbundle_instance=nothing) -> Dict{String,String}

Derives file:line source locations for the key production entry points, via the REAL Julia
method table (`methods`/`which`), not a static grep -- so this manifest reflects what THIS
process actually has loaded and will actually call, catching a shadowing/stale-copy problem a
grep-only audit could miss (a duplicate function definition earlier on LOAD_PATH, a stale
`Revise`-tracked copy, etc.). Functions are resolved by name lookup in `Main` (all Melitz
`include`s happen into `Main` in every production entry point / test file in this repo, per
Phase 3's own audit) -- if a name isn't defined, its manifest entry records that fact instead
of throwing, so one missing/renamed function doesn't prevent the rest of the manifest from
being produced.
"""
function melitz_source_location_manifest(; melitz_ccbundle_instance=nothing)
    names_to_check = [
        "melitz_recover_lfd", "melitz_recover_lfd_from_solution", "mul_G!", "mul_Gt!",
        "melitz_full_weighted_gram!", "melitz_full_weighted_gram_parallel!",
        "melitz_fixed_q_middle_constraint_system", "melitz_middle_objective_and_gradient!",
        "melitz_run_welfare_plus_a_sequential_search", "melitz_reduced_q_propose_direction",
        "melitz_reduced_q_propose_direction_threaded", "melitz_assert_production_bundle!",
        "melitz_live_backend_manifest", "evaluate_melitz_delta", "evaluate_melitz_delta_from_solution",
        "build_melitz_psi_bundle", "build_melitz_cc_bundle",
    ]
    manifest = Dict{String,String}()
    for nm in names_to_check
        if isdefined(Main, Symbol(nm))
            f = getfield(Main, Symbol(nm))
            locs = String[]
            for m in methods(f)
                push!(locs, "$(m.file):$(m.line)")
            end
            manifest[nm] = join(locs, " ; ")
        else
            manifest[nm] = "NOT DEFINED in Main at manifest-generation time"
        end
    end
    if melitz_ccbundle_instance !== nothing
        try
            n = melitz_ccbundle_instance.outer_constr_index
            m = which(melitz_ccbundle_instance, (typeof(zeros(n)),))
            manifest["MelitzCCBundle functor (obj(x))"] = "$(m.file):$(m.line)"
        catch e
            manifest["MelitzCCBundle functor (obj(x))"] = "error deriving: $(sprint(showerror, e))"
        end
    end
    return manifest
end

# ============================================================================
# manifest struct + refusal exception + top-level preflight entry point
# ============================================================================

struct MelitzLaunchRefusal <: Exception
    reason::String
end
Base.showerror(io::IO, e::MelitzLaunchRefusal) = print(io, "MelitzLaunchRefusal: ", e.reason)

Base.@kwdef struct MelitzProductionManifest
    generated_at::String
    git_common_dir::String
    worktree_path::String
    branch::String
    commit_sha::String
    tracked_dirty::Bool
    dirty_files::Vector{String}
    approved_base_commit::String
    is_descendant_of_approved_base::Bool
    julia_version::String
    knitro_version::String
    julia_nthreads::Int
    blas_nthreads::Int
    data_fingerprints::Dict{String,String}
    options_fingerprints::Dict{String,String}
    source_locations::Dict{String,String}
    diagnostic_override_used::Bool
end

function _manifest_to_text(m::MelitzProductionManifest)
    io = IOBuffer()
    println(io, "=== Melitz Production Manifest ===")
    println(io, "generated_at: ", m.generated_at)
    println(io, "git_common_dir: ", m.git_common_dir)
    println(io, "worktree_path: ", m.worktree_path)
    println(io, "branch: ", m.branch)
    println(io, "commit_sha: ", m.commit_sha)
    println(io, "tracked_dirty: ", m.tracked_dirty)
    if m.tracked_dirty
        println(io, "dirty_files:")
        for f in m.dirty_files
            println(io, "  ", f)
        end
    end
    println(io, "approved_base_commit: ", m.approved_base_commit)
    println(io, "is_descendant_of_approved_base: ", m.is_descendant_of_approved_base)
    println(io, "diagnostic_override_used: ", m.diagnostic_override_used)
    println(io, "julia_version: ", m.julia_version)
    println(io, "knitro_version: ", m.knitro_version)
    println(io, "julia_nthreads: ", m.julia_nthreads)
    println(io, "blas_nthreads: ", m.blas_nthreads)
    println(io, "data_fingerprints:")
    for (k, v) in sort(collect(m.data_fingerprints))
        println(io, "  ", k, " = ", v)
    end
    println(io, "options_fingerprints:")
    for (k, v) in sort(collect(m.options_fingerprints))
        println(io, "  ", k, " = ", v)
    end
    println(io, "source_locations:")
    for (k, v) in sort(collect(m.source_locations))
        println(io, "  ", k, " -> ", v)
    end
    return String(take!(io))
end

"""
    melitz_production_preflight!(repo_dir; allow_dirty=false, knitro_module=nothing,
        melitz_ccbundle_instance=nothing, real_data_dir=nothing, option_files=String[],
        write_manifest_to=nothing) -> MelitzProductionManifest

The Phase 8 canonical entry point. Every production launcher script should call this FIRST,
before building any fixture or touching KNITRO. Throws `MelitzLaunchRefusal` (NOT a generic
error) if:
  - the tracked working tree is dirty (modified/staged tracked files) and `allow_dirty` is
    not explicitly set true (the diagnostic override flag Phase 8 requires) -- untracked
    files (checkpoints, logs, result CSVs sitting in the worktree) are NEVER a refusal
    reason, only tracked-file modifications are;
  - the current commit is not a descendant of (or equal to) the approved production base
    commit (`melitz_approved_base_commit`).
On success, returns (and optionally writes to disk) a `MelitzProductionManifest` -- the
caller (a checkpoint/result-writing driver, Phase 9) should serialize/copy this same manifest
into every checkpoint and result directory it creates.
"""
function melitz_production_preflight!(repo_dir::AbstractString; allow_dirty::Bool=false,
        knitro_module=nothing, melitz_ccbundle_instance=nothing,
        real_data_dir::Union{Nothing,AbstractString}=nothing,
        option_files::AbstractVector{<:AbstractString}=String[],
        write_manifest_to::Union{Nothing,AbstractString}=nothing)

    git_state = melitz_git_state(repo_dir)
    println("Melitz production preflight:")
    println("  git_common_dir = ", git_state.common_dir)
    println("  worktree_path  = ", git_state.worktree_path)
    println("  branch         = ", git_state.branch)
    println("  commit_sha     = ", git_state.commit_sha)
    println("  tracked_dirty  = ", git_state.tracked_dirty)
    flush(stdout)

    if git_state.tracked_dirty && !allow_dirty
        throw(MelitzLaunchRefusal("tracked source/config files are modified relative to HEAD " *
            "($(length(git_state.dirty_files)) file(s): $(join(git_state.dirty_files, ", "))) -- " *
            "refusing to launch a production campaign against an uncommitted/modified tree. " *
            "Pass allow_dirty=true (an explicit diagnostic override) if this is intentional."))
    end

    approved_base = melitz_approved_base_commit(repo_dir)
    is_descendant = melitz_check_ancestry(repo_dir, git_state.commit_sha, approved_base)
    println("  approved_base_commit = ", approved_base)
    println("  is_descendant_of_approved_base = ", is_descendant)
    flush(stdout)
    if !is_descendant
        throw(MelitzLaunchRefusal("current commit $(git_state.commit_sha) does NOT descend from " *
            "the approved production base commit $approved_base (git merge-base --is-ancestor " *
            "failed) -- refusing to launch. If this is a deliberate new integration, update " *
            "melitz_production_approved_base.txt (or set MELITZ_PRODUCTION_APPROVED_BASE) " *
            "after review, do not bypass this check silently."))
    end

    toolchain = melitz_toolchain_report(; knitro_module=knitro_module)

    data_fps = Dict{String,String}()
    if real_data_dir !== nothing
        data_fps[real_data_dir] = melitz_data_fingerprint(real_data_dir)
    end
    opt_fps = Dict{String,String}()
    for p in option_files
        opt_fps[basename(p)] = melitz_file_fingerprint(p)
    end

    source_locs = melitz_source_location_manifest(; melitz_ccbundle_instance=melitz_ccbundle_instance)

    manifest = MelitzProductionManifest(;
        generated_at=string(Dates.now()),
        git_common_dir=git_state.common_dir, worktree_path=git_state.worktree_path,
        branch=git_state.branch, commit_sha=git_state.commit_sha,
        tracked_dirty=git_state.tracked_dirty, dirty_files=git_state.dirty_files,
        approved_base_commit=approved_base, is_descendant_of_approved_base=is_descendant,
        julia_version=toolchain.julia_version, knitro_version=toolchain.knitro_version,
        julia_nthreads=toolchain.julia_nthreads, blas_nthreads=toolchain.blas_nthreads,
        data_fingerprints=data_fps, options_fingerprints=opt_fps,
        source_locations=source_locs, diagnostic_override_used=(git_state.tracked_dirty && allow_dirty))

    if write_manifest_to !== nothing
        mkpath(dirname(write_manifest_to))
        write(write_manifest_to, _manifest_to_text(manifest))
        println("  manifest written to ", write_manifest_to)
    end
    println("PREFLIGHT PASSED.")
    flush(stdout)
    return manifest
end

melitz_manifest_to_text(m::MelitzProductionManifest) = _manifest_to_text(m)
