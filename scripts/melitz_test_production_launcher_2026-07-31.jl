#!/usr/bin/env julia
# Production-consolidation Phase 8 (2026-07-31): regression test for
# src/melitz/production_launcher.jl's refusal and pass-through behavior.
#
# Exercises melitz_production_preflight! against REAL git plumbing (a genuine, disposable
# `git worktree add` scratch worktree of this exact repo, cleaned up at the end) rather than
# mocking git -- the whole point of this gate is to catch REAL dirty-tree/wrong-ancestry
# states via the REAL `git` binary, so the test should exercise that real path.
#
# Covers:
#   1. Ancestry check unit tests against real, known commit pairs in this repo's own history.
#   2. Pass-through: a clean scratch worktree at a commit descending from the approved base
#      passes preflight.
#   3. Refusal (dirty tree): modifying a TRACKED file in the scratch worktree causes refusal,
#      UNLESS allow_dirty=true is passed (which passes through instead).
#   4. Refusal (wrong ancestry): overriding MELITZ_PRODUCTION_APPROVED_BASE to a real commit
#      that is NOT an ancestor of the scratch worktree's HEAD causes refusal.
#   5. Untracked files never block launch: an untracked (new, unadded) file in the scratch
#      worktree does NOT cause a refusal.
#
# USAGE: julia --project=. scripts/melitz_test_production_launcher_2026-07-31.jl

REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Test

const SCRATCH_WORKTREE = mktempdir(; prefix="melitz_launcher_test_")

function git_here(repo, args...)
    read(Cmd(String["git", "-C", repo, string.(args)...]), String)
end

# A non-ancestor commit SHA is needed to exercise the "wrong ancestry" refusal path. This
# used to be a hardcoded SHA borrowed from an unrelated branch in the repo this was originally
# written against (trade_robustness_modular) -- that object is NOT part of what gets pushed to
# a shared remote (only commits reachable from a pushed branch tip travel with it), so the
# test broke immediately the first time this integration branch was relocated to a different
# clone (cdw/melitz/...), exactly the kind of repo-portability bug Phase 3's own audit is
# supposed to catch. Fixed to fabricate a guaranteed-non-ancestor commit locally instead:
# `git commit-tree` on the empty tree with NO parent creates a real, locally-reachable commit
# object with no path to/from HEAD, so it exists in every clone (we just created it) and is
# provably not an ancestor (it has no parents at all).
const EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's well-known empty-tree hash, universal across all repos
NONANCESTOR_SHA = strip(git_here(REPO, "commit-tree", EMPTY_TREE_SHA, "-m", "melitz_test_production_launcher: synthetic non-ancestor commit (no parent)"))

@testset "melitz_check_ancestry: real known commit pairs" begin
    # d0904c7 -> 6e803d4 -> ... -> HEAD is this repo's own real, linear Melitz campaign
    # history (the exact ancestry chain the governing prompt describes).
    @test melitz_check_ancestry(REPO, "6e803d43875e73deb3cf8acf3c04e8febd808e53", "d0904c7") == true
    @test melitz_check_ancestry(REPO, readchomp(`git -C $REPO rev-parse HEAD`), "6e803d43875e73deb3cf8acf3c04e8febd808e53") == true
    # A synthetic, parentless commit: guaranteed to exist in THIS clone (just created above)
    # and guaranteed not an ancestor of HEAD in either direction (it has no parents at all).
    @test melitz_check_ancestry(REPO, readchomp(`git -C $REPO rev-parse HEAD`), NONANCESTOR_SHA) == false
end

# Real, disposable scratch worktree at the current commit -- isolated from the main
# integration worktree, removed at the end regardless of test outcome.
git_here(REPO, "worktree", "add", "-d", SCRATCH_WORKTREE, "HEAD")

try
    @testset "melitz_production_preflight!: pass-through on a clean scratch worktree" begin
        manifest = melitz_production_preflight!(SCRATCH_WORKTREE)
        @test manifest.tracked_dirty == false
        @test manifest.is_descendant_of_approved_base == true
        @test manifest.diagnostic_override_used == false
        @test !isempty(manifest.source_locations)
    end

    @testset "melitz_production_preflight!: untracked file does NOT block launch" begin
        untracked_path = joinpath(SCRATCH_WORKTREE, "melitz_launcher_test_untracked_scratch_file.txt")
        write(untracked_path, "scratch, never added to git")
        manifest = melitz_production_preflight!(SCRATCH_WORKTREE)
        @test manifest.tracked_dirty == false   # untracked files must not count
        rm(untracked_path)
    end

    @testset "melitz_production_preflight!: refuses on a dirty TRACKED file, unless allow_dirty=true" begin
        tracked_file = joinpath(SCRATCH_WORKTREE, "src", "melitz", "types.jl")
        @assert isfile(tracked_file)
        original = read(tracked_file, String)
        write(tracked_file, original * "\n# scratch dirty-tree test marker, never committed\n")
        try
            refusal = nothing
            try
                melitz_production_preflight!(SCRATCH_WORKTREE)
                error("expected MelitzLaunchRefusal, got no exception")
            catch e
                refusal = e
            end
            @test refusal isa MelitzLaunchRefusal
            @test occursin("types.jl", refusal.reason)

            # allow_dirty=true is the documented diagnostic override -- must pass through.
            manifest_override = melitz_production_preflight!(SCRATCH_WORKTREE; allow_dirty=true)
            @test manifest_override.tracked_dirty == true
            @test manifest_override.diagnostic_override_used == true
        finally
            write(tracked_file, original)   # restore -- scratch worktree is disposable anyway, but be tidy
        end
    end

    @testset "melitz_production_preflight!: refuses on wrong ancestry (env override)" begin
        ENV["MELITZ_PRODUCTION_APPROVED_BASE"] = NONANCESTOR_SHA
        try
            refusal = nothing
            try
                melitz_production_preflight!(SCRATCH_WORKTREE)
                error("expected MelitzLaunchRefusal, got no exception")
            catch e
                refusal = e
            end
            @test refusal isa MelitzLaunchRefusal
            @test occursin("does NOT descend", refusal.reason)
        finally
            delete!(ENV, "MELITZ_PRODUCTION_APPROVED_BASE")
        end
    end

    @testset "melitz_production_preflight!: write_manifest_to actually writes a readable file" begin
        out_path = joinpath(SCRATCH_WORKTREE, "scratch_manifest_test.txt")
        manifest = melitz_production_preflight!(SCRATCH_WORKTREE; write_manifest_to=out_path)
        @test isfile(out_path)
        txt = read(out_path, String)
        @test occursin(manifest.commit_sha, txt)
        @test occursin("source_locations", txt)
    end

    println("\nALL production_launcher.jl regression tests PASSED.")
finally
    # Cleanup: remove the scratch worktree registration + its directory. Never touches the
    # real integration worktree or any named branch/checkpoint the governing prompt protects.
    try
        git_here(REPO, "worktree", "remove", "--force", SCRATCH_WORKTREE)
    catch e
        @warn "Could not cleanly `git worktree remove` the scratch worktree, removing directory directly" exception=e
        rm(SCRATCH_WORKTREE; recursive=true, force=true)
    end
end
