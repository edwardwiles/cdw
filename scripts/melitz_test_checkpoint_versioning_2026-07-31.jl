#!/usr/bin/env julia
# Production-consolidation Phase 9 (2026-07-31): regression tests for
# src/melitz/checkpoint_versioning.jl's resume-gate behavior.
#
# Covers the three required scenarios:
#   1. Same-commit resume passes silently.
#   2. Different-commit resume without migrate=true refuses (MelitzCheckpointVersionMismatch).
#   3. Different-commit resume WITH migrate=true cold-reverifies (via a caller-supplied
#      callback) and, on success, records both old and new commits; on a FAILED
#      cold-reverification, migration itself refuses (does not silently trust the stale
#      checkpoint).
#
# Uses a real, disposable scratch worktree (same pattern as
# melitz_test_production_launcher_2026-07-31.jl) so the "different commit" scenario is a
# REAL git commit difference, not a mocked one.

REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "misc", "doubleDiff.jl"))
include(joinpath(REPO, "src", "melitz", "include_melitz.jl"))
using Test

const SCRATCH = mktempdir(; prefix="melitz_ckpt_test_")
git_here(repo, args...) = read(Cmd(String["git", "-C", repo, string.(args)...]), String)
git_here(REPO, "worktree", "add", "-d", SCRATCH, "HEAD")

try
    ckpt_path = joinpath(SCRATCH, "test_checkpoint.jls")
    payload = (theta_free=[0.1, 0.2, 0.3], Delta=0.0123, label="scratch test payload")

    @testset "melitz_save_checkpoint! + same-commit resume passes silently" begin
        ckpt = melitz_save_checkpoint!(ckpt_path, payload, SCRATCH)
        @test ckpt.version_info.commit_sha == strip(git_here(SCRATCH, "rev-parse", "HEAD"))
        @test isfile(ckpt_path)

        loaded_payload, info = melitz_resume_checkpoint(ckpt_path, SCRATCH)
        @test loaded_payload == payload
        @test info isa MelitzCheckpointVersionInfo
        @test info.commit_sha == ckpt.version_info.commit_sha
    end

    # Advance the scratch worktree to a NEW commit (a real, trivial, disposable commit --
    # never touches the real integration worktree or any named branch).
    marker_file = joinpath(SCRATCH, "scratch_ckpt_marker.txt")
    write(marker_file, "advance commit for checkpoint-versioning test\n")
    git_here(SCRATCH, "add", "scratch_ckpt_marker.txt")
    git_here(SCRATCH, "-c", "user.email=test@test.com", "-c", "user.name=test", "commit", "-m", "scratch advance commit")
    new_commit = strip(git_here(SCRATCH, "rev-parse", "HEAD"))
    old_commit = strip(read(Cmd(String["git", "-C", REPO, "rev-parse", "HEAD"]), String))
    @assert new_commit != old_commit

    @testset "different-commit resume WITHOUT migrate=true refuses" begin
        refusal = nothing
        try
            melitz_resume_checkpoint(ckpt_path, SCRATCH)
            error("expected MelitzCheckpointVersionMismatch, got no exception")
        catch e
            refusal = e
        end
        @test refusal isa MelitzCheckpointVersionMismatch
        @test occursin(old_commit, refusal.reason)
        @test occursin(new_commit, refusal.reason)
    end

    @testset "different-commit resume WITH migrate=true, cold_reverify=true -> migrates, records both commits" begin
        reverify_calls = Ref(0)
        cold_reverify_pass(p) = (reverify_calls[] += 1; p.Delta > 0)  # trivially true for this payload
        loaded_payload, migration_info = melitz_resume_checkpoint(ckpt_path, SCRATCH;
            migrate=true, cold_reverify=cold_reverify_pass)
        @test loaded_payload == payload
        @test reverify_calls[] == 1
        @test migration_info.old_commit == old_commit
        @test migration_info.new_commit == new_commit
        @test migration_info.cold_reverified == true
    end

    @testset "different-commit resume WITH migrate=true, cold_reverify=false -> migration refuses" begin
        cold_reverify_fail(p) = false
        refusal = nothing
        try
            melitz_resume_checkpoint(ckpt_path, SCRATCH; migrate=true, cold_reverify=cold_reverify_fail)
            error("expected MelitzCheckpointVersionMismatch, got no exception")
        catch e
            refusal = e
        end
        @test refusal isa MelitzCheckpointVersionMismatch
        @test occursin("did NOT reverify", refusal.reason)
    end

    @testset "different-commit resume WITH migrate=true, cold_reverify THROWS -> migration refuses (not silently trusted)" begin
        cold_reverify_throws(p) = error("simulated re-verification crash")
        refusal = nothing
        try
            melitz_resume_checkpoint(ckpt_path, SCRATCH; migrate=true, cold_reverify=cold_reverify_throws)
            error("expected MelitzCheckpointVersionMismatch, got no exception")
        catch e
            refusal = e
        end
        @test refusal isa MelitzCheckpointVersionMismatch
        @test occursin("THREW", refusal.reason)
    end

    @testset "migrate=true with no cold_reverify callback at all -> refuses" begin
        refusal = nothing
        try
            melitz_resume_checkpoint(ckpt_path, SCRATCH; migrate=true)
            error("expected MelitzCheckpointVersionMismatch, got no exception")
        catch e
            refusal = e
        end
        @test refusal isa MelitzCheckpointVersionMismatch
        @test occursin("requires a cold_reverify", refusal.reason)
    end

    @testset "melitz_tracked_diff_hash is deterministic and changes with tracked content" begin
        h1 = melitz_tracked_diff_hash(SCRATCH)
        h2 = melitz_tracked_diff_hash(SCRATCH)
        @test h1 == h2   # clean tree (no diff vs its own HEAD) -- stable
        tracked = joinpath(SCRATCH, "src", "melitz", "types.jl")
        original = read(tracked, String)
        write(tracked, original * "\n# scratch marker\n")
        h3 = melitz_tracked_diff_hash(SCRATCH)
        @test h3 != h1
        write(tracked, original)
    end

    println("\nALL checkpoint_versioning.jl regression tests PASSED.")
finally
    try
        git_here(REPO, "worktree", "remove", "--force", SCRATCH)
    catch e
        @warn "Could not cleanly git worktree remove the scratch worktree" exception=e
        rm(SCRATCH; recursive=true, force=true)
    end
end
