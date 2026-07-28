# verify_manifest_gravity.jl -- post-hoc gravity-residual addendum for the ex-ante five-starts
# manifest (2026-07-28 shakedown campaign).
#
# Reuses gravity_from_logz(logA, ctx) -- the SAME independently-validated "outer gravity equality"
# metric used in test_gravity_pivot_vs_moment_d4.jl / GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md
# (machine-zero at 100 deterministic pivot-valid perturbations, ~1e-3 at naive bypassed-pivot
# points -- the exact CLAUDE.md-flagged "A_od==1 is not calibration" failure mode this check is
# designed to catch). Does NOT reconstruct A independently via a different formula -- it calls the
# real production pivot_expand on each start's own stored zfree, then hands the result to the same
# gravity_from_logz the outer driver's own algebra trace already established as decisive evidence.
#
# Usage: julia --project=. verify_manifest_gravity.jl <manifest_json> <out_md>
const _D4E = @__DIR__
for f in ["draw_design.jl", "winners.jl", "oracle.jl", "common_marginals_moments.jl", "common_marginals_interval.jl",
          "instrumentation.jl", "oracle_fast.jl", "gravity_elimination.jl",
          "compressed_moments.jl", "structured_moment_build.jl", "compressed_cc_inner.jl", "compressed_live.jl"]
    include(joinpath(_D4E, f))
end
include(joinpath(_D4E, "json_lite.jl"))
using Printf, Statistics

lp(xs...) = (println(xs...); flush(stdout))

MANIFEST_PATH = ARGS[1]
OUT_MD = ARGS[2]

manifest = json_load(MANIFEST_PATH)
cfg = manifest["config"]
W = Int(cfg["W"])
draw_design = Symbol(cfg["draw_design"])
draw_seed = Int(cfg["draw_seed"])
dest_sample = Symbol(cfg["destination_sample"])
delta = Float64(cfg["delta"])

lp(">> rebuilding ctx (must match the search script's own build exactly): W=", W, " draw_design=", draw_design,
   " draw_seed=", draw_seed, " destination_sample=", dest_sample, " delta=", delta)
ctx = d20_real_setup_design(W = W, δ = delta, find_smallest = true, draw_design = draw_design,
                            draw_seed = draw_seed, destination_sample = dest_sample)
pe = build_pivot_elimination(ctx)
lp(">> draw checksums: uniform=", ctx.draw_meta.checksum_uniform, " transformed=", ctx.draw_meta.checksum_transformed)
lp(">> manifest's own recorded checksums: uniform=", manifest["config"]["draw_checksum_uniform"],
   " transformed=", manifest["config"]["draw_checksum_transformed"])
manifest["config"]["draw_checksum_uniform"] == ctx.draw_meta.checksum_uniform ||
    error("draw checksum (uniform) mismatch -- this ctx does not reproduce the manifest's own draws.")
manifest["config"]["draw_checksum_transformed"] == ctx.draw_meta.checksum_transformed ||
    error("draw checksum (transformed) mismatch -- this ctx does not reproduce the manifest's own draws.")

rows = NamedTuple[]
for st in manifest["starts"]
    idx = Int(st["index"])
    zfree = jf64(st["zfree_pivot_reduced_log"])
    logA_full = pivot_expand(zfree, pe)
    grav_eq = gravity_from_logz(logA_full, ctx)
    lp("  start ", idx, " (", st["label"], "): outer_gravity_equality = ", @sprintf("%.6e", grav_eq))
    push!(rows, (index = idx, label = st["label"], gp = Float64(st["gp"]), outer_gravity_equality = grav_eq))
end

max_abs = maximum(abs(r.outer_gravity_equality) for r in rows)
lp(">> max|outer_gravity_equality| over all 5 starts = ", @sprintf("%.6e", max_abs))
verdict = max_abs < 1e-8 ? "MACHINE_ZERO_ALL_STARTS" : "NONZERO -- INVESTIGATE BEFORE RUNNING CAMPAIGN"
lp(">> verdict = ", verdict)

open(OUT_MD, "w") do io
    println(io, "# Ex-ante manifest gravity-residual verification (2026-07-28)\n")
    println(io, "Independently re-derives each accepted start's `outer_gravity_equality` via the ",
                "real production pivot decoder (`pivot_expand`) + `gravity_from_logz` -- the same ",
                "metric `GRAVITY_MOMENT_FINAL_DECISION_2026-07-28.md` established as machine-zero at ",
                "100 deterministic pivot-valid points and ~1e-3 at naive bypassed-pivot points. Not a ",
                "restatement of the search script's own internal coordinate round-trip check -- this ",
                "reconstructs full A from each start's stored `zfree` independently and re-evaluates ",
                "the gravity identity on it directly.\n")
    println(io, "Draw checksums cross-verified against the manifest's own recorded values before any ",
                "residual was computed (see run log).\n")
    println(io, "| start | label | gp | outer_gravity_equality |")
    println(io, "|---|---|---|---|")
    for r in rows
        println(io, "| ", r.index, " | `", r.label, "` | ", round(r.gp, digits = 8), " | ",
                @sprintf("%.6e", r.outer_gravity_equality), " |")
    end
    println(io, "\n**max|outer_gravity_equality| = ", @sprintf("%.6e", max_abs), "**\n")
    println(io, "**Verdict: ", verdict, "**")
end
lp(">> wrote ", OUT_MD)
