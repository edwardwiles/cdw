# ============================================================================
# Continuation 8, workstream C: ONE empirical dense-vs-compressed comparison
# of profile_delta_at_gamma_c8's FULL local minimization (not just a single
# F-eval) -- per this workstream's brief ("try both once and pick"). Compares
# apples-to-apples: SAME g, SAME start point, SAME KNITRO settings, only the
# F-callback's moment_representation differs. The gradient callback
# (composite_gradient_at_fast) is unchanged in both cases (compressed has no
# gradient variant, see c8_gammabranch_core.jl header) -- so this measures
# whether compressed's cheaper F-evals matter net of the (larger, unchanged)
# gradient-callback cost that dominates a KNITRO local solve here.
# ============================================================================
include(joinpath(@__DIR__, "c8_gammabranch_core.jl"))
using Printf

const COMMIT_C8 = strip(read(`git rev-parse --short HEAD`, String))
const BENCH_OUTDIR = joinpath(D4X_ROOT, "results", "fullA_d4", COMMIT_C8, "c8_gammabranch_compressed_vs_dense_bench")
mkpath(BENCH_OUTDIR)

test_points = [
    ("low_g_incumbent",  G_INCUMBENT_C8,      ZFREE_INCUMBENT_C8),
    ("high_g_lower_incumbent", G_LOWER_INCUMBENT, ZFREE_LOWER_INCUMBENT),
]

open(joinpath(BENCH_OUTDIR, "bench_results.csv"), "w") do io
    println(io, "label,g,mode,knitro_status,n_eval,wall,best_Delta")
    for (label, g, zf0) in test_points
        for mode in (:dense, :compressed)
            # one throwaway warmup call to absorb JIT before the timed run (JIT cost is a
            # one-time process artifact, not a genuine per-eval cost difference -- excluding
            # it from the comparison matches this investigation's established discipline,
            # see MEMORY "Verify before causal claims")
            profile_delta_at_gamma_c8(g, zf0, ctx, pe; moment_repr = mode, maxtime_real = 5.0, hessopt_tag = "sr1")
            res = profile_delta_at_gamma_c8(g, zf0, ctx, pe; moment_repr = mode, maxtime_real = 30.0, hessopt_tag = "sr1")
            @printf("  [%-24s] mode=%-10s status=%d n_eval=%3d wall=%6.2fs best_Delta=%.8f\n",
                    label, mode, res.knitro_status, res.n_eval, res.wall, res.best_Delta)
            println(io, label, ",", g, ",", mode, ",", res.knitro_status, ",", res.n_eval, ",", res.wall, ",", res.best_Delta)
            flush(io)
        end
    end
end
println("\nWrote ", joinpath(BENCH_OUTDIR, "bench_results.csv"))
