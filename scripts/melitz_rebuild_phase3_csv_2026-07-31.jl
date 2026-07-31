# Rebuild docs/key_results/overnight_qpoll_delta0p5_2026-07-31/phase3_phase4_grid_points.csv from
# the individual per-point .jls dumps (unaffected by the concurrent-append CSV race observed when
# 31 processes appended to the shared CSV simultaneously without file locking). The .jls dumps are
# each written to a unique per-(seed,m,gt_mode) path and are therefore race-free.
using Serialization, Printf

const REPO2 = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
const OVERDIR = joinpath(REPO2, "docs", "key_results", "overnight_qpoll_delta0p5_2026-07-31")
const PDIR = joinpath(OVERDIR, "phase3_phase4_points")
const OUTCSV = joinpath(OVERDIR, "phase3_phase4_grid_points.csv")

files = sort(readdir(PDIR))
open(OUTCSV, "w") do io
    println(io, "seed_chain_id,seed_direction,m,gt_mode,GT_target,outcome,classification,Delta,lfd_ok,lfd_Delta,wall_s,source_file")
    for f in files
        endswith(f, ".jls") || continue
        d = deserialize(joinpath(PDIR, f))
        seed_dir = occursin("upper", d.seed_chain_id) ? "upper" : "lower"
        outcome = d.lfd_ok ? "FiniteSolvedVerified" : "FiniteSolvedLfdUnverified"
        println(io, join([d.seed_chain_id, seed_dir, d.m, d.gt_mode, d.GT_target, outcome,
            d.classification, d.Delta, d.lfd_ok, d.lfd_Delta, "NA", f], ","))
    end
end
println("Rebuilt ", OUTCSV, " from ", count(f -> endswith(f, ".jls"), files), " point files.")
