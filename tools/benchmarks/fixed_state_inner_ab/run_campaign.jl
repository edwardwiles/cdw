# Task "fixed-state FULL-vs-REDUCED inner A/B", steps 6-9: the reusable campaign runner. Loads
# the real sentinel point bank (step 4), and for every point runs a genuine cold solve on BOTH
# arms from the SAME decoded state (step 6.1), recording real telemetry to CSV. One process, one
# CSV, append-as-it-goes (each row flushed immediately so a killed/timed-out run keeps whatever it
# already finished, per this repo's own "always flush, verify PIDs" standing practice).
#
# Usage: julia --project=. run_campaign.jl <mode_a|mode_b> <maxtime_real_seconds> <output_csv>
#        [family1,family2,...]   (optional: restrict to a comma-separated family subset)

include(joinpath(@__DIR__, "cold_solve_pair.jl"))
using Printf, JLD2

const POINT_BANK_PATH = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/FIXED_STATE_POINT_BANK.jld2"
const SCRATCH_ROOT = "/bbkinghome/edav/repo_scratch/profiled-fixed-state-inner-ab-2026-08-04/cold_solve_runs"

function run_campaign(; mode::Symbol, maxtime_real::Float64, output_csv::String,
        families::Vector{Symbol} = collect(CANONICAL_FAMILIES))
    points = JLD2.load(POINT_BANK_PATH, "points")
    mode_points = filter(p -> p.mode == mode && p.family in families, points)
    sci = mode === :mode_a ? mode_a_scientific_manifest() : mode_b_scientific_manifest()

    is_new = !isfile(output_csv)
    io = open(output_csv, "a")
    if is_new
        println(io, "family,point_id,point_type,mode,W,arm,knitro_status,n_eval,n_grad,wall_s,gp,Delta,error")
        flush(io)
    end

    for family in families
        fam_points = filter(p -> p.family == family, mode_points)
        isempty(fam_points) && continue
        println("="^90); println("Family=$family, mode=$mode, $(length(fam_points)) points"); flush(stdout)
        println("="^90)

        t_setup = @elapsed setup = build_reduced_setup(family, sci)
        @printf("  REDUCED setup built in %.1fs\n", t_setup); flush(stdout)
        ctx = setup.ctx

        for p in fam_points
            println("-- point $(p.point_id) ($(p.point_type)) --"); flush(stdout)

            # REDUCED arm
            reduced_ckpt = joinpath(SCRATCH_ROOT, "reduced_$(family)_$(mode)_$(p.point_id)")
            local r_reduced, err_reduced
            err_reduced = ""
            try
                r_reduced = reduced_cold_solve(setup, copy(p.w); eta_nu0 = isempty(p.eta_nu) ? nothing : copy(p.eta_nu),
                    delta = 1.0, maxtime_real = maxtime_real, threaded_gradient = true,
                    ckpt_dir = reduced_ckpt, run_id = "campaign_$(family)_$(p.point_id)")
            catch e
                err_reduced = sprint(showerror, e)
                r_reduced = (knitro_status = -99999, n_eval = 0, n_grad = 0, wall = 0.0, best = nothing)
            end
            gp_r = r_reduced.best === nothing ? NaN : r_reduced.best.gp
            Delta_r = r_reduced.best === nothing ? NaN : r_reduced.best.Delta
            @printf(io, "%s,%s,%s,%s,%d,reduced,%d,%d,%d,%.3f,%s,%s,\"%s\"\n",
                family, p.point_id, p.point_type, mode, sci.W, r_reduced.knitro_status,
                r_reduced.n_eval, r_reduced.n_grad, r_reduced.wall,
                isnan(gp_r) ? "" : @sprintf("%.10f", gp_r), isnan(Delta_r) ? "" : @sprintf("%.10f", Delta_r),
                replace(err_reduced, "\"" => "'"))
            flush(io)
            @printf("  REDUCED: status=%d n_eval=%d n_grad=%d wall=%.1fs gp=%s Delta=%s %s\n",
                r_reduced.knitro_status, r_reduced.n_eval, r_reduced.n_grad, r_reduced.wall,
                isnan(gp_r) ? "NaN" : @sprintf("%.6f", gp_r), isnan(Delta_r) ? "NaN" : @sprintf("%.6f", Delta_r),
                isempty(err_reduced) ? "" : "ERROR: $err_reduced")
            flush(stdout)

            # FULL arm -- same decoded state, bridged into this family's real coordinate mode
            full_ckpt = joinpath(SCRATCH_ROOT, "full_$(family)_$(mode)_$(p.point_id)")
            local r_full, err_full
            err_full = ""
            try
                r_full = full_cold_solve(family, p.gp, p.z_full, ctx; eta_nu = isempty(p.eta_nu) ? nothing : copy(p.eta_nu),
                    delta = 1.0, maxtime_real = maxtime_real, ckpt_dir = full_ckpt, sci = sci, meanzc_K_mean = sci.K_mean)
            catch e
                err_full = sprint(showerror, e)
                r_full = (knitro_status = -99999, n_eval = 0, n_grad = 0, wall = 0.0, best = nothing)
            end
            gp_f = r_full.best === nothing ? NaN : r_full.best.gp
            Delta_f = r_full.best === nothing ? NaN : r_full.best.Delta
            @printf(io, "%s,%s,%s,%s,%d,full,%d,%d,%d,%.3f,%s,%s,\"%s\"\n",
                family, p.point_id, p.point_type, mode, sci.W, r_full.knitro_status,
                r_full.n_eval, r_full.n_grad, r_full.wall,
                isnan(gp_f) ? "" : @sprintf("%.10f", gp_f), isnan(Delta_f) ? "" : @sprintf("%.10f", Delta_f),
                replace(err_full, "\"" => "'"))
            flush(io)
            @printf("  FULL:    status=%d n_eval=%d n_grad=%d wall=%.1fs gp=%s Delta=%s %s\n",
                r_full.knitro_status, r_full.n_eval, r_full.n_grad, r_full.wall,
                isnan(gp_f) ? "NaN" : @sprintf("%.6f", gp_f), isnan(Delta_f) ? "NaN" : @sprintf("%.6f", Delta_f),
                isempty(err_full) ? "" : "ERROR: $err_full")
            flush(stdout)
        end
    end
    close(io)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(ARGS[1])
    maxtime_real = parse(Float64, ARGS[2])
    output_csv = ARGS[3]
    families = length(ARGS) >= 4 ? Symbol.(split(ARGS[4], ",")) : collect(CANONICAL_FAMILIES)
    run_campaign(mode = mode, maxtime_real = maxtime_real, output_csv = output_csv, families = families)
    println("DONE")
end
