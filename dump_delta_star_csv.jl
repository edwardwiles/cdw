using JLD2
dir = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular_perf/sequential_gravity"
for W in ["W80000", "W800000"]
    d = JLD2.load(joinpath(dir, "delta_star_schedule_$(W)_out.jld2"))
    results = d["results"]
    open(joinpath(dir, "delta_star_schedule_$(W).csv"), "w") do io
        println(io, "gammap,GT,delta_star,nStatus,ok")
        for r in results
            println(io, "$(r.γp),$(r.κ),$(r.δstar),$(r.nStatus),$(r.ok)")
        end
    end
    println("wrote $(length(results)) rows for $W")
end
