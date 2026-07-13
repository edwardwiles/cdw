# Parse KNITRO outlev=iter tables from the solve logs and plot per-iteration
# optimality (KKT) error and feasibility error for full vs reduced (Section 1).
using Plots, Printf
gr()
OUT = joinpath(@__DIR__,"out")

# extract the two iteration tables (upper before the *LOWER marker, lower after) from a log
function parse_tables(path, lowermarker)
    txt = readlines(path)
    splitat = findfirst(l->occursin(lowermarker, l), txt)
    splitat === nothing && (splitat = length(txt))
    function grab(lines)
        it=Int[]; obj=Float64[]; feas=Float64[]; opt=Float64[]
        for l in lines
            f = split(strip(l))
            length(f) < 3 && continue
            m = tryparse(Int, f[1]); m === nothing && continue
            o = tryparse(Float64, f[2]); o === nothing && continue
            push!(it, m); push!(obj, o)
            fe = length(f)>=3 ? tryparse(Float64,f[3]) : nothing
            op = length(f)>=4 ? tryparse(Float64,f[4]) : nothing
            push!(feas, fe===nothing ? NaN : fe)
            push!(opt,  op===nothing ? NaN : op)
        end
        (it,obj,feas,opt)
    end
    (grab(txt[1:splitat]), grab(txt[splitat+1:end]))
end

(fU,fL) = parse_tables(joinpath(OUT,"solve_full.log"), "full LOWER")
(rU,rL) = parse_tables(joinpath(OUT,"solve_reduced.log"), "reduced LOWER")

clip(v) = map(x-> (isnan(x)||x<=0) ? NaN : x, v)

p1 = plot(title="KKT optimality error", xlabel="outer iteration",
    ylabel="OptError (log)", yscale=:log10, legend=:topright, framestyle=:box, gridalpha=0.3)
plot!(p1, fU[1], clip(fU[4]), label="full  upper", lw=2, color=:steelblue)
plot!(p1, fL[1], clip(fL[4]), label="full  lower", lw=2, ls=:dash, color=:steelblue)
plot!(p1, rU[1], clip(rU[4]), label="reduced upper", lw=2, color=:darkorange)
plot!(p1, rL[1], clip(rL[4]), label="reduced lower", lw=2, ls=:dash, color=:darkorange)

# KNITRO minimizes −κ on the upper solve (κ=−obj) and +κ on the lower solve (κ=+obj)
p2 = plot(title="objective κ per iteration",
    xlabel="outer iteration", ylabel="κ", legend=:right, framestyle=:box, gridalpha=0.3)
plot!(p2, fU[1], -fU[2], label="full  upper", lw=2, color=:steelblue)
plot!(p2, fL[1],  fL[2], label="full  lower (stalls at 0.0181)", lw=2, ls=:dash, color=:steelblue)
plot!(p2, rU[1], -rU[2], label="reduced upper", lw=2, color=:darkorange)
plot!(p2, rL[1],  rL[2], label="reduced lower (reaches 0.0090)", lw=2, ls=:dash, color=:darkorange)
hline!(p2, [0.008955], color=:red, ls=:dot, alpha=0.7, label="reduced lower pt (feasible in full!)")

plt = plot(p1, p2, layout=(1,2), size=(1050,430))
savefig(plt, joinpath(OUT,"convergence.png"))
println("wrote ", joinpath(OUT,"convergence.png"))

# print final-iteration summary for the report table
for (nm,U,L) in (("full",fU,fL),("reduced",rU,rL))
    @printf("%s  upper: iters=%d  κ=%.5f  opt=%.3g  feas=%.2g\n", nm, U[1][end], -U[2][end], U[4][end], U[3][end])
    @printf("%s  lower: iters=%d  κ=%.5f  opt=%.3g  feas=%.2g\n", nm, L[1][end], -L[2][end], L[4][end], L[3][end])
end
