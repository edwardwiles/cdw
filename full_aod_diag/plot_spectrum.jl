# Plot the singular-value spectrum of the free moment Jacobian, full vs reduced.
using JLD2, Plots, LinearAlgebra, Printf
OUT = joinpath(@__DIR__,"out")
gr()

df = load(joinpath(OUT,"cond_full.jld2"))
dr = load(joinpath(OUT,"cond_reduced.jld2"))
Sf = df["S"]; Sr = dr["S"]

plt = plot(size=(760,480), yscale=:log10, legend=:bottomleft,
    xlabel="singular-value index (largest → smallest)", ylabel="singular value (log scale)",
    title="Moment-Jacobian ∂E[g]/∂θ spectrum  (D=4)", framestyle=:box, gridalpha=0.3)
scatter!(plt, 1:length(Sf), Sf, label=@sprintf("full A_od (18 free, cond=%.0f)", Sf[1]/Sf[end]),
    marker=(:circle,6), color=:steelblue)
scatter!(plt, 1:length(Sr), Sr, label=@sprintf("reduced A_od (9 free, cond=%.0f)", Sr[1]/Sr[end]),
    marker=(:diamond,6), color=:darkorange)
# mark the struct/A_od cliff
hline!(plt, [Sf[6], Sf[7]], color=:gray, ls=:dash, alpha=0.5, label="")
annotate!(plt, 1.2, 0.5, text("6 structural dirs (μ,γ,γ′) ↑", 8, :left, :steelblue))
annotate!(plt, 9.5, 0.32, text("weakly-identified A_od tail ↓", 8, :left, :gray))
savefig(plt, joinpath(OUT,"sv_spectrum.png"))
println("wrote ", joinpath(OUT,"sv_spectrum.png"))
