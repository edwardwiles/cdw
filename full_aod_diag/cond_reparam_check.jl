# Precise check for Q1: what actually changes cond(J_free)?
#  (1) raw
#  (2) ORTHOGONAL double-difference (ANOVA) rotation of the A-block  -> should be INVARIANT
#  (3) column equilibration (diagonal rescale each param by 1/‖col‖) -> identification-neutral fix
#  (4) DROP the FE directions (keep only ΔΔ)                          -> a RESTRICTION, not a reparam
using JLD2, LinearAlgebra, Printf, Statistics
d = load(joinpath(@__DIR__,"out","cond_full.jld2"))
Jf = d["Jf"]; free = d["free"]; group = d["group"]; θ = d["θ"]
gfree = [group[i] for i in free]
D = 4; Aod_offset = 3 + D
# indices within `free` that are A, and their (o,d)
Apos = Tuple{Int,Int}[]; Acol = Int[]
for (k,i) in enumerate(free)
    if group[i]==:A
        o = ((i-Aod_offset-1) % D)+1; dd = ((i-Aod_offset-1) ÷ D)+1
        push!(Apos,(o,dd)); push!(Acol,k)
    end
end
Scol = [k for k in 1:length(free) if gfree[k]==:struct]
condof(M) = (s=svd(M).S; s[1]/s[end])
@printf("(1) raw cond(J_free)                                  = %.1f\n", condof(Jf))

# (3) column equilibration (identification-neutral diagonal reparam)
cn = [norm(Jf[:,k]) for k in 1:size(Jf,2)]
Jeq = Jf * Diagonal(1 ./ cn)
@printf("(3) column-equilibrated cond (per-param rescale)      = %.1f\n", condof(Jeq))

# (2) ORTHOGONAL ANOVA rotation of ONLY the A block.
# Build an orthonormal basis of the 12-dim free-A space split into
# {origin-FE, dest-FE, double-diff(interaction)} on the 3×4 support (means over the free grid).
nA = length(Acol)
os = sort(unique(first.(Apos))); ds = sort(unique(last.(Apos)))
# raw indicator design for a two-way ANOVA on the (o,d) grid, then Gram-Schmidt into blocks
function vec_on_support(f)  # f(o,d)->scalar  → length-nA vector aligned with Acol order
    [f(Apos[j]...) for j in 1:nA]
end
gm  = vec_on_support((o,dd)->1.0)
orig = [vec_on_support((o,dd)-> (o==oo ? 1.0 : 0.0)) for oo in os]
dest = [vec_on_support((o,dd)-> (dd==dd0 ? 1.0 : 0.0)) for dd0 in ds]
B = hcat(gm, orig..., dest...)          # spans grand+origin+dest (the FE part)
Qfe = Matrix(qr(B).Q)[:, 1:rank(B; rtol=1e-9)]     # orthonormal basis of the FE subspace
# interaction = orthogonal complement within the A space
Qfull = Matrix(qr(hcat(Qfe, randn(nA, nA))).Q)      # complete to full orthonormal basis
Qdd = Qfull[:, size(Qfe,2)+1:end]                   # ΔΔ (interaction) basis
Q = hcat(Qfe, Qdd)                                  # 12×12 orthogonal reparam of the A block
@assert isapprox(Q'Q, I; atol=1e-8)
JA = Jf[:, Acol]
Jrot = copy(Jf); Jrot[:, Acol] = JA * Q             # orthogonal reparam of A coords only
@printf("(2) after ORTHOGONAL ΔΔ/FE rotation of A block        = %.1f   (invariant ⇒ reparam alone does nothing)\n", condof(Jrot))
@printf("    #FE dirs=%d  #ΔΔ dirs=%d\n", size(Qfe,2), size(Qdd,2))

# (4) DROP the FE directions: keep struct + ΔΔ(interaction) only  (a RESTRICTION)
Jdrop = hcat(Jf[:, Scol], JA*Qdd)
@printf("(4) DROP FE dirs, keep struct+ΔΔ only (RESTRICTION)   = %.1f   (this is what the report's 125 really was)\n", condof(Jdrop))

# (4b) column-equilibrate the dropped-FE problem too
cnd = [norm(Jdrop[:,k]) for k in 1:size(Jdrop,2)]
@printf("(4b) drop-FE + column equilibration                   = %.1f\n", condof(Jdrop*Diagonal(1 ./ cnd)))

# how big is the sensitivity mismatch the equilibration removes?
@printf("\ncol-norm ranges: struct %.2f–%.2f   A %.3f–%.3f  (ratio ≈ %.0f×)\n",
    minimum(cn[Scol]),maximum(cn[Scol]), minimum(cn[Acol]),maximum(cn[Acol]),
    median(cn[Scol])/median(cn[Acol]))
