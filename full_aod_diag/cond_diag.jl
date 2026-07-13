# ================================================================================
# Full-A_od conditioning diagnostic (Section 4 of the research prompt)
# Measures the identification / degeneracy structure of the OUTER parameter vector θ
# via the moment-map Jacobian J = ∂ E_{F*}[g] / ∂θ (d moments × l params).
#
# Additive, read-only w.r.t. the core pipeline: builds objects exactly like diag.jl,
# computes J by autodiff, restricts to FREE columns, and analyses SVD / rank / null-space.
# Env flags:
#   RED=0/1        0 = FULL A_od (all 12 free), 1 = REDUCED (only target-destination col free)
#   THETA=init/sol which point to evaluate at (sol reads a saved cc_output θ if present)
# Writes: full_aod_diag/out/cond_<tag>.txt  and a JLD2 with J and the spectra.
# ================================================================================
using Parameters, Base.Threads, Random, Dates, DelimitedFiles, Printf
using Distributions, Statistics, SpecialFunctions, InvertedIndices
using NLsolve, ForwardDiff, Calculus, LinearAlgebra, Plots, JLD2
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT,"setup/include_setup.jl")); include(joinpath(ROOT,"prestep/include_prestep.jl"))
include(joinpath(ROOT,"prepare_cc/include_prepare_cc.jl")); include(joinpath(ROOT,"moments/include_moments.jl"))
include(joinpath(ROOT,"cc_algo/include_cc_algo.jl")); include(joinpath(ROOT,"lfd/include_lfd.jl")); include(joinpath(ROOT,"misc/include_misc.jl"))
using .CounterfactualSensitivity; const CS = CounterfactualSensitivity

const RED  = get(ENV, "RED", "0") == "1"
const TAG  = (RED ? "reduced" : "full") * get(ENV, "CTAG", "")
const OUT  = joinpath(@__DIR__, "out"); isdir(OUT) || mkpath(OUT)
io = open(joinpath(OUT, "cond_$(TAG).txt"), "w")
logln(s...) = (println(stdout, s...); println(io, s...); flush(io))

params = (server=1,user=2,fakeData=1,DFake=4,seedFakeData=889,counterType=1,counterExplicit=0,θHat=0,σHat=2.5,baseIndex=2,W=8000,seedU=888,importanceSampling=0,importanceSamplingFactor=2,stratifiedSampling=0,IndMomentOrder=5,θConstant=0,gravMoment=1,localGravityMoment=0,localGravityCrossMoment=0,GravityMomentFirstApproach=0,sameMarginalsMoment=0,NoScalingforSameMartingale=1,useCDFforMarginalMatching=0,independenceMoment=0,momentOrder=5,momentOrderForBaseIndex=50,ForceFrechetMarginal=0,OuterScaling=1,useParallel=0,usePMM=0,PMMGammaOnly=0,NormalizeMoments=0,useConfidenceIntervals=0,ConfidenceLevel=0.05,δGridType=0,δ_ref=1,refIndex1=1,OuterLoop=1,UoModel=1,use_Jacobian=0,calc_δ_star_initial=1,Jac_W=8000,theta_init=0,runLFD=1,runLFDCounterFactual=1)
so=master_setup(params); up=(; params..., D=so.D, EK_moments! = EK_moments!, EK_moments_Jacobian! = EK_moments_Jacobian!)
checkParams(up); ps=master_prestep(so.data,so.counters,up); pp=master_prepare_cc(so.data,so.counters,ps,up)
@unpack θ_initial,U,γ,outer_constr_index,inequality_index,nTotalMoments,complement_index = pp
D=so.D; σ=params.σHat; bi=params.baseIndex
θ = copy(θ_initial)
# optionally evaluate conditioning at a saved solution θ (Section 4a: near-solution point)
if haskey(ENV,"THETAJLD")
    d = load(ENV["THETAJLD"]); θ .= d[get(ENV,"THETAKEY","θ_up")]
    println(">>> loaded θ from ", ENV["THETAJLD"], " key=", get(ENV,"THETAKEY","θ_up"))
end

obj = PsiObjectiveBundleImplicit(δ=1.0,find_smallest=true,γ=γ,(moments!)=EK_moments!,
    moments_jacobian! = error, d=nTotalMoments, outer_constr_index=outer_constr_index,
    inequality_index=inequality_index, complement_index=complement_index, l=length(θ),
    U=U, N=params.Jac_W, lower_limit=-50,
    outer_loop_opt="csw_outer_loop_settings_cluster.opt", inner_loop_opt="ek_inner_loop_options.opt")

# ---------- parameter layout & FREE/FIXED masks (mirrors ccOuter.jl bounds logic) ----------
l = length(θ)
Aod_offset = 3 + D                       # counterType=1, OuterScaling=1  => A block starts at Aod_offset+1
Aidx(o,d) = Aod_offset + o + (d-1)*D     # global θ index of A[o,d]  (column-major, matches diag.jl)
# structural free params: μ (1). σ (2) fixed. γ_θ[1..D] (3..2+D). γ'_θ (3+D = Aod_offset).
fixed = falses(l)
fixed[2] = true                          # σ fixed
for d in 1:D; fixed[Aidx(1,d)] = true; end   # A[1,d]=1 normalization (all variants)
if RED                                   # reduced: fix every A column d != baseIndex
    for d in 1:D, o in 1:D
        d == bi && continue
        fixed[Aidx(o,d)] = true
    end
end
free = findall(.!fixed)

# label + group each param
label = Vector{String}(undef, l); group = Vector{Symbol}(undef, l)
label[1]="mu"; group[1]=:struct
label[2]="sigma(FIX)"; group[2]=:struct
for d in 1:D; label[2+d]="gamma[$d]"; group[2+d]=:struct; end
label[Aod_offset]="gammaP"; group[Aod_offset]=:struct
for o in 1:D, d in 1:D
    label[Aidx(o,d)]="A[$o,$d]"; group[Aidx(o,d)]=:A
end

logln("="^78)
logln("CONDITIONING DIAGNOSTIC  variant=$(TAG)   D=$D  baseIndex=$bi  σ=$σ")
logln("l=$l total params, ", length(free), " FREE  (", count(f->group[f]==:A, free), " of them A_od)")
logln("nTotalMoments d = $nTotalMoments   outer_constr_index = $outer_constr_index")
logln("free indices: ", free)
logln("="^78)

# ---------- moment-map Jacobian J = ∂E[g]/∂θ  and κ gradient, by autodiff ----------
# reuse the pipeline's own autodiff filler, then average per-draw jacobian over draws.
CS.calculate_jac_θ_autodiff!(obj, θ)
N = obj.N
# jac_h[draw, Hcol, param];  Hcol 1=κ, 2=ones, 3:2+d = moments
Jdraw = @view obj.jac_h[1:N, 3:2+nTotalMoments, :]     # N × d × l
J = reshape(mean(Jdraw, dims=1), nTotalMoments, l)     # d × l  moment-expectation Jacobian
gradk = vec(mean(@view(obj.jac_h[1:N, 1, :]), dims=1)) # l    ∂κ/∂θ (κ const across draws for implicit)

# also grab the moment values E[g] at θ for reference (feasibility of F* moments)
obj.moments!(@view(obj.H[:,1]), CS.select_G_from_H(obj,obj.H), θ, obj.U, obj)
Eg = vec(mean(obj.H[:, 3:end], dims=1))

logln("\n-- moment values E_{F*}[g] (should be ~0 for matched moments) --")
logln("  max|E[g]| over the ", nTotalMoments, " moments = ", round(maximum(abs.(Eg)); sigdigits=3))

# ---------- restrict to free columns; report scaling by group ----------
Jf = J[:, free]
logln("\n-- parameter & gradient SCALING by group (Section 4e) --")
for grp in (:struct, :A)
    idxs = [i for i in free if group[i]==grp]
    isempty(idxs) && continue
    pv = abs.(θ[idxs]); gv = abs.(gradk[idxs])
    colnorm = [norm(J[:,i]) for i in idxs]
    logln("  group=$grp  n=$(length(idxs))")
    logln("     |θ|      min/med/max = ", round.((minimum(pv),median(pv),maximum(pv)); sigdigits=3))
    logln("     |∂κ/∂θ|  min/med/max = ", round.((minimum(gv),median(gv),maximum(gv)); sigdigits=3))
    logln("     ‖J col‖  min/med/max = ", round.((minimum(colnorm),median(colnorm),maximum(colnorm)); sigdigits=3))
end

# ---------- SVD / rank / conditioning (Section 4a,c) ----------
F = svd(Jf)
S = F.S
smax = S[1]; tol = smax * 1e-8
r = count(>(tol), S)
nn_tol = smax * 1e-6                       # "near-null" threshold
nnull = count(<(nn_tol), S)
logln("\n-- SVD of free moment Jacobian  J_free  (", size(Jf,1), "×", size(Jf,2), ")  [Section 4a,c] --")
logln("  rank(tol=σmax·1e-8) = $r  of ", length(free), " free params")
logln("  σmax = ", round(smax; sigdigits=4), "   σmin = ", round(S[end]; sigdigits=4))
logln("  cond(J_free) = σmax/σmin = ", round(smax/max(S[end],eps()); sigdigits=4))
logln("  #singular values < σmax·1e-6 (near-null dim) = $nnull")
logln("  full singular spectrum (log10):")
for (k,s) in enumerate(S); logln("     σ[$k] = ", @sprintf("%.4e", s), "   log10=", round(log10(max(s,1e-300));digits=3)); end

# ---------- interpret near-null right-singular vectors in ln A coordinates (Section 4a) ----------
# convert free A-columns to d(lnA) coords: dlnA = dA / A  ⇒ scale null-vector A-entries by A.
logln("\n-- near-null direction structure (Section 4a: FE flats in ln A_od?) --")
Afree = [i for i in free if group[i]==:A]
# helper: given a free-space vector v (length=#free), return its ln-A picture as D×D (0 where fixed)
function lnA_picture(v)
    M = zeros(D,D)
    for (k,i) in enumerate(free)
        group[i]==:A || continue
        o = ((i-Aod_offset-1) % D)+1; d = ((i-Aod_offset-1) ÷ D)+1
        M[o,d] = v[k] * θ[i]              # dA * A? -> chain rule dlnA = dA/A, but v is in A units; dlnA≈ v/A.
    end
    return M
end
# NOTE: right-singular vectors are unit in θ(A-level) space. dlnA = v ./ A elementwise.
function lnA_dir(v)
    M = zeros(D,D)
    for (k,i) in enumerate(free)
        group[i]==:A || continue
        o = ((i-Aod_offset-1) % D)+1; d = ((i-Aod_offset-1) ÷ D)+1
        M[o,d] = v[k] / θ[i]
    end
    return M
end
# decompose a D×D matrix (on the FREE support) into origin-FE + dest-FE + double-demeaned energy
function fe_decomp(Mv, support)
    # support: BitMatrix of which (o,d) are free
    vals = Mv[support]
    tot = norm(vals); tot < 1e-300 && return (0.0,0.0,0.0,0.0)
    # origin means over free entries in each row, dest means over free entries in each col
    rowmean = zeros(D); colmean = zeros(D)
    for o in 1:D
        r = [Mv[o,d] for d in 1:D if support[o,d]]; rowmean[o] = isempty(r) ? 0.0 : mean(r)
    end
    for d in 1:D
        c = [Mv[o,d] for o in 1:D if support[o,d]]; colmean[d] = isempty(c) ? 0.0 : mean(c)
    end
    gm = mean(vals)
    orig = zeros(D,D); dest = zeros(D,D); dd = zeros(D,D)
    for o in 1:D, d in 1:D
        support[o,d] || continue
        orig[o,d] = rowmean[o]-gm
        dest[o,d] = colmean[d]-gm
        dd[o,d]   = Mv[o,d]-rowmean[o]-colmean[d]+gm
    end
    # energies (note origin/dest/dd not perfectly orthogonal on a partial support, report raw norms)
    return (norm(orig[support])/tot, norm(dest[support])/tot, norm(dd[support])/tot, tot)
end
support = falses(D,D)
for i in Afree
    o = ((i-Aod_offset-1) % D)+1; d = ((i-Aod_offset-1) ÷ D)+1; support[o,d]=true
end
logln("  (energy fractions of each near-null dir, in dlnA coords, on the free-A support)")
logln("  rank | σ_k       | A-energy frac | originFE | destFE | doubleDemean | struct-energy")
V = F.V
for k in length(S):-1:max(1,length(S)-6)   # the smallest few singular directions
    v = V[:,k]
    Apart = [v[j] for (j,i) in enumerate(free) if group[i]==:A]
    Spart = [v[j] for (j,i) in enumerate(free) if group[i]==:struct]
    aener = norm(Apart); sener = norm(Spart)
    Md = lnA_dir(v)
    (fo,fd,fdd,_) = fe_decomp(Md, support)
    logln("   ", lpad(k,3), " | ", @sprintf("%.3e",S[k]), " |    ",
          rpad(round(aener;digits=3),8), " | ", rpad(round(fo;digits=3),7), " | ",
          rpad(round(fd;digits=3),6), " | ", rpad(round(fdd;digits=3),11), " | ", round(sener;digits=3))
end

# ---------- κ-gradient flatness along the near-null space (Section 4b) ----------
gk_f = gradk[free]
logln("\n-- κ-objective sensitivity along the near-null directions (Section 4b) --")
logln("  ‖∇κ (free)‖ = ", round(norm(gk_f); sigdigits=4))
for k in length(S):-1:max(1,length(S)-6)
    proj = abs(dot(gk_f, V[:,k]))
    logln("   |∇κ · v_$k| = ", @sprintf("%.3e", proj), "   (σ_$k=", @sprintf("%.2e",S[k]), ")")
end

# ---------- Section 5 preview: does double-demeaning the A-block remove the null space? ----------
# Build a basis for the FREE θ space that replaces raw A directions by their double-demeaned
# picture, and re-SVD. If the FE flats are the near-null space, the reparam Jacobian's smallest
# singular value should jump up toward the struct-block scale.
logln("\n-- Section 5 preview: reparametrize free-A block to double-demeaned coords --")
# Columns of J for free A in dlnA units: Ja[:,k] corresponds to ∂E[g]/∂lnA[o,d] = A*∂/∂A
JA = zeros(nTotalMoments, length(Afree))
for (k,i) in enumerate(Afree); JA[:,k] = J[:,i]*θ[i]; end   # chain rule to d lnA
# double-demean operator on the free support: map full-A dlnA vector -> demeaned. On a rectangular
# free support this is only exact when the support is a full D×D block (FULL variant). For REDUCED
# (single column) demeaning across origins is the only nontrivial op.
# We construct P = projection matrix onto the double-demeaned subspace over the free (o,d) support.
nA = length(Afree)
# build index map
Apos = Dict{Tuple{Int,Int},Int}()
for (k,i) in enumerate(Afree)
    o = ((i-Aod_offset-1) % D)+1; d = ((i-Aod_offset-1) ÷ D)+1; Apos[(o,d)]=k
end
# demean matrix: y = x - rowmean(o) - colmean(d) + grandmean, restricted to free support
Dmat = zeros(nA, nA)
for (od1,k1) in Apos
    o1,d1 = od1
    rowset = [Apos[(o1,dd)] for dd in 1:D if haskey(Apos,(o1,dd))]
    colset = [Apos[(oo,d1)] for oo in 1:D if haskey(Apos,(oo,d1))]
    allset = collect(values(Apos))
    for k2 in 1:nA
        Dmat[k1,k2] += (k2==k1 ? 1.0 : 0.0)
        (k2 in rowset) && (Dmat[k1,k2] -= 1.0/length(rowset))
        (k2 in colset) && (Dmat[k1,k2] -= 1.0/length(colset))
        (k2 in allset) && (Dmat[k1,k2] += 1.0/length(allset))
    end
end
rP = rank(Dmat; rtol=1e-9)
logln("  free-A count = $nA;  rank of double-demean projector = $rP  (=(D-1)^2=$(((D-1)^2)) for a full block)")
# Effective reparam Jacobian columns living in the demeaned subspace:
JA_dd = JA * Dmat
# combine struct columns (unchanged) with demeaned-A columns, re-SVD
structfree = [i for i in free if group[i]==:struct]
Jstruct = J[:, structfree]
Jrep = hcat(Jstruct, JA_dd)
Srep = svd(Jrep).S
# keep only nonzero directions of the demeaned block for a fair conditioning number
Srep_nz = Srep[Srep .> Srep[1]*1e-10]
logln("  reparam Jacobian σmin(all)=", @sprintf("%.3e",Srep[end]),
      "   σmin(nonzero)=", @sprintf("%.3e",Srep_nz[end]),
      "   cond(nonzero)=", round(Srep[1]/Srep_nz[end]; sigdigits=4))
logln("  (compare raw-A cond(J_free) above; if the near-null was FE flats, the demeaned")
logln("   block drops the ", nA - rP, " FE directions and the remaining spectrum is well-conditioned.)")

# save numerics
@save joinpath(OUT,"cond_$(TAG).jld2") J gradk Eg free label group S V θ Jf Jrep Srep
logln("\nSaved: ", joinpath(OUT,"cond_$(TAG).jld2"))
logln("DONE ($TAG)")
close(io)
