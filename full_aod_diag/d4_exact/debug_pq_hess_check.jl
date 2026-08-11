include(joinpath(@__DIR__, "debug_pq_fg_check.jl"))   # reuses everything already built there (ctx, st, aug, x0, g, obj, hess_ctx not yet built)

println("\n=== Hessian check ===")
hess_ctx = PairwiseQuantileCoreHessCtx(ncore1 + 1, aug.op, mass_state, aug.core_cf_ref)
# NCORE here = ncore1+1 (the +1 is zeta) -- matches WinnerPairHessCtx's own H_EE sizing convention (n=1+ncolI, ncolI=ncore1)

n = obj.outer_constr_index
obj.arg0 .= st.arg0   # already synced by the functor call in debug_pq_fg_check.jl, but be explicit
hess_cb = pairwisequantile_hess_cb_builder(hess_ctx)

# fake KNITRO evalResult/evalRequest shims (only .hess / .x are read by the callback)
struct FakeEvalRequest
    x::Vector{Float64}
end
mutable struct FakeEvalResult
    hess::Vector{Float64}
end
npacked = n * (n + 1) ÷ 2
evalResult = FakeEvalResult(zeros(npacked))
evalRequest = FakeEvalRequest(x0)
hess_cb(nothing, nothing, evalRequest, evalResult, obj)

Hpacked = evalResult.hess
Hdense = zeros(n, n)
k = 0
for i in 1:n, j in i:n
    global k += 1
    Hdense[i, j] = Hpacked[k]; Hdense[j, i] = Hpacked[k]
end
println("Hessian built OK, size ", size(Hdense), " max abs entry = ", maximum(abs.(Hdense)))

# finite-difference the GRADIENT to get an independent Hessian estimate, a handful of rows
# (full n=130 x 130 FD would be 130*2=260 extra FG calls -- cheap enough, do all rows)
h = 1e-5
Hfd = zeros(n, n)
gtmp = zeros(n)
for i in 1:n
    xp = copy(x0); xp[i] += h
    xm = copy(x0); xm[i] -= h
    gp = zeros(n); st(xp, gp)
    gm = zeros(n); st(xm, gm)
    Hfd[:, i] .= (gp .- gm) ./ (2h)
end
Hfd_sym = 0.5 .* (Hfd .+ Hfd')

err = maximum(abs.(Hdense .- Hfd_sym))
relerr = err / maximum(abs.(Hfd_sym))
println("max |H_analytic - H_fd| = ", err, "   relative to max|H_fd| = ", relerr)

# report worst offending block for diagnosis if it fails
if err > 1e-3
    diffmat = abs.(Hdense .- Hfd_sym)
    (val, idx) = findmax(diffmat)
    println("worst entry at (", idx[1], ",", idx[2], ") analytic=", Hdense[idx], " fd=", Hfd_sym[idx])
end
println(err < 1e-3 ? "HESSIAN CHECK: PASS" : "HESSIAN CHECK: FAIL")
