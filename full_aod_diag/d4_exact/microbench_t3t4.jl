# Side-by-side A/B of the T3/T4 scatter strategies at real D=20/W=100,000 dimensions, in ONE
# process, so the comparison is immune to the machine-load differences that make cross-run
# comparisons unreliable here.
using Printf
const W = 100_000; const D = 20
combs(n,k) = binomial(n,k)

function old_scatter!(T3, T4, tri, quad, bin, h, nlast, nt, T3loc_all, T4loc_all)
    fill!(T3, 0.0); fill!(T4, 0.0)
    for t in 1:nt; fill!(T3loc_all[t], 0.0); fill!(T4loc_all[t], 0.0); end
    Threads.@threads :static for tid in 1:nt
        lo = 1 + div((tid-1)*W, nt); hi = div(tid*W, nt)
        T3loc = T3loc_all[tid]; T4loc = T4loc_all[tid]
        @inbounds for w in lo:hi
            hw = h[w]
            for k in 1:length(tri)
                (o,p,q) = tri[k]
                a = bin[w,o]; a > nlast && continue
                b = bin[w,p]; b > nlast && continue
                c = bin[w,q]; c > nlast && continue
                T3loc[a,b,c,k] += hw
            end
            for k in 1:length(quad)
                (o1,o2,o3,o4) = quad[k]
                a = bin[w,o1]; a > nlast && continue
                b = bin[w,o2]; b > nlast && continue
                c = bin[w,o3]; c > nlast && continue
                d = bin[w,o4]; d > nlast && continue
                T4loc[a,b,c,d,k] += hw
            end
        end
    end
    for tid in 1:nt; T3 .+= T3loc_all[tid]; T4 .+= T4loc_all[tid]; end
end

function new_scatter!(T3, T4, tri, quad, bin, h, nlast)
    fill!(T3, 0.0)
    Threads.@threads :dynamic for k in 1:length(tri)
        (o,p,q) = tri[k]
        @inbounds for w in 1:W
            a = bin[w,o]; a > nlast && continue
            b = bin[w,p]; b > nlast && continue
            c = bin[w,q]; c > nlast && continue
            T3[a,b,c,k] += h[w]
        end
    end
    fill!(T4, 0.0)
    Threads.@threads :dynamic for k in 1:length(quad)
        (o1,o2,o3,o4) = quad[k]
        @inbounds for w in 1:W
            a = bin[w,o1]; a > nlast && continue
            b = bin[w,o2]; b > nlast && continue
            c = bin[w,o3]; c > nlast && continue
            d = bin[w,o4]; d > nlast && continue
            T4[a,b,c,d,k] += h[w]
        end
    end
end

nt = Threads.nthreads()
println("threads = ", nt, "   W = ", W, "   D = ", D)
tri  = [(o,p,q) for o in 1:D-2 for p in o+1:D-1 for q in p+1:D]
quad = [(a,b,c,d) for a in 1:D-3 for b in a+1:D-2 for c in b+1:D-1 for d in c+1:D]
@printf("C(D,3) = %d   C(D,4) = %d\n\n", length(tri), length(quad))

for L in (5, 10)
    nc = L-1; nlast = UInt8(nc)
    bin = UInt8.(rand(1:L, W, D)); h = rand(W) .+ 0.05
    T3a = zeros(nc,nc,nc,length(tri));      T4a = zeros(nc,nc,nc,nc,length(quad))
    T3b = similar(T3a);                     T4b = similar(T4a)
    T3loc = [zeros(nc,nc,nc,length(tri)) for _ in 1:nt]
    T4loc = [zeros(nc,nc,nc,nc,length(quad)) for _ in 1:nt]
    per_thread_mb = (length(T3a)+length(T4a))*8/2^20
    old_scatter!(T3a,T4a,tri,quad,bin,h,nlast,nt,T3loc,T4loc)   # warm
    new_scatter!(T3b,T4b,tri,quad,bin,h,nlast)                  # warm
    agree = maximum(abs, T4a .- T4b) / max(maximum(abs,T4a), eps())
    t_old = @elapsed old_scatter!(T3a,T4a,tri,quad,bin,h,nlast,nt,T3loc,T4loc)
    t_new = @elapsed new_scatter!(T3b,T4b,tri,quad,bin,h,nlast)
    @printf("L=%2d  table cells %6.2f M   per-thread scratch %6.1f MB x %d = %5.2f GB\n",
            L, (length(T3a)+length(T4a))/1e6, per_thread_mb, nt, nt*per_thread_mb/1024)
    @printf("      OLD draws-outer + per-thread copies   %7.3f s\n", t_old)
    @printf("      NEW combos-outer + shared table       %7.3f s   (%.2fx)\n", t_new, t_old/t_new)
    @printf("      max relative disagreement (summation order only): %.3e\n\n", agree)
end
