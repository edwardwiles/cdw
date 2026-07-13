import KNITRO
include("/bbkinghome/edav/gravity_robustness/trade_robustness_modular/cc_algo/knitro_compat.jl")
optf = "ek_inner_loop_options.opt"
# time KN_new + load_param_file + free (the per-inner-solve fixed overhead)
function newloadfree()
    kc = KNITRO.KN_new(); KNITRO.KN_load_param_file(kc, optf); KNITRO.KN_free(kc)
end
newloadfree() # warmup
n=30; t=@elapsed (for _ in 1:n; newloadfree(); end)
println("PBK KN_new+load_param_file+free : ", round(t/n*1e3;digits=2), " ms/call  (opt file = ", filesize(optf), " B)")
# time just KN_new + free (no param file)
function newfree(); kc=KNITRO.KN_new(); KNITRO.KN_free(kc); end
newfree(); t2=@elapsed (for _ in 1:n; newfree(); end)
println("PBK KN_new+free (no param load): ", round(t2/n*1e3;digits=2), " ms/call")
println("PBK => param-file load costs ~", round((t-t2)/n*1e3;digits=2), " ms per inner solve")
