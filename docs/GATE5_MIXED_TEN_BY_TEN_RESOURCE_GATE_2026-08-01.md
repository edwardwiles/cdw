# Gate 5: mixed-family ten-by-ten resource gate (2026-08-01)

Config: 5 cm_meanzc + 5 origin_zc concurrent fresh Julia
processes (10 total) x 10 Julia threads each, BLAS threads=8,
disjoint taskset core affinity (0-99 of this host's cores),
all running the all_optimized-backend single true-cold inner solve at W=100,000, K_mean=K_pair=3.

Total wall time for all 10 to complete: 142s.

```
RESULT family=cm_meanzc arm=all_optimized W=100000 pid=587456 ctx_build_s=62.931 solve_s=48.269 nStatus=0
RESULT family=cm_meanzc arm=all_optimized W=100000 pid=587457 ctx_build_s=64.548 solve_s=53.115 nStatus=0
RESULT family=cm_meanzc arm=all_optimized W=100000 pid=587458 ctx_build_s=61.806 solve_s=48.330 nStatus=0
RESULT family=cm_meanzc arm=all_optimized W=100000 pid=587459 ctx_build_s=64.112 solve_s=51.034 nStatus=0
RESULT family=cm_meanzc arm=all_optimized W=100000 pid=587460 ctx_build_s=59.779 solve_s=51.743 nStatus=0
RESULT family=origin_zc arm=all_optimized W=100000 pid=587462 ctx_build_s=62.493 solve_s=24.160 nStatus=0
RESULT family=origin_zc arm=all_optimized W=100000 pid=587463 ctx_build_s=64.924 solve_s=25.413 nStatus=0
RESULT family=origin_zc arm=all_optimized W=100000 pid=587464 ctx_build_s=64.003 solve_s=25.460 nStatus=0
RESULT family=origin_zc arm=all_optimized W=100000 pid=587465 ctx_build_s=59.246 solve_s=23.631 nStatus=0
RESULT family=origin_zc arm=all_optimized W=100000 pid=587466 ctx_build_s=61.120 solve_s=23.589 nStatus=0
```

fail_flag=0 (0 = all processes exited cleanly)
