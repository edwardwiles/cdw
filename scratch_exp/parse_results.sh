#!/usr/bin/env bash
cd /bbkinghome/edav/gravity_robustness/trade_robustness_modular/scratch_exp/results
printf "%-22s %6s %8s %8s %6s %6s %7s %7s %8s %6s %6s\n" TAG kappaLO kappaHI Ustat Lstat Uiters Liters Uinfeas Lsolv Utime Ltime
for L in *.log; do
  tag=${L%.log}; [ "$tag" = "profile_inner_flat" ] && continue
  # bounds from the run's own stdout (writedlm prints [.. .. ..]); fallback to grep the DR csv line
  lo=$(grep -oE "κ_lower[^0-9-]*[-0-9.e]+" "$L" | tail -1 | grep -oE "[-0-9.e]+$")
  hi=$(grep -oE "κ_upper[^0-9-]*[-0-9.e]+" "$L" | tail -1 | grep -oE "[-0-9.e]+$")
  U=$(grep "OUTER_SOLVE find_smallest=false" "$L" | tail -1)
  Ln=$(grep "OUTER_SOLVE find_smallest=true" "$L" | tail -1)
  gv(){ echo "$1" | grep -oE "$2=[-0-9.e]+" | head -1 | cut -d= -f2; }
  printf "%-22s %8s %8s %8s %6s %6s %7s %7s %8s %6s %6s\n" "$tag" \
    "$(gv "$Ln" obj)" "$(gv "$U" obj)" \
    "$(gv "$U" status)" "$(gv "$Ln" status)" \
    "$(gv "$U" outer_iters)" "$(gv "$Ln" outer_iters)" \
    "$(gv "$U" inner_infeas)/$(gv "$Ln" inner_infeas)" \
    "$(gv "$U" inner_solves)/$(gv "$Ln" inner_solves)" \
    "$(gv "$U" knitro_time_s)" "$(gv "$Ln" knitro_time_s)"
done
