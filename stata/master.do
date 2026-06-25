* ============================================================================
* master.do  --  run the full CDW pipeline for one year, producing the model
*                inputs (pi, tau, tau_adj, L, w) plus the gravity estimates.
*
* Usage:   cd <repo>/stata ;  do master.do
* Edit paths/year in ../config.do first.
*
* Steps:
*   01_clean_icio   raw OECD ICIO  -> gross flow matrix (+ goods/services shares)
*   02_build_pi     flow matrix    -> block import-share matrix  pi
*   build_tau.py    Teti + BACI    -> tariff matrices tau, tau_adj (goods-share adj)
*   03_build_L      WB labour      -> block labour vector  L
*   build_wages.py  pi, L          -> balanced-trade wages  w   (flows only)
*   04_gravity      pi, tau, w     -> trade-elasticity estimates (OLS + FRA 2SLS)
* ============================================================================
version 16.0
clear all
set more off

do "../config.do"

do "01_clean_icio.do"
do "02_build_pi.do"
shell $PYTHON "$REPO/python/build_tau.py"   --repo "$REPO" --work "$WORK" --out "$OUT" --tmp "$TMP" --year $YEAR
do "03_build_L.do"
shell $PYTHON "$REPO/python/build_wages.py" --out "$OUT" --tmp "$TMP" --year $YEAR
do "04_gravity.do"

noi di as result "==============================================================="
noi di as result " CDW pipeline complete for $YEAR."
noi di as result " Outputs in $OUT :  pi_${YEAR}.csv  tau_${YEAR}.csv  tau_adj_${YEAR}.csv"
noi di as result "                    L_${YEAR}.csv  w_${YEAR}.csv"
noi di as result "                    gravity_${YEAR}.tex  gravity_coef_${YEAR}.csv"
noi di as result "==============================================================="
