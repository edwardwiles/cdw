* ============================================================================
* master.do  --  run the full CDW pipeline for one year, producing the three
*                Julia inputs (pi, tau, L) plus the gravity estimates.
*
* Usage:   cd <repo>/stata ;  do master.do
* Edit paths/year in ../config.do first.
*
* Steps:
*   01_clean_icio   raw OECD ICIO  -> gross flow matrix (+ goods/services shares)
*   02_build_pi     flow matrix    -> block import-share matrix  pi
*   build_tau.py    Teti + BACI    -> block tariff matrix        tau   (python)
*   03_build_L      WB labour      -> block labour vector        L
*   04_gravity      pi, tau        -> trade-elasticity estimates
* ============================================================================
version 16.0
clear all
set more off

do "../config.do"

do "01_clean_icio.do"
do "02_build_pi.do"

* tau (python step): pass the config paths through to build_tau.py
shell $PYTHON "$REPO/python/build_tau.py" --repo "$REPO" --work "$WORK" --out "$OUT" --tmp "$TMP" --year $YEAR

do "03_build_L.do"
do "04_gravity.do"

noi di as result "==============================================================="
noi di as result " CDW pipeline complete for $YEAR."
noi di as result " Outputs in $OUT :  pi_${YEAR}.csv  tau_${YEAR}.csv  L_${YEAR}.csv"
noi di as result "                    gravity_coef_${YEAR}.csv  gravity_${YEAR}.tex"
noi di as result "==============================================================="
