* ============================================================================
* config.do  --  paths and settings for the CDW gravity pipeline
*
* EDIT THE PATHS BELOW to relocate. Everything else in the pipeline reads these
* globals, so this is the single place to change when moving machines.
* ============================================================================
version 16.0
set more off

* --- repo and data roots ---------------------------------------------------
global REPO  "/bbkinghome/edav/cdw"
* DATA holds the raw inputs and the Teti work/ pickles (outside the repo, not committed)
global DATA  "/bbkinghome/edav/gravity_robustness/tariff_build"

* --- raw inputs ------------------------------------------------------------
global ICIO_DIR "$DATA/raw/icio"                                          // folder of yearly OECD ICIO csvs (YYYY.csv)
global WB_CSV   "$DATA/raw/wb/API_SL.TLF.TOTL.IN_DS2_en_csv_v2_761.csv"    // World Bank total labour force
global WORK     "$DATA/work"                                              // Teti/BACI pickles used by python/build_tau.py

* --- outputs ---------------------------------------------------------------
global TMP "$REPO/output/intermediate"     // intermediate .dta
global OUT "$REPO/output"                  // final csvs (pi, tau, L, gravity)
cap mkdir "$OUT"
cap mkdir "$TMP"

* --- run settings ----------------------------------------------------------
global YEAR 2016
* python interpreter used for the Teti tau step
global PYTHON "python3"

* The fixed node set (19 named + ROW) lives in $REPO/countries.csv and is the
* single source of truth for ordering and membership across all steps.
