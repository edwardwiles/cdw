* ============================================================================
* 04_gravity.do  --  structural gravity: ln(pi_{o,d}) on ln(tau_{o,d}) with
*                    exporter and importer fixed effects. The tau coefficient
*                    identifies -theta (the trade elasticity).
* Uses pi (step 02) and tau (build_tau.py). Writes a coefficient table.
* ============================================================================
version 16.0
local y $YEAR

use "$TMP/pi_long_`y'.dta", clear
keep exp_id imp_id pi
preserve
    import delimited "$TMP/tau_long_`y'.csv", varnames(1) clear
    tempfile tau
    save `tau'
restore
merge 1:1 exp_id imp_id using `tau', keep(match) nogen keepusing(tau)

gen ln_pi  = ln(pi)
gen ln_tau = ln(tau)
drop if missing(ln_pi) | missing(ln_tau) | pi <= 0

eststo clear
* (1) all 20 nodes
eststo m_all: reghdfe ln_pi ln_tau, absorb(exp_id imp_id) vce(robust)
estadd local IncludesROW "Y"
* (2) exclude ROW (node 20)
eststo m_norow: reghdfe ln_pi ln_tau if exp_id != 20 & imp_id != 20, absorb(exp_id imp_id) vce(robust)
estadd local IncludesROW "N"

esttab m_all m_norow using "$OUT/gravity_`y'.tex", replace ///
    se star(* 0.10 ** 0.05 *** 0.01) booktabs ///
    keep(ln_tau) coeflabels(ln_tau "ln(1+t)") ///
    scalars("IncludesROW Includes ROW" "r2 R-squared") ///
    mtitles("All nodes" "Excl. ROW") nonumbers obslast ///
    note("ln(pi) on ln(tau), exporter and importer FE; robust SE.")
esttab m_all m_norow, keep(ln_tau) se mtitles("All" "ExclROW")

* tidy csv of the headline coefficients
clear
set obs 2
gen year = `y'
gen spec = ""
gen beta_ln_tau = .
gen se = .
gen n = .
replace spec = "all"      in 1
replace spec = "excl_row" in 2
estimates restore m_all
replace beta_ln_tau = _b[ln_tau] in 1
replace se = _se[ln_tau] in 1
replace n = e(N) in 1
estimates restore m_norow
replace beta_ln_tau = _b[ln_tau] in 2
replace se = _se[ln_tau] in 2
replace n = e(N) in 2
export delimited using "$OUT/gravity_coef_`y'.csv", replace
noi di as result "[04] saved gravity_`y'.tex and gravity_coef_`y'.csv"
