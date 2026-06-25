* ============================================================================
* 04_gravity.do  --  gravity / trade-elasticity estimates.
*
*   OLS:      ln(pi_{o,d}) on ln(tau_{o,d})  with exporter & importer FE  ->  -theta
*   FRA 2SLS: restrict to French imports (d = fra); regress ln(pi_{o,fra}) on the
*             endogenous ln(w_o * tau_{o,fra}), instrumented by ln(tau_{o,fra}).
*             w_o = balanced-trade wage from build_wages.py (flows only).
* Four versions per tariff measure: OLS and 2SLS, each with and without ROW.
* SELF-TRADE (the domestic diagonal o==d) is RETAINED -- it is part of the
* structural model and is available from ICIO.
* Reported for the standard tariff and the goods-share-adjusted tariff.
* ============================================================================
version 16.0
local y $YEAR
local fra 2

use "$TMP/pi_long_`y'.dta", clear
keep exp_id imp_id pi
preserve
    import delimited "$TMP/tau_long_`y'.csv", varnames(1) clear
    tempfile tau
    save `tau'
restore
merge 1:1 exp_id imp_id using `tau', nogen keep(match) keepusing(tau tau_adj)
preserve
    import delimited "$TMP/w_long_`y'.csv", varnames(1) clear
    rename node_id exp_id
    tempfile wage
    save `wage'
restore
merge m:1 exp_id using `wage', nogen keep(match) keepusing(w)

gen ln_pi       = ln(pi)
gen ln_tau      = ln(tau)
gen ln_tau_adj  = ln(tau_adj)
gen ln_comb     = ln(w * tau)          // ln(w_o (1+t))
gen ln_comb_adj = ln(w * tau_adj)      // ln(w_o (1+t_adj))
drop if missing(ln_pi) | pi <= 0       // self-trade (o==d) kept

eststo clear
* ----- standard tariff: OLS(+/-ROW), 2SLS(+/-ROW) -----
eststo t_ols_all: reghdfe ln_pi ln_tau, absorb(exp_id imp_id) vce(robust)
estadd local incrow "Y" : t_ols_all
eststo t_ols_nr:  reghdfe ln_pi ln_tau if exp_id!=20 & imp_id!=20, absorb(exp_id imp_id) vce(robust)
estadd local incrow "N" : t_ols_nr
eststo t_iv_all:  ivreg2 ln_pi (ln_comb = ln_tau) if imp_id==`fra', robust
estadd local incrow "Y" : t_iv_all
eststo t_iv_nr:   ivreg2 ln_pi (ln_comb = ln_tau) if imp_id==`fra' & exp_id!=20, robust
estadd local incrow "N" : t_iv_nr
* ----- tariff x goods share -----
eststo a_ols_all: reghdfe ln_pi ln_tau_adj, absorb(exp_id imp_id) vce(robust)
estadd local incrow "Y" : a_ols_all
eststo a_ols_nr:  reghdfe ln_pi ln_tau_adj if exp_id!=20 & imp_id!=20, absorb(exp_id imp_id) vce(robust)
estadd local incrow "N" : a_ols_nr
eststo a_iv_all:  ivreg2 ln_pi (ln_comb_adj = ln_tau_adj) if imp_id==`fra', robust
estadd local incrow "Y" : a_iv_all
eststo a_iv_nr:   ivreg2 ln_pi (ln_comb_adj = ln_tau_adj) if imp_id==`fra' & exp_id!=20, robust
estadd local incrow "N" : a_iv_nr

local mods t_ols_all t_ols_nr t_iv_all t_iv_nr a_ols_all a_ols_nr a_iv_all a_iv_nr
esttab `mods' using "$OUT/gravity_`y'.tex", replace ///
    se star(* 0.10 ** 0.05 *** 0.01) booktabs ///
    rename(ln_tau b ln_tau_adj b ln_comb b ln_comb_adj b) keep(b) ///
    coeflabels(b "\$-\theta\$") ///
    mgroups("Standard tariff" "Tariff \(\times\) goods share", pattern(1 0 0 0 1 0 0 0) ///
        prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
    mtitles("OLS" "OLS" "2SLS" "2SLS" "OLS" "OLS" "2SLS" "2SLS") ///
    scalars("incrow Incl. ROW") sfmt(%9.0g) nonumbers obslast ///
    note("OLS: exporter and importer FE. 2SLS: French imports, ln(w(1+t)) instrumented by ln(1+t)." ///
         "Self-trade (domestic diagonal) retained. Robust SE in parentheses.")
esttab `mods', keep(b) rename(ln_tau b ln_tau_adj b ln_comb b ln_comb_adj b) se ///
    mtitles("OLS+" "OLS-" "IV+" "IV-" "aOLS+" "aOLS-" "aIV+" "aIV-")

* tidy coefficient csv
preserve
clear
set obs 8
gen year = `y'
gen tariff = ""
gen estimator = ""
gen incl_row = ""
gen beta = .
gen se = .
gen n = .
local i = 0
foreach mod of local mods {
    local ++i
    estimates restore `mod'
    local key "ln_tau"
    if substr("`mod'",1,1)=="a" local key "ln_tau_adj"
    if strpos("`mod'","iv") {
        local key = cond(substr("`mod'",1,1)=="a", "ln_comb_adj", "ln_comb")
    }
    replace tariff    = cond(substr("`mod'",1,1)=="a","goods_adj","standard") in `i'
    replace estimator = cond(strpos("`mod'","iv"),"2SLS","OLS") in `i'
    replace incl_row  = cond(strpos("`mod'","_nr"),"N","Y") in `i'
    replace beta = _b[`key'] in `i'
    replace se   = _se[`key'] in `i'
    replace n    = e(N) in `i'
}
export delimited using "$OUT/gravity_coef_`y'.csv", replace
restore
noi di as result "[04] saved gravity_`y'.tex and gravity_coef_`y'.csv"
