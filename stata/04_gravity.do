* ============================================================================
* 04_gravity.do  --  gravity / trade-elasticity estimates.
*
*   OLS:      ln(pi_{o,d}) on ln(tau_{o,d})  with exporter & importer FE  ->  -theta
*   FRA 2SLS: restrict to French imports (d = fra); regress ln(pi_{o,fra}) on the
*             endogenous ln(w_o * tau_{o,fra}), instrumented by ln(tau_{o,fra}).
*             w_o = balanced-trade wage from build_wages.py (flows only).
* Reported for the standard tariff and for the goods-share-adjusted tariff.
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
* exporter wage w_o
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
gen ln_comb     = ln(w * tau)          // ln(w_o (1+t))           [standard]
gen ln_comb_adj = ln(w * tau_adj)      // ln(w_o (1+t_adj))       [goods-share adj]
drop if missing(ln_pi) | pi <= 0
drop if exp_id==imp_id          // gravity uses international bilateral flows only (no domestic diagonal)

eststo clear
* ---- standard tariff ----
eststo t_all:   reghdfe ln_pi ln_tau, absorb(exp_id imp_id) vce(robust)
estadd local est "OLS" : t_all
estadd local samp "All" : t_all
eststo t_norow: reghdfe ln_pi ln_tau if exp_id!=20 & imp_id!=20, absorb(exp_id imp_id) vce(robust)
estadd local est "OLS" : t_norow
estadd local samp "Excl ROW" : t_norow
eststo t_iv:    ivreg2 ln_pi (ln_comb = ln_tau) if imp_id==`fra', robust
estadd local est "2SLS" : t_iv
estadd local samp "FRA imp" : t_iv
* ---- tariff x goods share ----
eststo a_all:   reghdfe ln_pi ln_tau_adj, absorb(exp_id imp_id) vce(robust)
estadd local est "OLS" : a_all
estadd local samp "All" : a_all
eststo a_norow: reghdfe ln_pi ln_tau_adj if exp_id!=20 & imp_id!=20, absorb(exp_id imp_id) vce(robust)
estadd local est "OLS" : a_norow
estadd local samp "Excl ROW" : a_norow
eststo a_iv:    ivreg2 ln_pi (ln_comb_adj = ln_tau_adj) if imp_id==`fra', robust
estadd local est "2SLS" : a_iv
estadd local samp "FRA imp" : a_iv

esttab t_all t_norow t_iv a_all a_norow a_iv using "$OUT/gravity_`y'.tex", replace ///
    se star(* 0.10 ** 0.05 *** 0.01) booktabs ///
    rename(ln_tau b ln_tau_adj b ln_comb b ln_comb_adj b) keep(b) ///
    coeflabels(b "\$-\theta\$") ///
    mgroups("Standard tariff" "Tariff \(\times\) goods share", pattern(1 0 0 1 0 0) ///
        prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) ///
    mtitles("OLS" "OLS" "2SLS" "OLS" "OLS" "2SLS") ///
    scalars("samp Sample" "N Obs.") sfmt(%9.0g) nonumbers obslast ///
    note("OLS columns: exporter and importer fixed effects." ///
         "2SLS: French imports; ln(w(1+t)) instrumented by ln(1+t). Robust SE.")
esttab t_all t_norow t_iv a_all a_norow a_iv, keep(b) rename(ln_tau b ln_tau_adj b ln_comb b ln_comb_adj b) ///
    se mtitles("tOLS" "tOLSx" "tIV" "aOLS" "aOLSx" "aIV")

* tidy coefficient csv
preserve
clear
set obs 6
gen year = `y'
gen tariff = ""
gen estimator = ""
gen sample = ""
gen beta = .
gen se = .
gen n = .
local i = 0
foreach mod in t_all t_norow t_iv a_all a_norow a_iv {
    local ++i
    estimates restore `mod'
    local key = cond(inlist("`mod'","t_iv","a_iv"), "ln_comb", "ln_tau")
    if inlist("`mod'","a_all","a_norow") local key "ln_tau_adj"
    if "`mod'"=="a_iv" local key "ln_comb_adj"
    replace tariff    = cond(substr("`mod'",1,1)=="a","goods_adj","standard") in `i'
    replace estimator = cond(strpos("`mod'","iv"),"2SLS","OLS") in `i'
    replace sample    = cond(strpos("`mod'","norow"),"excl_row", cond(strpos("`mod'","iv"),"fra","all")) in `i'
    replace beta = _b[`key'] in `i'
    replace se   = _se[`key'] in `i'
    replace n    = e(N) in `i'
}
export delimited using "$OUT/gravity_coef_`y'.csv", replace
restore
noi di as result "[04] saved gravity_`y'.tex and gravity_coef_`y'.csv"
