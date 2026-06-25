* ============================================================================
* 03_build_L.do  --  World Bank total labour force -> block labour vector L.
*
* Selects the target year by its column LABEL (robust; avoids the fragile
* hardcoded v61). Excludes World Bank regional aggregates (WLD, EUU, ...) by
* keeping only real ISO3 countries (BACI country list), so ROW = sum of real
* non-named countries only. Writes $OUT/L_<year>.csv (node order).
* ============================================================================
version 16.0
local y $YEAR

* real-country filter (lower-case ISO3) from the BACI country list
import delimited "$DATA/raw/baci/country_codes_V202601.csv", varnames(1) clear
keep country_iso3
rename country_iso3 code
replace code = lower(code)
duplicates drop
tempfile realc
save `realc'

* node lookup
import delimited "$REPO/countries.csv", varnames(1) clear stringcols(2)
keep node_id code
tempfile nodes
save `nodes'
preserve
    keep if node_id <= 19
    keep code node_id
    tempfile named
    save `named'
restore

* WB labour force; find the column whose original header (label) is the year
import delimited "$WB_CSV", varnames(5) clear
local yv ""
foreach v of varlist _all {
    local lab : variable label `v'
    if "`lab'" == "`y'" local yv `v'
}
if "`yv'" == "" {
    noi di as error "[03] Could not find a column labelled `y' in the WB file."
    exit 459
}
keep countrycode `yv'
rename `yv' L
rename countrycode code
replace code = lower(code)
drop if missing(L)

* block = named code, else row (but only for real countries; drop WB aggregates)
merge m:1 code using `named', keep(master match) nogen keepusing(node_id)
gen blk = code
replace blk = "" if missing(node_id)            // not a named country
drop node_id
* tag real countries
merge m:1 code using `realc', keep(master match) gen(_isreal)
replace blk = "row" if blk == "" & _isreal == 3 // real, non-named -> ROW
drop if blk == ""                               // drop WB regional aggregates
drop _isreal code

collapse (sum) L, by(blk)
rename blk code
merge 1:1 code using `nodes', keep(match using) nogen
replace L = 0 if missing(L)
sort node_id
export delimited L using "$OUT/L_`y'.csv", novarnames replace
noi di as result "[03] saved L_`y'.csv (`=_N' nodes)"
