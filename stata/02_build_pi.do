* ============================================================================
* 02_build_pi.do  --  ICIO gross flow matrix -> block import-share matrix pi.
*
* pi_{o,d} = X_{o,d} / sum_o X_{o,d}   (column d = importer; includes domestic).
* Basic prices, no tariffs. Blocks = 19 named + ROW from countries.csv; every
* non-named ICIO entity (incl ICIO's own "row") folds into ROW.
* Writes $OUT/pi_<year>.csv (20x20, node order) and $TMP/pi_long_<year>.dta.
* ============================================================================
version 16.0
local y $YEAR

* node lookup: named codes (1..19) -> block = code; row = 20
import delimited "$REPO/countries.csv", varnames(1) clear stringcols(2)
keep node_id code
tempfile nodes
save `nodes'
preserve
    keep if node_id <= 19
    keep code node_id
    rename code blkcode
    tempfile named
    save `named'
restore

use "$TMP/flows_`y'.dta", clear
replace exp_cntry = lower(exp_cntry)
reshape long imp_, i(exp_cntry) j(imp_cntry) string
rename imp_ flow
replace imp_cntry = lower(imp_cntry)

* map exporter to block
rename exp_cntry blkcode
merge m:1 blkcode using `named', keep(master match) nogen keepusing(node_id)
gen exp_blk = blkcode
replace exp_blk = "row" if missing(node_id)
drop node_id blkcode
* map importer to block
rename imp_cntry blkcode
merge m:1 blkcode using `named', keep(master match) nogen keepusing(node_id)
gen imp_blk = blkcode
replace imp_blk = "row" if missing(node_id)
drop node_id blkcode

collapse (sum) flow, by(exp_blk imp_blk)

* ensure full 20x20 grid, fill 0
rename (exp_blk imp_blk) (code_i code_j)
fillin code_i code_j
replace flow = 0 if missing(flow)

* import shares within each importer column
bysort code_j: egen col_total = total(flow)
gen pi = cond(col_total > 0, flow/col_total, 0)

* attach node ids
rename code_i code
merge m:1 code using `nodes', keep(match) nogen keepusing(node_id)
rename (code node_id) (exp_blk exp_id)
rename code_j code
merge m:1 code using `nodes', keep(match) nogen keepusing(node_id)
rename (code node_id) (imp_blk imp_id)

sort exp_id imp_id
save "$TMP/pi_long_`y'.dta", replace

* wide export in node order (rows=exporter node, cols=importer node)
preserve
    keep exp_id imp_id pi
    reshape wide pi, i(exp_id) j(imp_id)
    sort exp_id
    local pivars
    forvalues c = 1/20 {
        local pivars `pivars' pi`c'
    }
    export delimited `pivars' using "$OUT/pi_`y'.csv", novarnames replace
restore
noi di as result "[02] saved pi_`y'.csv (20x20 import-share matrix)"
