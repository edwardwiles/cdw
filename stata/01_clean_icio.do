* ============================================================================
* 01_clean_icio.do  --  raw OECD ICIO -> country x country gross flow matrix
*                       (basic prices) + bilateral goods/services shares.
*
* Based on Noah Siderhurst's clean_icio.do, with two portability fixes
* (file-discovery word-count guard; single-backslash year regex) and an added
* goods/services-share computation. Reads $ICIO_DIR, writes to $TMP:
*    flows_<year>.dta            exp_cntry x imp_<cntry> gross flows (incl domestic diagonal)
*    services_share_<year>.dta   bilateral goods/services flows and shares
* ============================================================================
version 16.0

local files : dir "$ICIO_DIR" files "*.csv"
* `: dir ... files' returns quoted names; use word count, not `if "`files'"==""`.
if `: word count `files'' == 0 {
    noi di as error "No CSV files found in $ICIO_DIR"
    exit 601
}

clear
set obs 0
gen filename = ""
foreach f of local files {
    set obs `=_N+1'
    replace filename = "`f'" in L
}
gen ffull = "$ICIO_DIR/" + filename
gen filename_l = lower(filename)
* single backslash before the dot ("\\." matches a literal backslash on this regex engine)
gen year = real(regexs(1)) if regexm(filename_l, "([0-9][0-9][0-9][0-9])\.csv$")
count if missing(year)
if r(N) > 0 {
    noi di as error "Could not parse year from some ICIO files:"
    list filename if missing(year), noobs
    exit 459
}
isid year
sort year
levelsof year, local(years)

foreach y of local years {
    levelsof ffull if year == `y', local(file)
    local f : word 1 of `file'

    import delim using "`f'", varnames(1) clear

    * organizing country and industry codes
    gen exp_ind   = substr(v1, 5, .)
    gen exp_cntry = substr(v1, 1, 3)
    order exp_cntry exp_ind
    replace exp_cntry = lower(exp_cntry)
    assert out == 0 if inlist(exp_cntry, "chn", "mex")
    replace exp_cntry = "mex" if inlist(exp_cntry, "mx1", "mx2")
    replace exp_cntry = "chn" if inlist(exp_cntry, "cn1", "cn2")   // mex and chn split into two

    ds mx1*
    foreach var in `r(varlist)' {
        local ind = substr("`var'", 4, .)
        assert mex`ind' == 0
        drop mex`ind'
        rename `var' mex_`ind'
    }
    ds cn1*
    foreach var in `r(varlist)' {
        local ind = substr("`var'", 4, .)
        assert chn`ind' == 0
        drop chn`ind'
        rename `var' chn`ind'
    }
    ds mx2*
    foreach var in `r(varlist)' {
        local ind = substr("`var'", 4, .)
        rename `var' mex_`ind'_2
    }
    ds cn2*
    foreach var in `r(varlist)' {
        local ind = substr("`var'", 4, .)
        rename `var' chn`ind'_2
    }

    drop out v1
    drop if inlist(exp_cntry, "tls", "va", "out")

    * --- bilateral goods/services shares (goods = ISIC A,B,C merchandise, by exporting industry) ---
    preserve
        gen byte goods = inlist(upper(substr(exp_ind, 1, 1)), "A", "B", "C")
        ds exp_cntry exp_ind goods, not
        collapse (sum) `r(varlist)', by(exp_cntry goods)
        levelsof exp_cntry, local(scntrys)
        foreach cntry in `scntrys' {
            egen imp_`cntry' = rowtotal(`cntry'*)
            drop `cntry'*
        }
        reshape long imp_, i(exp_cntry goods) j(imp_cntry) string
        rename imp_ flow
        reshape wide flow, i(exp_cntry imp_cntry) j(goods)
        foreach v in flow0 flow1 {
            capture confirm variable `v'
            if _rc gen double `v' = 0
            replace `v' = 0 if missing(`v')
        }
        gen double services_flow  = flow0
        gen double goods_flow      = flow1
        gen double total_flow      = flow0 + flow1
        gen double goods_share     = cond(total_flow > 0, goods_flow/total_flow, .)
        gen double services_share  = cond(total_flow > 0, services_flow/total_flow, .)
        keep  exp_cntry imp_cntry goods_flow services_flow total_flow goods_share services_share
        order exp_cntry imp_cntry goods_flow services_flow total_flow goods_share services_share
        sort  exp_cntry imp_cntry
        save "$TMP/services_share_`y'", replace
    restore

    * --- gross flow matrix: collapse over exporting industry, then sum importer industries ---
    ds exp_cntry exp_ind, not
    collapse (sum) `r(varlist)', by(exp_cntry)
    levelsof exp_cntry, local(cntrys)
    foreach cntry in `cntrys' {
        egen imp_`cntry' = rowtotal(`cntry'*)
        drop `cntry'*
    }
    save "$TMP/flows_`y'", replace
    noi di as result "[01] year `y': saved flows_`y'.dta and services_share_`y'.dta"
}
