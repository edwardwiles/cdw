function setwd(server, user)
    # sets working directories
    if server == 1 # if working on the econ servers
        if user == 1
            cd(raw"/bbkinghome/mhansari/Robustness")
            global folderData = "/bbkinghome/mhansari/Robustness"
        end 
        if user == 2
            # Ed's modular working copy — run from here so the .opt files (loaded by
            # bare filename at runtime) and output CSVs resolve to this repo root.
            cd(raw"/bbkinghome/edav/gravity_robustness/trade_robustness_modular")
            global folderData = "/bbkinghome/edav/gravity_robustness/trade_robustness_modular"
        end
    elseif server == 0 # if working on laptop 
        if user == 1
            cd(raw"C:/2. MIT/Model Robustness")
            global folderData = "C:/2. MIT/Model Robustness/Data"
        end
        if user == 2
            cd(raw"C:/Users/edwar/Dropbox (MIT)/Gravity robustness/Analysis/Julia Consolidated")
            global folderData = "C:/Users/edwar/Dropbox (MIT)/Gravity robustness/Analysis/WIOD Data"
        end        
    end
end