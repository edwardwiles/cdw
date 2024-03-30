function setwd(server, user)
    # sets working directories
    if server == 1 # if working on the econ servers
        if user == 1
            cd(raw"/bbkinghome/mhansari/Robustness")
            global folderData = "/bbkinghome/mhansari/Robustness"
        end 
        if user == 2
            cd(raw"/bbkinghome/edav/gravity")
            global folderData = "/bbkinghome/edav/gravity"
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