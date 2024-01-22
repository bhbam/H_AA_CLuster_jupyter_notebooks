import glob, os
import json
# local="/eos/uscms/store/group/lpcml/bbbam/Ntuples/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2/230420_051900/0000"
local="/eos/uscms//store/user/bhbam/GenInfo_Unboosted_signal_M8/Unboosted_gen_HToAATo4Tau_Hadronic_tauDR0p4_M8_ctau0To3_eta0To2p4_pythia8_2018UL_lessPerFile/Signal_M8/*/0000"
rhFileList = '%s/output*.root'%(local)
# local1="/eos/uscms//store/group/lpcml/bbbam/Ntuples_v2/aToTauTau_Hadronic_tauDR0p4_m14To17p2_pT30To180_ctau0To3_eta0To2p4_pythia8_dataset_2/aToTauTau_Hadronic_tauDR0p4_m14p8To17p2_eta0To2p4_pythia8_unbiased4ML_v2_dataset_2/230826_063114/0001"
# rhFileList1 = '%s/output*.root'%(local1)
# rhFileList = '%s/out*.root'%(local)
# rhFileList = '%s/out*.root'%(local)
rhFileList = glob.glob(rhFileList)
# rhFileList1 = glob.glob(rhFileList1)
assert len(rhFileList) > 0
total_files = len(rhFileList)
# assert len(rhFileList1) > 0
# total_files1 = len(rhFileList1)
print("Total file found :  ",total_files) #, "and  ", total_files1)
# Define the output JSON file path
List = rhFileList #+ rhFileList1

output_json_file = "Unboosted_H_AA_4Tau_M8.json"

# Write the filenames to the JSON file
with open(output_json_file, "w") as fout:
    json.dump(List, fout)
