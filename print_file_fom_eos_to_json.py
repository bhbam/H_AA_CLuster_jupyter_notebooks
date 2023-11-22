import glob, os
import json
# local="/eos/uscms/store/group/lpcml/bbbam/Ntuples/aToTauTau_Hadronic_tauDR0p4_m3p6To16_pT30To180_ctau0To3_eta0To1p4_pythia8_unbiased4ML_dataset_1/aToTauTau_Hadronic_tauDR0p4_m3p6To14_dataset_1_reunbaised_mannual_v2/230420_051900/0000"
local="/eos/uscms/store/user/bbbam/unphysical_sample_test/aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_pT30To180_ctau0To3_eta0To2p4_unphysical/crab_aToTauTau_Hadronic_tauDR0p4_m1p2To3p6_pythia8_GEN_unphysical/230929_033832/0000"
# rhFileList = '%s/output*.root'%(local)
rhFileList = '%s/aToTauTau_GEN_SIM_unphysical*.root'%(local)
rhFileList = glob.glob(rhFileList)
assert len(rhFileList) > 0
total_files = len(rhFileList)
print("Total file found :  ",total_files)
# Define the output JSON file path
output_json_file = "aToTauTau_massreg_unphysical_test.json"

# Write the filenames to the JSON file
with open(output_json_file, "w") as fout:
    json.dump(rhFileList, fout)
