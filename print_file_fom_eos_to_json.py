import glob, os
import json
import argparse
parser = argparse.ArgumentParser(description="Collect and save sorted list of .root files in a directory to a JSON file.")
parser.add_argument("--local", type=str,  default="/store/group/lpcml/bbbam/Ntuples_v2/signal/gen_HToAATo4Tau_Hadronic_tauDR0p4_M14_ctau0To3_eta0To2p4_pythia8_2018UL/HtoAATo4Tau_Hadronic_tauDR0p4_M14_eta0To2p4_pythia8_signal_v2/231013_053602/0000", help="Directory containing the .root files.")
parser.add_argument("--output_json_file", type=str, default="signal_sample_M14.json", help="Name of the output JSON file.")
args = parser.parse_args()

local= args.local
output_json_file = args.output_json_file
rhFileList = '/eos/uscms%s/output*.root'%(local)
rhFileList = sorted(glob.glob(rhFileList))
assert len(rhFileList) > 0
total_files = len(rhFileList)
print("Total file found :  ",total_files) #, "and  ", total_files1)
# Write the filenames to the JSON file
rhFileList = [path.replace("/eos/uscms", "root://cmseos.fnal.gov/") for path in rhFileList]
with open(output_json_file, "w") as fout:
    json.dump(rhFileList, fout)
print(f"output_json_file created: {output_json_file}")
