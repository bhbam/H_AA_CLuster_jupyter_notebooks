#!/bin/bash

#source run_print_file_fom_eos_to_json_py_over_all_signal_background.sh to run this

# Exit immediately if a command exits with a non-zero status
set -e

# Pass through the python executable from the user's active environment
PYTHON_BIN=$(type -p python || command -v python || echo "python3")

# Base EOS directory path
BASE_DIR="/store/group/lpcml/bbbam/Ntuples_Signal_Background_inference_root_to_root_miniAOD_June15_2026"

# List of all signal and background sample folder names
SAMPLES=(
    "Signal_M3p7_GeV_with_mass_classifier_infernce_miniAOD_June15"
    "Signal_M4_GeV_with_mass_classifier_infernce_miniAOD_June15"
    "Signal_M5_GeV_with_mass_classifier_infernce_miniAOD_June15"
    "Signal_M6_GeV_with_mass_classifier_infernce_miniAOD_June15"
    "Signal_M8_GeV_with_mass_classifier_infernce_miniAOD_June15"
    "Background_Wjets_with_mass_classifier_infernce_miniAOD_June15"
    "Background_DYto2L_with_mass_classifier_infernce_miniAOD_June15"
    "Background_HTo2Tau_with_mass_classifier_infernce_miniAOD_June15"
    "Background_TTBar_with_mass_classifier_infernce_miniAOD_June15"
    "Background_QCD_with_mass_classifier_infernce_miniAOD_June15"
)

echo "Starting execution over all datasets..."

# Loop over each sample and run the python command
for sample in "${SAMPLES[@]}"; do
    input_path="${BASE_DIR}/${sample}"
    output_json="${sample}.json"

    echo "--------------------------------------------------"
    echo "Processing: ${sample}"
    echo "--------------------------------------------------"

    python print_file_fom_eos_to_json.py \
        --local "${input_path}" \
        --output_json "${output_json}"
done

echo "--------------------------------------------------"
echo "All tasks finished successfully!"
