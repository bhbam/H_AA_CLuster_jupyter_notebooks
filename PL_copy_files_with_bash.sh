#!/bin/bash
for j in {1..100}
	do
	for i in {1..255}
		do
 		cp /pscratch/sd/b/bbbam/datasetTau_13channels/Train/Background/WJets_46_batchNum22_$i.png /pscratch/sd/b/bbbam/png_classification/back_WJets/train/WJets_46_batchNum22_$i.png
   		echo "copied $i file"
		done
	done
