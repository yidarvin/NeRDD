#This program parses all the .h5 files in the CNN input file directory, extracts ground
#truth and predictions and calculate perImage and perPatient accuracy
#Author: Archana Shenoy

import h5py
import numpy as np
import os

#rootdir = "/Users/archanashenoy/Desktop/h5images/"
rootdir = "/media/dnr/Documents/data/NeRDD/AS_validation_dataCopy/"

truthPreds_listOfLists = []
for folder in os.listdir(rootdir):
    new_dir = rootdir + str(folder) + "/"
    for root, dirs, files in os.walk(new_dir):
        for file_name in files:
            f = h5py.File(new_dir + file_name, 'r')
            truth = np.array(f[f.keys()[3]])
            predProbs = np.array(f[f.keys()[4]])
            preds = predProbs.argmax() + 1
            single_truthPred = (truth.tolist(), preds)
            truthPreds_listOfLists.append(single_truthPred)

print truthPreds_listOfLists

numCorrect = 0
for image in truthPreds_listOfLists:
    if image[0] == image[1]:
        numCorrect += 1


#perImageAccuracy = float(numCorrect)/len(truthPreds_listOfLists)
#print(perImageAccuracy)

