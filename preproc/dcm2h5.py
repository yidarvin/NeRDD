import dicom
import h5py
import numpy as np
from os import listdir,mkdir
from os.path import isdir,join

# Filepaths
path_pats = '/home/NAS-DNR'
path_save = '/home/dnr/Documents/data/NeRDD/h5'
path_tsvs = '/home/dnr/Documents/data/NeRDD/tsv'
# Corollaries
name_tsv0 = 'MLungren_rad_summ1.tsv'
name_tsv1 = 'MLungren_rad_summ4.tsv'
name_tsv2 = 'MLungren_rad_summ9.tsv'
list_pats = listdir(path_pats)
if not isdir(path_save):
    mkdir(path_save)

# Read in tsvs for acc2lab dictionary.
acc2lab = {}
counter = 0
for iter_tsv,name_tsv in enumerate([name_tsv0, name_tsv1, name_tsv2]):
    path_tsv = join(path_tsvs, name_tsv)
    tsv = open(path_tsv, 'r')
    line = tsv.readline()
    line = tsv.readline()
    while line:
        line = line.split('\t')
        try:
            acc2lab[line[3]] = iter_tsv
        except:
            counter += 1
            print 'Failed ' + str(counter) + ' times.'
        line = tsv.readline()
    tsv.close()

# Loop through all the images and save them in an h5 file.
for iter_pat,name_pat in enumerate(list_pats):
    if iter_pat > 1000:
        break
    path_pat = join(path_pats, name_pat)
    list_dcms = listdir(path_pat)
    # Loop through all the dicoms.
    for iter_dcm,name_dcm in enumerate(list_dcms):
        if name_dcm[-4:] != '.dcm':
            continue
        # Reading the image.
        name_img = name_dcm[:-4]
        path_dcm = join(path_pat, name_dcm)
        dcm = dicom.read_file(path_dcm)
        acc = dcm.AccessionNumber
        if acc not in acc2lab:
            continue
        img = dcm.pixel_array
        lab = acc2lab[acc]
        # Let's normalize the image.
        img = img.astype(np.float32)
        img -= np.mean(img)
        img /= np.std(img)
        img = np.clip(img, -3, 3)
        img += 3
        img /= 6
        if dcm.PhotometricInterpretation[-1] == '1':
            img = 1 - img
        # Saving the image.
        path_acc = join(path_save, acc)
        if not isdir(path_acc):
            mkdir(path_acc)
        path_h5 = join(path_acc, name_img+'.h5')
        h5f = h5py.File(path_h5, 'w')
        h5f.create_dataset('data', data=img)
        h5f.create_dataset('label', data=lab)
        h5f.create_dataset('name', data=name_dcm)
        h5f.create_dataset('pat', data=name_pat)
        h5f.create_dataset('acc', data=acc)
        h5f.close()
