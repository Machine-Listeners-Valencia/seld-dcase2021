import sys
import os
import numpy as np
from shutil import copyfile


def concatenate_representation(path2foa, path2mic):
    if os.path.isdir('seld_feat_label_both') is False:
        os.mkdir('seld_feat_label_both')
    if os.path.isdir('seld_feat_label_both/both_dev') is False:
        os.mkdir('seld_feat_label_both/both_dev')
    if os.path.isdir('seld_feat_label_both/both_dev_label') is False:
        os.mkdir('seld_feat_label_both/both_dev_label')
    if os.path.isdir('seld_feat_label_both/both_dev_norm') is False:
        os.mkdir('seld_feat_label_both/both_dev_norm')

    foa_files = [f for f in os.listdir(os.path.join(path2foa, 'foa_dev')) if f.endswith('.npy')]
    mic_files = [f for f in os.listdir(os.path.join(path2mic, 'mic_dev')) if f.endswith('.npy')]

    for i in range(0, len(foa_files)):
        foa_aux = np.load(os.path.join(path2foa, 'foa_dev', foa_files[i]))
        mic_aux = np.load(os.path.join(path2mic, 'mic_dev', mic_files[i]))

        combined_aux = np.concatenate((foa_aux, mic_aux), axis=1)

        np.save(os.path.join('seld_feat_label_both/both_dev', foa_files[i]), combined_aux)

    foa_files = [f for f in os.listdir(os.path.join(path2foa, 'foa_dev_norm')) if f.endswith('.npy')]
    mic_files = [f for f in os.listdir(os.path.join(path2mic, 'mic_dev_norm')) if f.endswith('.npy')]

    for i in range(0, len(foa_files)):
        foa_aux = np.load(os.path.join(path2foa, 'foa_dev_norm', foa_files[i]))
        mic_aux = np.load(os.path.join(path2mic, 'mic_dev_norm', mic_files[i]))

        combined_aux = np.concatenate((foa_aux, mic_aux), axis=1)

        np.save(os.path.join('seld_feat_label_both/both_dev_norm', foa_files[i]), combined_aux)

    labels = [f for f in os.listdir(os.path.join(path2foa, 'foa_dev_label')) if f.endswith('.npy')]

    for i in range(0, len(labels)):
        copyfile(os.path.join(path2foa, 'foa_dev_label', labels[i]),
                 os.path.join('seld_feat_label_both', 'both_dev_label', labels[i]))


if __name__ == '__main__':

    #if len(sys.argv) == 3:
    #    concatenate_representation(sys.argv[1], sys.argv[2])
    if True:
        concatenate_representation('/home/javi/repos/seld-dcase2021/seld_feat_label_foa',
                                   '/home/javi/repos/seld-dcase2021/seld_feat_label_mic')
    else:
        print('Insert path to foa and mic dataset')
