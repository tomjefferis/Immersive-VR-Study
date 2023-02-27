import os

import mne_icalabel
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.viz import plot_ica_components
from mne_icalabel import label_components
from pyprep import PrepPipeline
import multiprocessing



def preprocessing(filepath):
    raw = mne.io.read_raw_bdf(filepath)
    raw.load_data()
    montage = mne.channels.make_standard_montage('biosemi32')
    # if first channel A1 then rename
    if raw.ch_names[0] == 'A1':
        old_chans = raw.ch_names
        new_chans = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'Po4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status']
        raw.rename_channels(dict(zip(old_chans, new_chans)))
        raw.drop_channels(
            ch_names=['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet',
                      'Temp', 'Status'])
    elif "A1" in raw.ch_names:
        raw.drop_channels(
            ch_names=["LEOG", "REOG", "A1", "A2", "EXG7", "EXG8", 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet',
                      'Temp', 'Status', 'UEOG', 'DEOG'])
    else:
        raw.drop_channels(
            ch_names=['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet',
                      'Temp', 'Status'])
    raw.rename_channels({'Po4': 'PO4'})
    raw.set_montage(montage)
    raw = raw.resample(250)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, 250 / 2, 50),
    }
    prep = PrepPipeline(raw, prep_params, montage)
    raw = prep.fit().raw
    #raw = raw.notch_filter(np.arange(50, 101, 50), fir_design='firwin')
    raw = raw.filter(1, 60.)
    ica = ICA(n_components=20, max_iter=50000, random_state=69, method='infomax')
    ica.fit(raw)
    #ica.plot_components()
    dicts = label_components(raw, ica, method='iclabel')
    #topo = plot_ica_components(ica)
    prob = dicts['y_pred_proba']
    labels = dicts['labels']
    # remove comonents labled eye or muscle with prob over 0.8

    f = []
    for idx in range(0, len(prob)):
        if prob[idx] >= 0.8 and not labels[idx] == 'brain':
            f.append(idx)
    ica.exclude = f
    raw = ica.apply(raw)
    raw.save('../EEG/' + filepath[6:-4] + '.fif', overwrite=True)
    #raw.save('../EEG/' + file[:-4] + '.fif', overwrite=True)

#if __name__ == '__main__':
#    pool = multiprocessing.Pool()
filenames = []
for file in os.listdir("../EEG/"):
    if file.endswith(".bdf"):
        filepath = "../EEG/" + file
        filenames.append(filepath)
            #x = raw.get_data()
            #np.savetxt('../EEG/' + file[:-4] + '.csv', x, delimiter=',')
filenames = filenames[16:28]
    #result = pool.map(preprocessing, filenames)
for file in filenames:
    preprocessing(file)

