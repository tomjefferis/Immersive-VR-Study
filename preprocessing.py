import os
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

filenames = []
for file in os.listdir("../EEG/"):
    if file.endswith(".bdf"):
        filepath = "../EEG/" + file
        raw = mne.io.read_raw_bdf(filepath)
        raw.load_data()
        raw.drop_channels(
            ch_names=["LEOG", "REOG", "A1", "A2", "EXG7", "EXG8", 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet',
                      'Temp', 'Status', 'Po4', 'UEOG', 'DEOG'])
        raw = raw.notch_filter(np.arange(50, 251, 50), fir_design='firwin')
        raw = raw.filter(1, 60.)
        raw.set_montage(mne.channels.make_standard_montage('biosemi32'))
        ica = ICA(n_components=31, max_iter=50000, random_state=69,method='infomax')
        ica.fit(raw)
        dicts = label_components(raw, ica, method='iclabel')
        prob = dicts['y_pred_proba']
        labels = dicts['labels']
        # remove comonents labled eye or muscle with prob over 0.8

        f = []
        for idx in range(0, len(prob)):
            if prob[idx] >= 0.8 and not labels[idx] == 'brain':
                f.append(idx)
        ica.exclude = f
        raw = ica.apply(raw)
        raw = raw.set_eeg_reference(ref_channels='average')
        raw = raw.resample(sfreq=250)
        raw.save('../EEG/' + file[:-4] + '.fif', overwrite=True)
        x = raw.get_data()
        np.savetxt('../EEG/' + file[:-4] + '.csv', x, delimiter=',')


