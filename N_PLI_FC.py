import mne
import mne_connectivity
from mne.datasets import sample

# Load sample data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # bads + 2 more
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       stim=False, exclude='bads')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax,
                    picks=picks,
                    baseline=(None, 0),
                    reject=dict(mag=4e-12),
                    preload=True)

# Compute PLI connectivity matrix for alpha band (8-13 Hz)
fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq'] 
tmin_pli = -0.1 
tmax_pli = -0.05  
epochs_alpha = epochs.copy().filter(fmin=fmin,fmax=fmax,n_jobs=1,
                                    l_trans_bandwidth=1,h_trans_bandwidth=1)
con_alpha , freqs_alpha , times_alpha , n_epochs , n_tapers \
    = spectral_connectivity(epochs_alpha,sfreq=sfreq,
                            method='pli',mode='multitaper',
                            tmin=tmin_pli,tmax=tmax_pli,faverage=True)

# Plot connectivity circle
node_names=['MEG0113','MEG0122','MEG0132','MEG0213','MEG0222','MEG0232',
            'MEG0313','MEG0322','MEG0333','MEG0413','MEG0422','MEG0432',
            'MEG0513','MEG0522','MEG0532','MEG0613','MEG0622',
            'EEGOz']
mne.viz.plot_connectivity_circle(con_alpha,node_names,node_colors=node_colors,
                                 title='PLI Connectivity')