from ipdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import io, combine_evoked
from mne.baseline import rescale

electrodes = ['Fp1',
     'AF7',
     'AF3',
     'F1',
     'F3',
     'F5',
     'F7',
     'FT7',
     'FC5',
     'FC3',
     'FC1',
     'C1',
     'C3',
     'C5',
     'T7',
     'TP7',
     'CP5',
     'CP3',
     'CP1',
     'P1',
     'P3',
     'P5',
     'P7',
     'P9',
     'PO7',
     'PO3',
     'O1',
     'Iz',
     'Oz',
     'POz',
     'Pz',
     'CPz',
     'Fpz',
     'Fp2',
     'AF8',
     'AF4',
     'AFz',
     'Fz',
     'F2',
     'F4',
     'F6',
     'F8',
     'FT8',
     'FC6',
     'FC4',
     'FC2',
     'FCz',
     'Cz',
     'C2',
     'C4',
     'C6',
     'T8',
     'TP8',
     'CP6',
     'CP4',
     'CP2',
     'P2',
     'P4',
     'P6',
     'P8',
     'P10',
     'PO8',
     'PO4',
     'O2']

iter_freqs = [
    ('Delta', 1, 3),
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
]


def openDataFile(fn, tmin=0.2, tmax=1.0, samples=600, condition = '1', electrodes = electrodes, nelec = 70, reduce_memory = False, picks = None):
    data = io.read_raw_brainvision(fn + str('.vhdr'), verbose='ERROR')
    #events = mne.find_events(data, shortest_event=1, verbose=1)
        
    data_mod = data.get_data().reshape((nelec,-1,samples)).transpose(1,0,2)
    segmented = mne.EpochsArray(data_mod, data.info, tmin=-0.2, event_id={condition:1}, verbose='ERROR').pick_types(eeg=True).resample(512)
    
    if picks is not None:
        dropped = [c for c in electrodes if c not in picks]
        segmented = segmented.drop_channels(dropped)
    
    if(reduce_memory):
        return segmented.decimate(2)
    else:
        return segmented

class Study:
    _data_dir = './Data/'
    
    def __init__(self, name, paradigm, ages, sr, groups, subjects, conditions, reduce_memory=False, load_induced = False, continuous_conditions = None, continuous_name = None, nelec=70, picks=None):
        self._name = name
        self._paradigm = paradigm
        self._data_dir += self._name + '/'
        self._ages = ages
        self._sr = sr
        self._groups = groups
        self._subjects = subjects
        self._conditions = conditions
        self._nelec = nelec
        self.data = {}
        self._reduce_memory = reduce_memory
        self.single_trial_data = None
        self.subject_averages = None
        self.grand_averages = None
        self.picks = picks
        for g in self._groups:
            self.data[g] = []
        self._readData()
        if load_induced:
            self._continuous_conditions = continuous_conditions
            self._read_continuous_data(continuous_name)
        assert len(groups) == len(subjects)
       
    def __getitem__(self, key):
        return self.data[key]
        
    def compute_single_trials(self):
        self.single_trial_data = {}
        for g in self._groups:
            
            # Create empty attribute structure for single trials
            self.single_trial_data[g] = {name:list() for name in self._conditions}
            
            # Add all subjects in each group
            for s in self.data[g]:
                [self.single_trial_data[g][name].append(s.data[name]) for name in self._conditions]
                
    def visual_inspection(self, group, condition = ['Standard', 'Frequency Deviant', 'Duration Deviant', 'Intensity Deviant'], electrode = 47, subject = None, figsizex = 15, figsizey = 15):
        figs = []
        electrodes = [37 , 47, 30]
        for s in self.data[group]:
            print('Subject: %s'%(s._subid))
            fig, axes = plt.subplots(3, 3, sharey = True, figsize=(figsizex,figsizey))
            for i in range(3):
                self.plot_grand_averages('controls', condition = condition, electrode = electrodes[i], axis = axes[i][0], show=False)
                self.plot_grand_averages('concussed', condition = condition, electrode = electrodes[i], axis = axes[i][2], show=False) 
                s.plot(electrodes[i], condition = condition, axis = axes[i][1], show=False)
            figs.append(fig)
        return figs
                
    def compute_subject_averages(self):
        if self.single_trial_data is None:
            self.compute_single_trials()
        
        self.subject_averages = {}
        for g in self._groups:
        
            self.subject_averages[g] = {name:list() for name in self._conditions}
            
            for name in self._conditions:
                self.subject_averages[g][name] = list(map(lambda x: x.average(), self.single_trial_data[g][name]))
        
    def plot_grand_averages(self, group, condition = ['Standard', 'Frequency Deviant', 'Duration Deviant', 'Intensity Deviant'], electrode = 47, axis=None, show=False):
        if self.single_trial_data is None:
            self.compute_single_trials()
        if self.subject_averages is None:
            self.compute_subject_averages()
        
        return mne.viz.plot_compare_evokeds([self.subject_averages[group][i] for i in condition], picks=electrode, invert_y = True, axes = axis, show=show, show_legend = False) 
    
    def _readData(self):
        for g in self._groups:
            for s in self._subjects[g]:
                subject_data = {}
                for c in self._conditions:
                    fn = self._data_dir + s + self._conditions[c]
                    subject_data[c] = openDataFile(fn, reduce_memory = self._reduce_memory, samples = int(self._sr * 1.2), nelec = self._nelec, picks= self.picks, condition = c)
                self.data[g].append(Subject(s, g, subject_data))
                print('Completed importing subject %s'%(s))
        self._sr = 512
                
    def _read_continuous_data(self, continuous_name):
        self.subject_induced_averages = {}

        for g in self._groups:
            
            self.subject_induced_averages[g] = []
            
            for s in self._subjects[g]:
                subject_data = {name:list() for name in self._continuous_conditions}
                
                for c in self._continuous_conditions:
                    fn = self._data_dir + s + continuous_name

                    frequency_map = {}

                    for band, fmin, fmax in iter_freqs:
                        # (re)load the data to save memory
                        raw = io.read_raw_brainvision(fn + str('.vhdr'), verbose='ERROR', preload=True)
                        events = mne.find_events(raw, shortest_event = 1, verbose = 'ERROR')

                        # bandpass filter and compute Hilbert
                        raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
                                l_trans_bandwidth=1,  # make sure filter params are the same
                                h_trans_bandwidth=1,  # in each band and skip "auto" option.
                                fir_design='firwin')
                        raw.apply_hilbert(n_jobs=1, envelope=False)

                        epochs = mne.Epochs(raw, events, self._continuous_conditions[c], -0.2, 1.0, baseline=None,
                                                preload=True)
                        # remove evoked response and get analytic signal (envelope)
                        epochs.subtract_evoked() 
                        epochs = mne.EpochsArray(data=np.abs(epochs.get_data() * 1e6), info=epochs.info, tmin=epochs.tmin)
                        # now average and move on 
                        average = epochs.average()
                        gfp = np.std(average.data, axis = 0)
                        
                        gfp = mne.baseline.rescale(gfp, average.times, baseline=(None, 0))
                        
                        frequency_map[band] = gfp
                    subject_data[c] = frequency_map
                self.subject_induced_averages[g].append(subject_data)
                print('Completed induced response for subject %s'%(s))
                
    def __str__(self):
        return (""
        "Study Name: %s\n"
        "Paradigm: %s\n"
        "Age Group: %s\n"
        "Sampling Rate: %d\n"
        "Groups: %s (%d subjects), %s (%d subjects)\n"
        "Conditions: %s"%(self._name, self._paradigm, self._ages, self._sr, self._groups[0], 
             len(self._subjects[self._groups[0]]), self._groups[1], len(self._subjects[self._groups[1]]),
            self._conditions.keys()))
                     
class Subject:
    def __init__(self, subid, group, data):
        self._subid = subid
        self._group = group
        self.data = data
        
    def __str__(self):
        return (""
        "Subject Id: %s\n"
        "Group: %s\n"
        "Conditions: %s\n"
        "Data information: %s"%(self._subid, self._group, self.data.keys(), str(self.data)))
        
    def __repr__(self):
        return ('%s(%s,%s)'%(self.__class__.__name__, self._subid, self._group))
        
    def plot(self, electrode, condition=['Standard', 'Frequency Deviant', 'Duration Deviant', 'Intensity Deviant'], axis=None, show=False):
        single_subject = {name:list() for name in condition}
        #single_subject = {name:list() for name in ['Standard', 'Deviant']}

        for n in list(single_subject):
            single_subject[n] = [self.data[n].average()]
  
        return mne.viz.plot_compare_evokeds(single_subject, picks = electrode, invert_y=True, axes = axis, show=show)
