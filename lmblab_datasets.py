import itertools

from study import Study
from sklearn.externals.joblib import Memory
import pandas as pd

memory = Memory(cachedir='./tmp', verbose=0)

@memory.cache
def b2p_p3(reduce_memory = False, behavioural = True):
    studyname = 'b2p_study'
    ages = 'young'
    paradigm = 'Todd P300'
    sr = 512
    nelec = 74
    groups = ['controls', 'concussed']
    
    ###### Define all controls
    controls_ids = list(range(1,28))
    
    ###### Define all concussed
    concussed_ids = [
        1079,
        1088,
        1100,
        1101,
        1102,
        1110,
        1111,
        1118,
        1119,
        1125,
        1127,
        1128,
        1129,
        1130,
        1131,
        1132,
        1133,
        1138,
        1140,
        1142,
        1150,
        1151,
        1152,
        1153,
        1157,
        1166,
    ]
    
    subjects = {groups[0]: [
            'mTBI_P%02d'%(i) for i in controls_ids
            ], 
        groups[1]: [
            '%d'%(i) for i in concussed_ids
        ]}
    
    conditions = {
        'Standard': '_P3_std',
        'Frequency Deviant': '_P3_1',
        'Duration Deviant': '_P3_2',
        'Intensity Deviant': '_P3_3',
    }
    
    cont_conditions = {
        'Standard': 10,
        'Frequency Deviant': 20,
        'Duration Deviant': 30,
        'Intensity Deviant': 40,
    }
    
    
    if not behavioural:
        return Study(studyname, paradigm, ages, sr, groups, subjects, conditions, reduce_memory, load_induced = False, nelec = nelec)#, picks = ['Cz', 'Pz', 'Fz'])
    else:
        df = pd.read_csv('./Data/%s/behaviour_demographics.csv'%(studyname), delimiter=',')
        df = df.loc[df['id'].isin(subjects[groups[0]]) | df['id'].isin(subjects[groups[1]])]

        return (Study(studyname, paradigm, ages, sr, groups, subjects, conditions, reduce_memory, load_induced = False, nelec = nelec), df)
    
    
@memory.cache
def b2p_p3_long(reduce_memory = False, behavioural = True):
    studyname = 'b2p_study'
    ages = 'young'
    paradigm = 'Todd P300'
    sr = 512
    nelec = 74
    groups = ['concussed']
    
    ###### Define all concussed
    concussed_ids = [
        '1088_2',
        '1100_2',
        '1101_2',
        '1111_2',
        '1119_2',
        '1125_2',
        '1127_2nd',
        '1128_2',
        '1130_2',
        '1131_2',
        '1132_2',
        '1133_2',
        '1138_2',
        '1140_2',
        '1150_2',
        '1152_f',
        '1153_2_f',
        '1157_2_f',
        '1166_2',
    ]
    
    subjects = {
        groups[0]: [
            '%s'%(i) for i in concussed_ids
        ]}
    
    conditions = {
        'Standard': '_DC Detrend_P3_std',
        'Frequency Deviant': '_DC Detrend_P3_1',
        'Duration Deviant': '_DC Detrend_P3_2',
        'Intensity Deviant': '_DC Detrend_P3_3',
    }
    
    cont_conditions = {
        'Standard': 10,
        'Frequency Deviant': 20,
        'Duration Deviant': 30,
        'Intensity Deviant': 40,
    }
    
    
    if not behavioural:
        return Study(studyname, paradigm, ages, sr, groups, subjects, conditions, reduce_memory, load_induced = False, nelec = nelec)
    else:
        df = pd.read_csv('./Data/%s/behaviour_demographics.csv'%(studyname), delimiter=',')
        df = df.loc[df['id'].isin(subjects[groups[0]]) | df['id'].isin(subjects[groups[1]])]

        return (Study(studyname, paradigm, ages, sr, groups, subjects, conditions, reduce_memory, load_induced = False, nelec = nelec), df)
