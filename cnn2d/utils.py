from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
def get_fold(train,FOLD_NUM = 5):
    train_image_num_per_patient = train.groupby('StudyInstanceUID')['SOPInstanceUID'].nunique()
    target_cols = [c for i, c in enumerate(train.columns) if i > 2]
    
    train_per_patient_char = pd.DataFrame(index=train_image_num_per_patient.index, columns=['image_per_patient'], data=train_image_num_per_patient.values.copy())
    for t in target_cols:
        train_per_patient_char[t] = train_per_patient_char.index.map(train.groupby('StudyInstanceUID')[t].mean())
        
    
    bin_counts = [40] #, 20]
    digitize_cols = ['image_per_patient'] #, 'pe_present_on_image']
    non_digitize_cols = [c for c in train_per_patient_char.columns if c not in digitize_cols]
    for i, c in enumerate(digitize_cols):
        bin_count = bin_counts[i]
        percentiles = np.percentile(train_per_patient_char[c], q=np.arange(bin_count)/bin_count*100.)
        train_per_patient_char[c+'_digitize'] = np.digitize(train_per_patient_char[c], percentiles, right=False)
        
    train_per_patient_char['key'] = train_per_patient_char[digitize_cols[0]+'_digitize'].apply(str)
    for c in digitize_cols[1:]:
        train_per_patient_char['key'] = train_per_patient_char['key']+'_'+train_per_patient_char[c+'_digitize'].apply(str)
    folds = FOLD_NUM
    kfolder = StratifiedKFold(n_splits=folds, shuffle=True, random_state=719)
    val_indices = [val_indices for _, val_indices in kfolder.split(train_per_patient_char['key'], train_per_patient_char['key'])]
    train_per_patient_char['fold'] = -1
    for i, vi in enumerate(val_indices):
        patients = train_per_patient_char.index[vi]
        train_per_patient_char.loc[patients, 'fold'] = i
    return train_per_patient_char

def split_train_val(data_config,fold,FOLD_NUM=5):
    main_df = pd.read_csv(data_config.train_csv_path)
    train_df = main_df[data_config.ids+data_config.label]
    train_per_patient_char = get_fold(main_df,FOLD_NUM)
    TID = train_per_patient_char[train_per_patient_char.fold!=fold].index
    VID = train_per_patient_char[train_per_patient_char.fold==fold].index
    t_df = train_df[train_df['StudyInstanceUID'].isin(TID)]
    v_df = train_df[train_df['StudyInstanceUID'].isin(VID)]
    return t_df,v_df

def split_train_val_lstm(data_config,fold,FOLD_NUM=5):
    main_df = pd.read_csv(data_config.train_csv_path)
    train_df = main_df[data_config.ids+data_config.label_lstm]
    train_per_patient_char = get_fold(main_df,FOLD_NUM)
    TID = train_per_patient_char[train_per_patient_char.fold!=fold].index
    VID = train_per_patient_char[train_per_patient_char.fold==fold].index
    t_df = train_df[train_df['StudyInstanceUID'].isin(TID)]
    v_df = train_df[train_df['StudyInstanceUID'].isin(VID)]
    return t_df,v_df