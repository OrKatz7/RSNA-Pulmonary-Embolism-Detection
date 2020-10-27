from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import glob
import os
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

def get_train_val(data_config,FOLD,FOLDS):
    t_df,v_df = split_train_val_lstm(data_config,fold=FOLD,FOLD_NUM=FOLDS)
    path256 = f"{data_config.jpeg_dir}/*/*/*.jpg"
    data = glob.glob(path256)
    new_df = []
    for row in data:
        StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID = row.split("/")[-3:]
        num,SOPInstanceUID = SOPInstanceUID.replace(".jpg","").split("_")
        new_df.append([StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID,num])
    s_df = pd.DataFrame(new_df)
    s_df.columns = list(t_df.columns[:3])+["slice"]
    t_df = t_df.merge(s_df,on=list(t_df.columns[:3]),how='left')
    v_df = v_df.merge(s_df,on=list(v_df.columns[:3]),how='left')
    t = t_df.groupby(list(t_df.columns[:2]))
    mini_dfs= []
    s=0
    for i,row in t_df.groupby(list(t_df.columns[:2])):
        mini_dfs.append(row.sort_values("slice"))
    mini_dfs_val = []
    s=0
    for i,row in v_df.groupby(list(v_df.columns[:2])):
        mini_dfs_val.append(row.sort_values("slice"))
    return mini_dfs,mini_dfs_val