import glob
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
from skimage.color import gray2rgb
import functools
import torch
from tqdm.auto import tqdm
import models
from data import *
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_utils import trainer
from utils import split_train_val,split_train_val_lstm
import os
from apex import amp
from tqdm.auto import tqdm
from efficientnet_pytorch import EfficientNet
import argparse
import pretrainedmodels
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json 
  
# Opening JSON file 
with open('../settings.json') as json_file: 
    settings = json.load(json_file)
    
if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='RSNA - Pulmonary Embolism Detection: predict feature for lstm')
    my_parser.add_argument('--config',type=str)
    my_parser.add_argument('--fold',type=int)
    my_parser.add_argument('--folds',type=int)
    args = my_parser.parse_args()
    
    model_config = eval(args.config)

    t_df,v_df = split_train_val_lstm(data_config,fold=args.fold,FOLD_NUM=args.folds)

    path256 = f"{data_config.jpeg_dir}/*/*/*.jpg"
    data = glob.glob(path256)
    new_df = []
    for row in tqdm(data):
        StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID = row.split("/")[-3:]
        num,SOPInstanceUID = SOPInstanceUID.replace(".jpg","").split("_")
        new_df.append([StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID,num])
    s_df = pd.DataFrame(new_df)
    s_df.columns = list(t_df.columns[:3])+["slice"]
    t_df = t_df.merge(s_df,on=list(t_df.columns[:3]),how='left')
    v_df = v_df.merge(s_df,on=list(v_df.columns[:3]),how='left')
    t = t_df.groupby(list(t_df.columns[:2]))
    from tqdm.auto import tqdm
    mini_dfs= []
    for i,row in tqdm(t_df.groupby(list(t_df.columns[:2]))):
        mini_dfs.append(row.sort_values("slice"))
    mini_dfs_val = []
    for i,row in tqdm(v_df.groupby(list(v_df.columns[:2]))):
        mini_dfs_val.append(row.sort_values("slice"))
        
    preprocessing_fn = functools.partial(preprocess_input, **formatted_settings)
    train_dataset = CTDataset3D(mini_dfs,data_config.jpeg_dir,
                                transforms=get_validation_augmentation(),preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = CTDataset3D(mini_dfs_val,data_config.jpeg_dir,
                                transforms=get_validation_augmentation(),preprocessing=get_preprocessing(preprocessing_fn))
    
    train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=False)
    val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
    
    cnn = EfficientNet.from_pretrained(model_config.model_name,num_classes=model_config.classes).cuda()
    size = cnn._fc.in_features
    cnn = torch.nn.DataParallel(cnn)
    fun = cnn.module.extract_features 

    model_config.model_name = model_config.model_name + f"_cnn_{args.fold}"
    cnn.load_state_dict(torch.load(model_config.MODEL_PATH+"/{}_best.pth".format(model_config.model_name)))
    print(model_config.MODEL_PATH+"/{}_best.pth".format(model_config.model_name))
    f_name = "../"+settings['feature2D']
    dir_name = f"{f_name}/{model_config.model_name}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cnn.eval()
    for x,n in tqdm(val):
        with torch.no_grad():
            x = x[0]
            y = np.zeros([len(x),size+1]).astype(float)
            for row in range(0,len(x),100):
                y[row:row+100,:-1] = torch.nn.AdaptiveAvgPool2d(1)(cnn.module.extract_features(x[row:row+100].cuda().float()))[:,:,0,0].cpu().numpy()
                y[row:row+100,-1] = cnn(x[row:row+100].cuda().float()).cpu().numpy().reshape(-1)
            np.save(f"{dir_name}/{n[0]}.npy",y)
#     z = 0      
#     for x,n in tqdm(val):
#         with torch.no_grad():
#             x = x[0]
#             y = []
#             for row in range(0,len(x),100):
#                 y.append(fun(x[row:row+100].cuda().float()).cpu().numpy())
#             y = np.concatenate(y,axis=0)
#             np.save(f"{dir_name}/{n[0]}.npy",y)

        
        
    