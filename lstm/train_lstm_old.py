#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json 
  
with open('../settings.json') as json_file: 
    settings = json.load(json_file)
    
import glob
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
from skimage.color import gray2rgb
import functools
import torch
from tqdm.auto import tqdm
from config import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_train_val
import os
from apex import amp
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm.auto import tqdm
from apex import amp
import pretrainedmodels
from torch import nn
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
from losses import *


# In[ ]:


# set_seed(7777)


# In[ ]:


FOLD = 0
FOLDS=10
conf = 'lstm_pe_old'


# In[ ]:


model_config = eval(conf)
mini_dfs,mini_dfs_val = get_train_val(data_config,FOLD,FOLDS)


# In[ ]:


model_config.model_name


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import os
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
import functools
import torch
import pydicom
import vtk
from vtk.util import numpy_support


def Zcrop(img,label):
    z = np.random.randint(30)/100.0
    z1 = np.random.randint(30)/100.0
    s = img.shape[0]
    start = int(s*z)
    end = int((1-z1)*s)
    return img[start:end],label[start:end]

def spacing(img,label):
    z = np.random.randint(1,4)
    image = []
    labels = []
    for row in range(len(img)):
        if row%z==0:
            image.append(img[row]) 
            labels.append(label[row]) 
    return np.array(image),np.array(labels)

def randomflip(img,label):
    return np.flip(img,0),np.flip(label,0)

def randomcrop(img,label):
    z = np.random.randint(len(img)//2,len(img)-1)
    start = np.random.randint(len(img)-z-1)
    end = start+z
    return img[start:end],label[start:end]

def augment(img,label):
    if np.random.randint(3)==0:
        img,label = Zcrop(img,label)
    elif np.random.randint(2)==0:
        img,label = randomcrop(img,label)
#     if np.random.randint(3)==0:
#         img,label = randomflip(img,label)
    return img,label

    
class CTDatasetLstm(Dataset):
    def __init__(self,df,fet_dirs,transforms = None,preprocessing=None):
        self.df_main = df
        self.fet_dirs = fet_dirs
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        mini = self.df_main[idx].values
        fn = glob.glob(f"{data_config.dicom_file}/{mini[0,0]}/*/*.dcm")[0]
        dc = pydicom.dcmread(fn, stop_before_pixels=True )
        RES = {}
        for i in list(dc.keys()):
            RES[dc[i].description()] = dc[i].value
        meta = np.zeros([9])
        meta[0:6] = np.array(RES['Image Orientation (Patient)'])
        meta[6:8] = np.array(RES['Pixel Spacing'])/10
        meta[8] = np.array(RES['Slice Thickness'])/10
        paths = [glob.glob(f"{fdir}/{mini[0,0]}.npy")[0] for fdir in self.fet_dirs]
        x_scans = [np.load(p) for p in paths]
        try:
            x = np.concatenate(x_scans,axis=1)
        except:
            print(mini[0,0])
            return self.__getitem__(np.random.randint(self.__len__()))
#         global_rv_lv = np.load(f"../cnn3d/features_rv_lv/{mini[0,0]}_3dcnn.npy")
#         global_rlc = np.load(f"../cnn3d/features_densenet121_rlc/{mini[0,0]}_3dcnn.npy")
#         global_pe = np.load(f"../cnn3d/features_densenet121_pe/{mini[0,0]}_3dcnn.npy")
        global_rv_lv = np.load(f"../{settings['features_rv_lv']}/{mini[0,0]}_3dcnn.npy")
        global_rlc = np.load(f"../{settings['features_densenet121_rlc']}/{mini[0,0]}_3dcnn.npy")
        global_pe = np.load(f"../{settings['features_densenet121_pe']}/{mini[0,0]}_3dcnn.npy")
        label = mini[:,3:-1].astype(int)
        p = self.df_main[idx].groupby('StudyInstanceUID').max().values[0][2:-1]
        label0 = np.array([p[0],p[1],p[9]])
        label1 = np.array([p[2],p[3],abs(1-p[0])])
        label2 = np.array([p[5],p[7],abs(1-p[0]),abs(1-p[5]-p[7]-abs(1-p[0]))])
        label3 = np.array([p[6],p[4],p[8]])
        if self.transforms:
            x,label = self.transforms(x,label)
        return global_rv_lv,global_rlc,global_pe,np.array(x),torch.from_numpy(label[:,0]),torch.from_numpy(np.array([label0,label1,label3])),torch.from_numpy(label2)

    def __len__(self):
        return len(self.df_main)


# In[ ]:


train_dataset = CTDatasetLstm(mini_dfs,model_config.dirs,transforms=augment,preprocessing=None)#augment
val_dataset = CTDatasetLstm(mini_dfs_val,model_config.dirs,transforms=None,preprocessing=None)
train = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10, pin_memory=False)
val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=False)


# In[ ]:


g_x,g_y,g_z,x,y,y1,y3 = val_dataset[2]


# In[ ]:


from torch import nn
from torch.nn import functional as F
import torch

sigmoid = nn.Sigmoid()



class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
    
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda().float()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class NeuralNet(nn.Module):
    def __init__(self, embed_size=5379, LSTM_UNITS=512, DO = 0.3,g=0.05):
        super(NeuralNet, self).__init__()
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True,dropout=0.0)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True,dropout=0.0)
        self.Noise = GaussianNoise(g)
        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.avd3d = nn.AdaptiveAvgPool3d(1)
        self.avd1d = nn.AdaptiveAvgPool1d(1)
        
        self.linear_rv = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear_rlc = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        
        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
                            
        self.linear_global_fc0 = nn.Linear(LSTM_UNITS*2*4, 3) #ALL_PE,NEG,IND
        self.linear_global_fc1 = nn.Linear(LSTM_UNITS*2*2, 3) #RV>1,RV<1,Not PE
        self.linear_global_fc2 = nn.Linear(LSTM_UNITS*2*4, 4) #Chronic, Chronic+Acute, Not PE,Acute
        self.linear_global_fc3 = nn.Linear(LSTM_UNITS*2*2, 3) #right,left,center
        self.dropuot3d = nn.Dropout3d(DO)
        self.dropuot1d = nn.Dropout(DO)
        self.s3d = Swish_Module()
        self.s1d = Swish_Module()
        self.s2d = Swish_Module()
        
        
    def forward(self, x, x_rv,x_rlc,x_pe=None):
        
        x_rv = self.dropuot3d(x_rv)
        x_rv = self.Noise(self.avd3d(x_rv).reshape(1,-1))
        x_rv1 = self.s3d(self.linear_rv(x_rv))
        x_rv = x_rv.reshape(1,-1)
        x_rv1 = x_rv1.reshape(1,-1)
        
        
        x_rlc = self.dropuot3d(x_rlc)
        x_rlc = self.Noise(self.avd3d(x_rlc).reshape(1,-1))
        x_rlc1 = self.s3d(self.linear_rlc(x_rlc))
        x_rlc = x_rlc.reshape(1,-1)
        x_rlc1 = x_rlc1.reshape(1,-1)
        
        b,f = x.shape
        embedding = x.reshape(1,b,f)
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = self.s1d(self.linear1(h_lstm1))
        h_conc_linear2  = self.s2d(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2
        
        output = self.linear_pe(hidden)
        
        hidden2 = self.avd1d(hidden.transpose(2,1))
        hidden_rv = torch.cat([hidden2[:,:,0],x_rv+x_rv1],-1)
        hidden_rlc = torch.cat([hidden2[:,:,0],x_rlc+x_rlc1],-1)
        
        hidden_global = torch.cat([hidden_rv,hidden_rlc],-1)
        
        output_global0 = self.linear_global_fc0(hidden_global)
    
        output_global1 = self.linear_global_fc1(hidden_rv)
        
        output_global2 = self.linear_global_fc2(hidden_global)
        
        output_global3 = self.linear_global_fc3(hidden_rlc)
        return output,output_global0,output_global1,output_global2,output_global3


# In[ ]:


embed_size = x.shape[1]
embed_size


# In[ ]:


import torch
import numpy as np
from tqdm.auto import tqdm
import os
from apex import amp
import torch
import numpy as np
from tqdm.auto import tqdm
import os
from apex import amp
class trainer:
    def __init__(self,loss_fn,model,optimizer,scheduler,config):
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.metric = loss_fn
        self.CE = torch.nn.CrossEntropyLoss()

        
    def batch_train(self, x3d,x3d1,x3d2,batch_imgs, batch_labels0,batch_labels1,lebal3, batch_idx,e):
        batch_imgs = batch_imgs.cuda().float()[0]
        x3d = x3d.cuda().float()
        x3d1 = x3d1.cuda().float()
        x3d2 = x3d2.cuda().float()
        predicted = self.model(batch_imgs,x3d,x3d1,x3d2)
        batch_labels1 = batch_labels1[0]
        loss0,l0,rsna = self.loss_fn(predicted,batch_labels0,batch_labels1,lebal3)
        loss = rsna
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return l0.item(), predicted,rsna.detach().cpu().numpy(),loss0.item()

    def train_epoch(self, loader,e):
        self.model.train()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        rsna_all = 0
        loss_per_image = 0
        for batch_idx, (x3d,x3d1,x3d2,imgs,labels,labels1,lebal3) in enumerate(tqdm_loader):
            loss, predicted,rsna,loss0 = self.batch_train(x3d,x3d1,x3d2,imgs, labels,labels1,lebal3, batch_idx,e)
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
            rsna_all = (rsna_all * batch_idx + rsna) / (batch_idx + 1)
            loss_per_image = (loss_per_image * batch_idx + loss0) / (batch_idx + 1)
            tqdm_loader.set_description('loss: {:.4} rsna:{:.4}  im:{:.4} lr:{:.4}'.format(
                    current_loss_mean,rsna_all,loss_per_image, self.optimizer.param_groups[0]['lr']))
        return current_loss_mean
    
    def valid_epoch(self, loader,name="valid"):
        self.model.eval()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        current_loss_mean_image = 0
        current_loss_mean_scan = 0
        correct = 0
        loss_pre_class =[]
        rsna_all = []
        for batch_idx, (x3d,x3d1,x3d2,imgs,labels0,labels1,lebal3) in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_imgs = imgs.cuda().float()[0]
                x3d = x3d.cuda().float()[0]
                x3d1 = x3d1.cuda().float()[0]
                x3d2 = x3d2.cuda().float()[0]
                batch_labels0 = labels0.cuda().float()
                batch_labels1 = labels1[0]
                predicted = self.model(batch_imgs,x3d,x3d1,x3d2)
                loss0,l0,l1 = self.loss_fn(predicted,batch_labels0,batch_labels1,lebal3)
                rsna_all.append(l1.cpu().numpy())
                loss = l0.item()
                current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
                current_loss_mean_image = (current_loss_mean_image * batch_idx + l1.item()) / (batch_idx + 1)
                current_loss_mean_scan = (current_loss_mean_scan * batch_idx + loss0) / (batch_idx + 1)
                tqdm_loader.set_description(f"loss : {current_loss_mean:.4}, rsna : {current_loss_mean_image:.4}, image : {current_loss_mean_scan:.4}")
        print(f"rsna - {np.mean(rsna_all)}")
        score = 1-current_loss_mean
        print('metric {}'.format(score))
        return 1-np.mean(rsna_all)
    
    def run(self,train_loder,val_loder):
        best_score = -100000
        for e in range(self.config.epochs):
            print("----------Epoch {}-----------".format(e))
            current_loss_mean = self.train_epoch(train_loder,e)
            score = self.valid_epoch(val_loder)
            self.scheduler.step()
            if best_score < score:
                best_score = score
                torch.save(self.model.state_dict(),self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))
                print(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))

            
    def load_best_model(self):
        if os.path.exists(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)):
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)))
            print("load best model")


# In[ ]:


for g in [0.05]:
    for DO in [0.2]:
        print("*************************************************************************")
        print(f"dropout {DO}, Noise {g}")
        model = NeuralNet(embed_size=embed_size,DO=DO,g=g).cuda()
        model.cuda()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        plist = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = eval(model_config.optimizer)(plist,**model_config.optimizer_parm)
        scheduler = eval(model_config.scheduler)(optimizer,**model_config.scheduler_parm)
        loss_fn = eval(model_config.loss_fn)()
        model_config.model_name = model_config.model_name + "_fold_" +str(FOLD)
        print(scheduler,model_config.scheduler_parm,model_config.optimizer_parm)
        Trainer = trainer(ComboLoss(),model,optimizer,scheduler,config=model_config)
        Trainer.run(train,val)
        print("*************************************************************************")


# In[ ]:


v= iter(val)


# In[ ]:


Trainer.load_best_model()


# In[ ]:


# import matplotlib.patches as mpatches
# Trainer.model.eval()
# x3d,x3d1,_,x,y,y1,y2 = next(v)
# with torch.no_grad():
#     pred = Trainer.model(x.cuda().float()[0],x3d.cuda().float(),x3d1.cuda().float())
#     res = torch.sigmoid(pred[0].reshape(-1)).detach().cpu().numpy().reshape(-1)
#     res1 = torch.softmax(pred[1],dim=1).cpu().numpy().reshape(-1).tolist()
#     res1 = res1+ torch.softmax(pred[2],dim=1).cpu().numpy().reshape(-1).tolist()
#     res1 = res1+torch.sigmoid(pred[4]).cpu().numpy().reshape(-1).tolist()
#     res1 = res1+torch.softmax(pred[3],dim=1).cpu().numpy().reshape(-1).tolist()
#     y_numpy = y.detach().cpu().numpy().reshape(-1)
#     y1_numpy = y1.detach().cpu().numpy().reshape(-1).tolist() + y2.detach().cpu().numpy().reshape(-1).tolist() 
# plt.figure(figsize=[15,8])
# plt.subplot(131)
# plt.plot(y_numpy,label='gt',color='blue')
# # plt.plot(res/2+torch.sigmoid(x[0,:,:,-1]).cpu().numpy()/2,label='pred',color='green')
# plt.plot(res,label='pred',color='red')
# plt.xlabel("slices")
# plt.ylabel("PE")
# plt.ylim(0, 1.3)
# plt.title("lstm")
# red_patch = mpatches.Patch(color='red', label='pred')
# blue_patch = mpatches.Patch(color='blue', label='gt')
# plt.legend(handles=[red_patch,blue_patch])
# plt.subplot(132)
# plt.plot(y_numpy,label='gt',color='blue')
# plt.plot(torch.sigmoid(x[0,:,2048]).cpu().numpy(),color='green')
# # plt.plot(torch.sigmoid(x[0,0,:,2048]).cpu().numpy(),color='red')
# plt.ylim(0, 1.3)
# plt.title("classification - EfficientNet B5")
# plt.subplot(133)
# plt.plot(["PE","NEG","IND","RV>1","RV<1","NPE","R","L","C","CH","CH+AC","NPE2","AC"],y1_numpy,'o', color='red',label='gt')
# plt.plot(res1,'o', color='blue',label='pe')
# # plt.subplot(133)
# # plt.bar(np.arange(len(y1_numpy)),y1_numpy-res1)
# # plt.ylim(-1, 1)
# plt.show()
# print(x[0].shape)


# In[ ]:




