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

        
    def batch_train(self, batch_imgs, batch_labels, batch_idx):
        batch_imgs, batch_labels = batch_imgs.cuda().float(), batch_labels.cuda().float()
        predicted = self.model(batch_imgs)
        loss = self.loss_fn(predicted.float(), batch_labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), predicted
    
    
    def train_epoch(self, loader):
        self.model.train()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        for batch_idx, (imgs,labels) in enumerate(tqdm_loader):
            loss, predicted = self.batch_train(imgs, labels, batch_idx)
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
            tqdm_loader.set_description('loss: {:.4} lr:{:.6}'.format(
                    current_loss_mean, self.optimizer.param_groups[0]['lr']))
            self.scheduler.step()
            if batch_idx % 2000==0:
                torch.save(self.model.state_dict(),self.config.MODEL_PATH+"/{}_last.pth".format(self.config.model_name))

        print(f"train loss {current_loss_mean}")
        return current_loss_mean
    
    def valid_epoch(self, loader,name="valid"):
        self.model.eval()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        correct = 0
        for batch_idx, (imgs,labels) in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_imgs = imgs.cuda().float()
                batch_labels = labels.cuda()
                predicted = self.model(batch_imgs)
                loss = self.loss_fn(predicted.float(),batch_labels.float()).item()
                current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
                tqdm_loader.set_description(f"loss - {current_loss_mean:.4}")
        score = 1-current_loss_mean
        print('metric {}'.format(score))
        return score
    
    def run(self,train_loder,val_loder):
        best_score = -100000
        for e in range(self.config.epochs):
            print("----------Epoch {}-----------".format(e))
            current_loss_mean = self.train_epoch(train_loder)
            score = self.valid_epoch(val_loder)
            if best_score < score:
                best_score = score
                torch.save(self.model.state_dict(),self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))
                print(f"save best model epoch {e} score {best_score}")


            
    def load_best_model(self):
        if os.path.exists(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)):
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)))
            print("load best model")
