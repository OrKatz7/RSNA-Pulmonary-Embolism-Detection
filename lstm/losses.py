from torch import nn
import numpy as np
import torch


CFG = {
    'image_target_cols': [
        'pe_present_on_image', # only image level
    ],
    'exam_target_cols': [
        'negative_exam_for_pe', # exam level
        'rv_lv_ratio_gte_1', # exam level
        'rv_lv_ratio_lt_1', # exam level
        'leftsided_pe', # exam level
        'chronic_pe', # exam level
        'rightsided_pe', # exam level
        'acute_and_chronic_pe', # exam level
        'central_pe', # exam level
        'indeterminate' # exam level
    ], 
    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785, 0.1877300613, 0.09202453988],
}
def rsna_metric(label, predicted ,bce_func = torch.nn.BCELoss(reduction='none'),CFG=CFG):
    y_true_img = label[0].reshape(-1)
    
    y_true_exam = torch.zeros([9]).cuda()
    PE_NEG_IND = label[1].reshape(-1)
    RV_RV1_NEG = label[2].reshape(-1)
    RLC = label[4].reshape(-1)
    CH_CHAC_NEG_AC = label[3].reshape(-1)
    
    y_true_exam[0] = (PE_NEG_IND[1] + RV_RV1_NEG[2] + CH_CHAC_NEG_AC[2])/3
    y_true_exam[1] = RV_RV1_NEG[0]
    y_true_exam[2] = RV_RV1_NEG[1]
    y_true_exam[3] = RLC[1]
    y_true_exam[4] = CH_CHAC_NEG_AC[0]
    y_true_exam[5] = RLC[0]
    y_true_exam[6] = CH_CHAC_NEG_AC[1]
    y_true_exam[7] = RLC[2]
    y_true_exam[8] = PE_NEG_IND[2]
    
    y_pred_img = torch.sigmoid(predicted[0]).reshape(-1)
    max_y_pred_img = y_pred_img.max()
    PE_NEG_IND = torch.softmax(predicted[1],dim=1).reshape(-1)
    RV_RV1_NEG = torch.softmax(predicted[2],dim=1).reshape(-1)
    RLC = torch.sigmoid(predicted[4]).reshape(-1)
    CH_CHAC_NEG_AC = torch.softmax(predicted[3],dim=1).reshape(-1)
    y_pred_exam = torch.zeros([9]).cuda()
    y_pred_exam[0] = (PE_NEG_IND[1] + RV_RV1_NEG[2] + CH_CHAC_NEG_AC[2])/3
    y_pred_exam[1] = RV_RV1_NEG[0]
    y_pred_exam[2] = RV_RV1_NEG[1]
    y_pred_exam[3] = RLC[1]
    y_pred_exam[4] = CH_CHAC_NEG_AC[0]
    y_pred_exam[5] = RLC[0]
    y_pred_exam[6] = CH_CHAC_NEG_AC[1]
    y_pred_exam[7] = RLC[2]
    y_pred_exam[8] = PE_NEG_IND[2]
                
    total_loss = torch.tensor(0, dtype=torch.float32).cuda()
    total_weights = torch.tensor(0, dtype=torch.float32).cuda()
    
    label_w = torch.tensor(CFG['exam_weights']).view(1, -1).cuda()
    img_w = CFG['image_weight']
    
    exam_loss = bce_func(y_pred_exam, y_true_exam)
    exam_loss = torch.sum(exam_loss*label_w, 1)[0] # Kaggle us
    
    image_loss = bce_func(y_pred_img, y_true_img)
    qi = torch.sum(y_true_img)/len(y_true_img)
    image_loss = torch.sum(img_w*qi*image_loss)
    total_loss += exam_loss+image_loss
    total_weights += label_w.sum() + img_w*qi*len(y_true_img)
    final_loss = total_loss/total_weights
    return final_loss



class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.CE = torch.nn.CrossEntropyLoss()
        self.rsna_metric = rsna_metric
    def forward(self,predicted,batch_labels0,batch_labels1,lebal3):
        loss0 = self.bce(predicted[0].float().reshape(-1), batch_labels0.reshape(-1).cuda().float())
        loss1 = self.CE(predicted[1].float().reshape(1,-1), torch.tensor([torch.argmax(batch_labels1[0].reshape(-1))]).cuda())
        loss2 = self.CE(predicted[2].float().reshape(1,-1), torch.tensor([torch.argmax(batch_labels1[1].reshape(-1))]).cuda())
        loss3 = self.CE(predicted[3].float().reshape(1,-1), torch.tensor([torch.argmax(lebal3.reshape(-1))]).cuda())
        loss4 = self.bce(predicted[4].float().reshape(-1), batch_labels1[2].reshape(-1).cuda().float())
        rsna = self.rsna_metric([batch_labels0.cuda().float(),batch_labels1[0].cuda().float(),batch_labels1[1].cuda().float(),lebal3.cuda().float(),batch_labels1[2]],predicted)
        loss = loss0+loss1+loss2+loss3+loss4
        loss = loss/5.0
        return loss0,loss,rsna