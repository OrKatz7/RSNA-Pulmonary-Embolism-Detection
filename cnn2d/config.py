import os
import json 
  
# Opening JSON file 
with open('../settings.json') as json_file: 
    settings = json.load(json_file) 

class data_config:
    dicom_file = '../'+settings['dicom_file_train']#'../Datasets/RSNA/dicom/train'
    train_csv_path = '../'+settings['train_csv_path']#'../Datasets/RSNA/train.csv'
    jpeg_dir = '../'+settings['jpeg_dir']#'../Datasets/RSNA/train256/'
#     jpeg_dir_512 = '../'+settings['dicom_file_train']#'../Datasets/RSNA/train512'
    
    ids = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    label = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    
    label_lstm = ['pe_present_on_image','negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    
    

        
class efficientnetb3:
    model_name="efficientnet-b3"
    batch_size = 30*3
    WORKERS = 30
    classes =1
    resume = True
    gpu = "0,1,2"
    epochs = 2
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-3,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':1000, 'eta_min':1e-6}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = "../"+settings['MODEL_PATH2D']
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
class efficientnetb4:
    model_name="efficientnet-b4"
    batch_size = 30*3
    WORKERS = 30
    classes =1
    resume = True
    gpu = "0,1,2"
    epochs = 2
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-3,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':1000, 'eta_min':1e-6}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = "../"+settings['MODEL_PATH2D']#'log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
class efficientnetb5:
    model_name="efficientnet-b5"
    batch_size = 48*4
    WORKERS = 30
    classes =1
    resume = True
    epochs = 4
    gpu = "0,1,2"
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-3,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':1000, 'eta_min':1e-6}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = "../"+settings['MODEL_PATH2D']#'log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

 