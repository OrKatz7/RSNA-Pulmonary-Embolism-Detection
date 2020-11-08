import os
import json 
  
# Opening JSON file 
with open('../settings.json') as json_file: 
    settings = json.load(json_file)

class data_config:
    dicom_file = '../'+settings['dicom_file_train']#'../Datasets/RSNA/dicom/train'
    train_csv_path = '../'+settings['train_csv_path']#'../Datasets/RSNA/train.csv'
    jpeg_dir = '../'+settings['jpeg_dir']#'../Datasets/RSNA/train256/'
    
    ids = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    label = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    
    label_lstm = ['pe_present_on_image','negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    
class lstm_pe:
    model_name="lstm_pe"
    feature2D = settings['feature2D']
    dirs = [f"../{feature2D}/efficientnet-b5_cnn_*",
            f'../{feature2D}/efficientnet-b4_cnn_*',
            f'../{feature2D}/efficientnet-b3_cnn_*']   
    batch_size = 1
    WORKERS = 30
    epochs = 10
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':5e-5,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.StepLR"
    scheduler_parm = {'step_size':1, 'gamma':0.67}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = "../"+settings['MODEL_PATH_LSTM']
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
class lstm_pe_old:
    model_name="lstm_pe_old"
    feature2D = settings['feature2D']
    dirs = [f"../{feature2D}/efficientnet-b5_cnn_*",
            f'../{feature2D}/efficientnet-b4_cnn_*',
            f'../{feature2D}/efficientnet-b3_cnn_*']   
    batch_size = 1
    WORKERS = 30
    epochs = 6
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':5e-5,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.StepLR"
    scheduler_parm = {'step_size':1, 'gamma':0.67}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = "../"+settings['MODEL_PATH_LSTM']
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)



        
 