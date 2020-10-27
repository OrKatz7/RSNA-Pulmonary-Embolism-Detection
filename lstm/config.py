import os
class data_config:
    dicom_file = '../Datasets/RSNA/dicom/train'
    train_csv_path = '../Datasets/RSNA/train.csv'
    jpeg_dir = '../Datasets/RSNA/train256/train-jpegs'
    ids = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
    label = ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    
    label_lstm = ['pe_present_on_image','negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe','rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']
    

class lstm_pe:
    model_name="lstm_pe"
    dirs = ['../cnn2d/feature/efficientnet-b5_cnn_*','../cnn2d/feature/efficientnet-b4_cnn_*','../cnn2d/feature/efficientnet-b3_cnn_*']
    batch_size = 1
    WORKERS = 30
    epochs = 6
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':5e-5,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.StepLR"
    scheduler_parm = {'step_size':1, 'gamma':0.67}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
    MODEL_PATH = 'log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        
 