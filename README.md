# Pulmonary Embolism Detection
 This is the source code for the 10th place solution to the kaggle - RSNA STR Pulmonary Embolism Detection.
# Full Pipeline
![alt text](https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/RSNA.PNG)
# Datastes and Preprocessing
## dwonload datasets:
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
## preprocessing - based on Ian Pan:

https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930

kaggle datasets download -d vaillant/rsna-str-pe-detection-jpeg-256

## Windowing
###### RED channel / LUNG window / level=-600, width=1500
###### GREEN channel / PE window / level=100, width=700
###### BLUE channel / MEDIASTINAL window / level=40, width=400

# Data structures
```
$ kaggle competitions download -c rsna-str-pulmonary-embolism-detection
     
|-- Datasets
|   `-- RSNA
|       |-- dicom
|       `-- train256
|-- LICENSE
|-- README.md
|-- RSNA.PNG
|-- cnn2d
|   |-- config.py
|   |-- data.py
|   |-- feature
|   |   |-- efficientnet-b2_cnn_0
|   |   |-- efficientnet-b2_cnn_1
|   |   |-- efficientnet-b2_cnn_2
|   |   |-- efficientnet-b2_cnn_3
|   |   |-- efficientnet-b2_cnn_4
|   |   |-- efficientnet-b3_cnn_0
|   |   |-- efficientnet-b3_cnn_1
|   |   |-- efficientnet-b3_cnn_2
|   |   |-- efficientnet-b3_cnn_3
|   |   |-- efficientnet-b3_cnn_4
|   |   |-- efficientnet-b4_cnn_0
|   |   |-- efficientnet-b4_cnn_1
|   |   |-- efficientnet-b4_cnn_2
|   |   |-- efficientnet-b4_cnn_3
|   |   |-- efficientnet-b4_cnn_4
|   |   |-- efficientnet-b5_cnn_0
|   |   |-- efficientnet-b5_cnn_1
|   |   |-- efficientnet-b5_cnn_2
|   |   |-- efficientnet-b5_cnn_3
|   |   |-- efficientnet-b5_cnn_4
|   |   `-- efficientnet-b6_cnn_0
|   |-- log
|   |   `-- cpt
|   |   |   `-- efficientnet-b3_cnn_0_best.pth
|   |   |   |-- efficientnet-b3_cnn_0_last.pth
|   |   |   |-- efficientnet-b3_cnn_1_best.pth
|   |   |   |-- efficientnet-b3_cnn_2_best.pth
|   |   |   |-- efficientnet-b3_cnn_3_best.pth
|   |   |   |-- efficientnet-b3_cnn_4_best.pth
|   |   |   |-- efficientnet-b4_cnn_0_best.pth
|   |   |   |-- efficientnet-b4_cnn_1_best.pth
|   |   |   |-- efficientnet-b4_cnn_2_best.pth
|   |   |   |-- efficientnet-b4_cnn_3_best.pth
|   |   |   |-- efficientnet-b4_cnn_4_best.pth
|   |   |   |-- efficientnet-b5_cnn_0_best.pth
|   |   |   |-- efficientnet-b5_cnn_1_best.pth
|   |   |   |-- efficientnet-b5_cnn_2_best.pth
|   |   |   |-- efficientnet-b5_cnn_3_best.pth
|   |   |   `-- efficientnet-b5_cnn_4_best.pth
|   |-- models.py
|   |-- predict.sh
|   |-- predict_feature.py
|   |-- preprocessing.py
|   |-- train.py
|   |-- train.sh
|   |-- train_utils.py
|   `-- utils.py
|-- cnn3d
|   |-- densenet121_best_fold0.pth
|   |-- densenet121_best_fold1.pth
|   |-- densenet121_best_fold2.pth
|   |-- densenet121_best_fold3.pth
|   |-- densenet121_best_fold4.pth
|   |-- densenet121_model_fold0.pth
|   |-- densenet121_model_fold1.pth
|   |-- densenet121_model_fold2.pth
|   |-- densenet121_model_fold3.pth
|   |-- densenet121_pe_best_fold0.pth
|   |-- densenet121_pe_best_fold1.pth
|   |-- densenet121_pe_best_fold2.pth
|   |-- densenet121_pe_best_fold3.pth
|   |-- densenet121_pe_best_fold4.pth
|   |-- densenet121_pe_model_fold0.pth
|   |-- densenet121_pe_model_fold1.pth
|   |-- densenet121_pe_model_fold2.pth
|   |-- densenet121_pe_model_fold3.pth
|   |-- densenet121_pe_model_fold4.pth
|   |-- densenet121_rlc_best_fold0.pth
|   |-- densenet121_rlc_best_fold1.pth
|   |-- densenet121_rlc_best_fold2.pth
|   |-- densenet121_rlc_best_fold3.pth
|   |-- densenet121_rlc_best_fold4.pth
|   |-- densenet121_rlc_model_fold0.pth
|   |-- densenet121_rlc_model_fold1.pth
|   |-- densenet121_rlc_model_fold2.pth
|   |-- densenet121_rlc_model_fold3.pth
|   |-- densenet121_rlc_model_fold4.pth
|   |-- features_densenet121_pe
|   |-- features_densenet121_rlc
|   |-- features_rv_lv
|   |-- negative_exam_for_pe.ipynb
|   |-- rv_lv_ratio.ipynb
|   `-- sided_pe.ipynb
|-- lstm
|   |-- config.py
|   |-- log
|   |   `-- cpt
|   |   |    |-- lstm_pe_fold_0_best.pth
|   |   |    `-- lstm_pe_old_fold_0_best.pth
|   |-- losses.py
|   |-- train_lstm.ipynb
|   |-- train_lstm_old.ipynb
|   `-- utils.py
`-- submission.ipynb

```
# CNN2D
### 1. train -
Edit data_config in cnn2d/config.py
```
%cd cnn2d
$python3 preprocessing.py
$source train.sh
```
### 2. predict
```
%cd cnn2d
$source predict.sh
```

# CNN3D
### 1.trian and predict expert model for pe/non_pe
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/negative_exam_for_pe.ipynb
```
### 2.trian and predict expert model for rv/lv
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/rv_lv_ratio.ipynb
```
### 3.trian and predict expert model for side_pe
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/sided_pe.ipynb
```
# Sequence Model
### trian sequence model per image and exam
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/lstm/train_lstm.ipynb
```


