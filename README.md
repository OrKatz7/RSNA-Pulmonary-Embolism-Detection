# Pulmonary Embolism Detection
 This is the source code for the 10th place solution to the kaggle - RSNA STR Pulmonary Embolism Detection.
# Full Pipeline
![alt text](https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/RSNA.PNG)
# Datastes and Preprocessing
## Windowing
###### RED channel / LUNG window / level=-600, width=1500
###### GREEN channel / PE window / level=100, width=700
###### BLUE channel / MEDIASTINAL window / level=40, width=400

## 1.dwonload datasets:
```
mkdir Datasets/RSNA
mkdir Datasets/RSNA/dicom
mkdir Datasets/RSNA/train256
cd Datasets/RSNA/dicom
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
unzip rsna-str-pulmonary-embolism-detection.zip
cp train.csv ../
cp test.csv ../
```
## 1.1 If you want to save the pre-processing time you can download the data set after pre-processing - based on Ian Pan
```
cd ../ # Now we in Datasets/RSNA
kaggle datasets download -d vaillant/rsna-str-pe-detection-jpeg-256
unzip -q rsna-str-pe-detection-jpeg-256.zip
mkdir train256
mv  train-jpegs/* train256/*
rm -rf rsna-str-pe-detection-jpeg-256.zip
cd ../../ # now we in root path
```
## 2. DATA PROCESSING (if dont use 1.1)
```
preprocessing.sh # If you have not performed Section 1.1
```

### 2.1 Data input/output
you can change the input/output path from settings.json
```
{"dicom_file_train": "Datasets/RSNA/dicom/train",

 "train_csv_path": "Datasets/RSNA/train.csv",
 
 "jpeg_dir": "Datasets/RSNA/train256/",
 
 "test_csv_path": "Datasets/RSNA/dicom/test.csv",
 
 "dicom_file_test": "Datasets/RSNA/dicom/test/",
 
 "sample_submission":"Datasets/RSNA/dicom/sample_submission.csv",
 
 "submission_file":"submission.csv",
 
 "MODEL_PATH2D":"cnn2d/log/cpt",
 
 "feature2D":"cnn2d/feature",
 
 "MODEL_PATH3D":"cnn3d/",
 
 "features_densenet121_pe":"cnn3d/features_densenet121_pe",
 
 "features_densenet121_rlc":"cnn3d/features_densenet121_rlc",
 
 "features_rv_lv":"cnn3d/features_rv_lv",
 
 "MODEL_PATH_LSTM":"lstm/log/cpt"}
```

# Data structures
```
$ kaggle competitions download -c rsna-str-pulmonary-embolism-detection
├── Datasets  
│   ├── RSNA      
│   │   ├── dicom
│   │   │    ├── train
│   │   │    ├── test
│   │   │    ├── test.csv
│   │   │    ├── train.csv
│   │   │    ├── sample_submission.csv
│   │   ├── train256
│   │   │    ├── 0003b3d648eb
│   │   │    ├── 000f7f114264
│   │   │    ├── ...
│   │   ├── train.csv
│   │   ├── test.csv
```
# Models Weights
## from the root path:
```
kaggle datasets download -d orkatz2/rsna20
unzip -q rsna20.zip 
mv cnn2d_cpt/* cnn2d/
rm -rf cnn2d_cpt
mv cnn3d_cpt/* cnn3d/
rm -rf cnn3d_cpt
mkdir lstm/cpt/
mkdir lstm/cpt/log
mv lstm_pe_fold_0_best.pth lstm/cpt/log/
mv lstm_pe_old_fold_0_best.pth lstm/cpt/log/
rm -rf rsna20.zip
```
# run only submission
```
source run_sub.sh
```
# run all
Edit settings.json
```
source preprocessing.sh
source run_all.sh
```
# run by steps
## CNN2D
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

## CNN3D
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
## Sequence Model
### trian sequence model per image and exam
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/lstm/train_lstm.ipynb
```


