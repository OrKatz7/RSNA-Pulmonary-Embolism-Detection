# Pulmonary Embolism Detection
 This is the source code for the 11th place solution to the kaggle - RSNA STR Pulmonary Embolism Detection.
# Full Pipeline
![alt text](https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/RSNA.PNG)
# Datastes and Preprocessing
## dwonload datasets:
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
## preprocessing - based on Ian Pan:
https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930
kaggle datasets download -d vaillant/rsna-str-pe-detection-jpeg-256

## Windowing
RED channel / LUNG window / level=-600, width=1500
GREEN channel / PE window / level=100, width=700
BLUE channel / MEDIASTINAL window / level=40, width=400

# CNN2D
1. train -
```
%cd cnn2d
$source train.sh
```
2. predict
```
%cd cnn2d
$source predict.sh
```

# CNN3D
1.trian and predict expert model for pe/non_pe
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/negative_exam_for_pe.ipynb
```
2.trian and predict expert model for rv/lv
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/rv_lv_ratio.ipynb
```
2.trian and predict expert model for side_pe
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/cnn3d/sided_pe.ipynb
```
# Sequence Models
trian sequence model per image and exam
```
https://github.com/OrKatz7/RSNA-Pulmonary-Embolism-Detection/blob/main/lstm/train_lstm.ipynb
```


