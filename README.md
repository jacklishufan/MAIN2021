# MAIN: A Multi-Stage Audiogram Interpretation Network.

## File System Structure

``` 
├── annotations // Put JSON annotation files here
├── input_images // Put Open Audiogram dataset here
│   ├── all // All Raw Camera Photos
│   ├── cropped // All Cropped Camera Photos
│   ├── scanned // All Scanned Photos
│   └── test // Test Camera Photos
├── json_result // Output JSON Dir 
├── models // Put models here
│   ├── axis_detector 
│   ├── gram_detector
│   └── mask_detector
├── output_images // Output Image (Visualization) Dir 
├── utils // Helper Codes
├── acc_test.py
├── run_all_benchmark.sh
├── train_axis.py
├── trainer.py
├── train_gram.py
├── train_mask.py
```

## Dataset and Pretrained Weights

https://www.dropbox.com/sh/n277svr60go1k54/AAAZ4rW_KuF4UAt8NF-JjuTja?dl=0

## Replicate Results in Paper
1. Put the annotations and images from Open Audiogram Dataset into the correct location. Make sure your file system agree with the tree in previous section. 
2. Run `python train_<xxx>.py` for gram, axis and mask, or download pretrained models and put them into respective folder.
3. run `run_all_benchmark.sh` to generate outputs.
4. run `python acc_test.py` to get the metrics
Example output:


```

Summary of json_result/result_baseline_rectification_none.json
              enrty    recall  precision
0        All Labels  0.757119   0.718601
1  Frequency Labels  0.967337   0.918124
2       Loss Labels  0.804858   0.763911
3   +-5 Loss Labels  0.938023   0.890302
Summary of json_result/result_baseline_rectification_vp.json
              enrty    recall  precision
0        All Labels  0.856784   0.831707
1  Frequency Labels  0.969849   0.941463
2       Loss Labels  0.882747   0.856911
3   +-5 Loss Labels  0.960637   0.932520
Summary of json_result/result_baseline_rectification_mask.json
              enrty    recall  precision
0        All Labels  0.850467   0.849673
1  Frequency Labels  0.966355   0.965453
2       Loss Labels  0.873832   0.873016
3   +-5 Loss Labels  0.957009   0.956116
Summary of json_result/result_baseline_scanned.json
              enrty    recall  precision
0        All Labels  0.987234   0.991453
1  Frequency Labels  0.987234   0.991453
2       Loss Labels  0.987234   0.991453
3   +-5 Loss Labels  0.987234   0.991453

```
Note that results may have small variations due to the uncertainty caused by RANSAC algorithm.


## Inference on New Images

``` 
python baseline.py [-h] [--input_img_dir INPUT_IMG_DIR] [--output_json_pth OUTPUT_JSON_PTH] [--output_img_dir OUTPUT_IMG_DIR] [--axis_detector_model AXIS_DETECTOR_MODEL]
                   [--gram_detector_model GRAM_DETECTOR_MODEL] [--mask_model MASK_MODEL] [--cpu] [--gpu] [--rectification RECTIFICATION]

```
The detailed descriptions of the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
| input_img_dir | Path to the input image folder (defaults to './input_images/test') |
| output_json_pth | Path to the JSON output folder (defaults to './json_result/result_baseline.json') |
| output_img_dir |  Path to the Image (Visualization) output folder  (defaults to './output_images') |
| axis_detector_model | Path to the Axis and Mark Detector weights  (defaults to './models/axis_detector/model_final.pth') |
| gram_detector_model  | Path to the Gram Detector weights  (defaults to './models/gram_detector/model_final.pth') |
| mask_model  | Path to the Axis and Mask Detector weights  (defaults to './models/mask_detector/model_final.pth') |
| cpu | Trigger for using CPU (default False, i.e. using GPU) |
| rectification | Methods for perspective rectification. Should be 'none', 'vp', or 'mask' (See paper for detail) (defaults to `none`) |
