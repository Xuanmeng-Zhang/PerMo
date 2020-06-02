# PerMo

## Network & 3D reconstruction
### Requirements
* Python ≥ 3.6, PyTorch ≥ 1.4
* opencv, tqdm
* [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

### Usage
Step 1. Get the part segmentation, uv regression using our [pre-trained model](https://drive.google.com/file/d/1qsuVn1J4E3XJhrj9ijfjgm_1H1TToaM2/view?usp=sharing).
```
python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml stage_part_uv.pkl [path to images] --output part_uv_res.pkl -v
```
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_part.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_u.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_v.png" width="860"/>

Step 2. Sovle pose and reconstruct vehicle models from [Step 1's result](https://drive.google.com/file/d/1-3phQ23taaeO3mpo3z0DNAuMs60d40mI/view?usp=sharing).
Download the [template_models](https://drive.google.com/file/d/10o8a_TQo3633ArHikg0Pgkzb-ZJNfw-e/view?usp=sharing), [simplication_template_models](https://drive.google.com/file/d/1FC685JatxTlHmRwtnItfEkSZLWs926Ut/view?usp=sharing), [camera calib](https://drive.google.com/file/d/1VmX_S3jCYnfuj8CLKuv6X2x1tZ5IiB6q/view?usp=sharing). Modify config.yaml to set resource and ouput path.
```
python solve.py
```
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047.png" width="860"/>

## Dataset

## Labelling Tool

### Description

* This tool can be used to label the 6DOF pose and type of the vehicles in images. 
* We have successfully used this tool on Kitti and Apollo.


![Kitti labeled example](https://github.com/SA2020PerMo/PerMo/blob/master/3D_Tool/vis/006127.png)


### Requirements

* Ubuntu/MacOS/Windows
* Python3
* PyQt5
* opencv-python
* numpy

### Usage

    python win.py
* Slide the x, y, z, a, b and c to change the pose of the car.
* Choose car's type.
* The raw images are under /images.
* The label results are under /label_result.
* The camera information are under /calib.
* We provide 28 car models, which are under /models.
![](https://github.com/SA2020PerMo/PerMo/blob/master/3D_Tool/vis/tool2.png)




