# PerMo

## Network & 3D reconstruction
### Requirements
* Python ≥ 3.6, PyTorch ≥ 1.4
* opencv, tqdm
* [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

### Usage
Step 1. Get the part segmentation, uv regression using our pre-trained model.
```
python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_WC2_s1x.yaml stage_part_uv.pkl [path to images] --output part_uv_res.pkl -v
```
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_part.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_u.png" width="860"/>
<img src="https://github.com/SA2020PerMo/PerMo/blob/master/vis/004047_v.png" width="860"/>

Step 2. Sovle pose and reconstruct vehicle models from [Step 1's result](https://pan.baidu.com/s/1AieXOTvlRNGL4GQGdit83w)(password:rbpm).
Download the template_models, simplication_template_models, camera calib from [here](https://pan.baidu.com/s/1DlEVKVbqcxzr9F3DxeqhXQ)(password:7ssf). Modify config.yaml to set resource and ouput path.
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

### Kitti Label Results

The kitti label results can be download at [BaiduNetdisk](https://pan.baidu.com/s/1HnoZ3AAf1-xeFT7uTs-zLg)(password:330b)


