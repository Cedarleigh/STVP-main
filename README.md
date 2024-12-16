# STVP-main
This is the official implementation of STVP, a simple and efficient neural architecture for viewpoint prediction in 3D point cloud videos. For technical details, please refer to:

Viewport Prediction for Volumetric Video Streaming by Exploring Video Saliency and Trajectory Information. [PaPer](https://arxiv.org/abs/2311.16462)

Jie Li, Zhixia Zhao, Zhixin Li, Zhi Liu, Pengyuan Zhou, Richang Hong, Qiyue Li, Han Hu
![image](imgs/overview.PNG)

### Setup Instructions
This implementation has been tested with Python 3.5, TensorFlow 1.11, CUDA 9.0, and cuDNN 7.4.1, running on Ubuntu 18.04.

- 1.Clone the repository 
```
git clone --depth=1 https://github.com/Cedarleigh/STVP-main && cd STVP-main
```
- 2.Setup python environment
```
conda create -n stvp python=3.5
source activate stvp
pip install -r stvp_requirements.txt
```
### (2) Dataset preparation and preprocessing
8i dataset can be found 
<a href="http://plenodb.jpeg.org/pc/8ilabs">here</a>. 
Download all the files. Uncompress the folder and move it to 
`/your/data/path`.

View data can be found 
<a href="https://github.com/Yong-Chen94/6DoF_Video_FoV_Dataset">here</a>. 
Download all the files. Uncompress the folder and move it to 
`/your/view_data/path`.

- Preparing the dataset:
```
python utils/data_prepare.py
```
### (3)Train and Tset
```
# train
python main.py --mode train --gpu 0
# test
python main.py --mode test --gpu 0
```
- Move all the generated results (*.ply) in `/test` folder to `/your/results/path`, calculate the final mean IoU results:
```
python utils/6_fold_cv.py
```
