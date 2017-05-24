# KMC_cvprw_2017
This is the code for the paper "Kernalised Multi-resolution Convnet for Visual Tracking"


Purpose
=============
Code for monocular, generic object tracking.

******************************************************************************************************
Gist: Kernalised Correlation Filter -> Convnet Prediction
******************************************************************************************************
by Di WU: stevenwudi@gmail.com, 2017/05/01


Citation
-------
If you use this toolbox as part of a research project, please cite the corresponding paper
******************************************************************************************************
```yaml
@inproceedings{wu2017cvprw,
  title={Kernalised Multi-resolution Convnet for Visual Tracking},
  author={Wu, Di and Wenbin, Zou and Xia, Li and Yong, Zhao},
  booktitle={Proc. Conference on Computer Vision and Pattern Recognition (CVPR) Workshop},
  year={2017}
}
```
******************************************************************************************************


Dependency: Keras
-------
Some dependent libraries requirements:
Keras: for deep learning libarary:  https://github.com/fchollet/keras
Backend: tensorflow

	
Test
-------
To reproduce the experimental result for test submission, there is a Python file:

`....py` 

Train
-------
To train the network, you first need to extract the CNN from the OTB2015:

1)`step_1_OTB_100_collect_CNN.py`

Voila, here you go.

Dataset
-------
According to some reader recommendation, I supplement the links of the datasets used in the paper as follows:

1) `OTB-2015 Dataset` --> [http://cvlab.hanyang.ac.kr/tracker_benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/)


2) `UAV123Dataset` --> [https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)


Contact
-------
If you read the code and find it really hard to understand, please send feedback to: stevenwudi@gmail.com
Thank you!
