

<!--
 * @Author: Jianheng Liu
 * @Date: 2021-10-23 22:34:33
 * @LastEditors: Jianheng Liu
 * @LastEditTime: 2021-10-24 15:08:11
 * @Description: Description
-->
# mmsegmentation-ros
This is a ROS package for segmentation, which utilizes the toolbox [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) of [OpenMMLab](https://openmmlab.com/).

### Requirements

- ROS Melodic
- Python 3.6+, PyTorch 1.3+, CUDA 9.2+ and [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

### Installation

1. Clone packages

   ```bash
   git clone https://github.com/jianhengLiu/mmsegmentation_ros.git 
   cd mmdetection_ros
   git clone https://github.com/open-mmlab/mmsegmentation.git
   ```

   

2. `MMSegmentation` Requirements, please refer to https://mmsegmentation.readthedocs.io/en/latest/get_started.html#installation

3. Install `rospkg`.

   ```bash
   pip install rospkg
   ```


### ROS Interfaces

#### params

- `~publish_rate`: the debug image publish rate. default: 50hz
- `~is_service`: whether or not to use service instead of subscribe-publish
- `~visualization`: whether or not to show the debug image

#### topics

- `~debug_image`: publish the debug image
- `~objects`: publish the inference result, containing the information of detected objects
- `~image`: subscribe the input image. 

> Thanks: [mmdetection-ros](https://github.com/jcuic5/mmdetection-ros)