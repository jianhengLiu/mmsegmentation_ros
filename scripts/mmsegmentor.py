#!/usr/bin/env python
# coding=utf-8
'''
Author: Jianheng Liu
Date: 2021-10-23 23:05:43
LastEditors: Jianheng Liu
LastEditTime: 2021-11-02 12:31:27
Description: MMSegmentor
'''

# Check Pytorch installation
# from vision_msgs.msg import Detection2D, \
#     Detection2DArray, \
#     ObjectHypothesisWithPose
import threading
from mmseg.models import build_segmentor
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
import rospy
import numpy as np
import cv2
import sys
import os

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmseg
import mmcv
from logging import debug
from mmcv import image
import torch
import torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
print(mmseg.__version__)

# Check mmcv installation
print(get_compiling_cuda_version())
print(get_compiler_version())


# ROS related imports

# NOTE:
# CvBridge meet problems since we are using python3 env
# We can do the data transformation manually
# from cv_bridge import CvBridge, CvBridgeError

# from mmdetection_ros.srv import *


class Segmentor:

    def __init__(self):
        # # Choose to use a config and initialize the detecto
        # Config file
        self.config_path = rospy.get_param('~config_path')
        # Checkpoint file
        self.checkpoint_path = rospy.get_param('~checkpoint_path')
        # Device used for inference
        self.device = rospy.get_param('~device', 'cuda:0')
        # Color palette used for segmentation map 色彩选择
        self.palette = rospy.get_param('~palette', 'cityscapes')
        # Opacity of painted segmentation map. In (0, 1] range. 设置透明度
        self.opacity = rospy.get_param('~opacity', 0.5)

        self._publish_rate = rospy.get_param('~publish_rate', 50)
        self._is_service = rospy.get_param('~is_service', False)
        self._visualization = rospy.get_param('~visualization', True)

        # build the model from a config file and a checkpoint file
        self.model = init_segmentor(
            self.config_path, self.checkpoint_path, device=self.device)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self.image_pub = rospy.Publisher("~debug_image", Image, queue_size=1)
        # self.object_pub = rospy.Publisher(
        #     "~objects", Detection2DArray, queue_size=1)
        # self.bridge = CvBridge()
        
        image_sub = rospy.Subscriber(
                "~image_topic", Image, self._image_callback, queue_size=1)

    # def generate_obj(self, result, id, msg):
    #     obj = Detection2D()
    #     obj.header = msg.header
    #     obj.source_img = msg
    #     result = result[0]
    #     obj.bbox.center.x = (result[0] + result[2]) / 2
    #     obj.bbox.center.y = (result[1] + result[3]) / 2
    #     obj.bbox.size_x = result[2] - result[0]
    #     obj.bbox.size_y = result[3] - result[1]

    #     obj_hypothesis = ObjectHypothesisWithPose()
    #     obj_hypothesis.id = str(id)
    #     obj_hypothesis.score = result[4]
    #     obj.results.append(obj_hypothesis)

    #     return obj

    def run(self):
        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                # objArray = Detection2DArray()
                # try:
                #     cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # except CvBridgeError as e:
                #     print(e)
                # NOTE: This is a way using numpy to convert manually
                im = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                # image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                image_np = np.asarray(im)

                # Use the detector to do inference
                # NOTE: inference_detector() is able to receive both str and ndarray
                result = inference_segmentor(self.model, image_np)

                # objArray.detections = []
                # objArray.header = msg.header
                # object_count = 1

                # for i in range(len(result)):
                #     print(result[i])
                #     print(result[i].shape)
                #     if result[i].shape != (0, 5):
                #         object_count += 1
                #         objArray.detections.append(
                #             self.generate_obj(result[i], i, msg))

                # if not self._is_service:
                #     self.object_pub.publish(objArray)
                # # else:
                # #     rospy.loginfo('RESPONSING SERVICE')
                # #     return mmdetSrvResponse(objArray)

                # # Visualize results
                if self._visualization:
                    # NOTE: Hack the provided visualization function by mmdetection
                    # Let's plot the result
                    # show_result_pyplot(self.model, image_np, results, score_thr=0.3)
                    # if hasattr(self.model, 'module'):
                    # m = self.model.module
                    debug_image = self.model.show_result(
                        image_np, result, palette=get_palette(self.palette), show=False, opacity=self.opacity)
                    # img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    # image_out = Image()
                    # try:
                    # image_out = self.bridge.cv2_to_imgmsg(img,"bgr8")
                    # except CvBridgeError as e:
                    #     print(e)
                    # image_out.header = msg.header
                    image_out = msg
                    # NOTE: Copy other fields from msg, modify the data field manually
                    # (check the source code of cvbridge)
                    image_out.data = debug_image.tobytes()

                    self.image_pub.publish(image_out)
            rate.sleep()

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

    def service_handler(self, request):
        return self._image_callback(request.image)


def main():
    rospy.init_node('mmdetector')

    obj = Segmentor()
    obj.run()


if __name__ == '__main__':
    main()
