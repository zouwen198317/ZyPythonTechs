# -*- coding:UTF-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# MNIST数据集
MINIST_FILE = "..\..\..\data\MNIST_data"

# MNIST数据集
MINIST_FILE2 = "..\..\data\MNIST_data"

ROJECTOR_DIR = "..\..\..\projector\projector"

ROJECTOR_DATA = "..\..\..\projector\data"

# 图像识别 inception模型下载地址
INCEPTION_PRETRAIN_MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
NCEPTION_MODEL_File = "..\..\..\data\inception_model"
LABEL_LOOKUP_PATH = '..\..\..\data\inception_model\imagenet_2012_challenge_label_map_proto.pbtxt'
UID_LOOKUP_PATH = '..\..\..\data\inception_model\imagenet_synset_to_human_label_map.txt'
CLASSIFY_IMAGE_GRAPH_DEF_PB_PATH = '..\..\..\data\inception_model\classify_image_graph_def.pb'
