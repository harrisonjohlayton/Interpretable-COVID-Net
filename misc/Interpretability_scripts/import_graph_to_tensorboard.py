
import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from data import process_image_file
from tensorflow.python.summary import summary

"""
create tensorboard log_dir for graph exploration
"""

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--weightspath', default='models/B', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1545', type=str, help='Name of model ckpts')
parser.add_argument('--log_dir', default='output/tensorboard_log/', type=str, help='Name of log directory')

args = parser.parse_args()

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

sess = tf.Session()
tf.get_default_graph()
saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
saver.restore(sess, os.path.join(args.weightspath, args.ckptname))

graph = tf.get_default_graph()

writer = summary.FileWriter(args.log_dir)
writer.add_graph(sess.graph)
print("done exporting model for tensorboard")