import numpy as np
import tensorflow as tf
import os, argparse
import cv2

from lucid.modelzoo.vision_models import Model

"""
Save CovidNet in a way usable by Lucid
"""

parser = argparse.ArgumentParser(description='SAVE COVID-NET FOR LUCID')
parser.add_argument('--weightspath', default='models/B', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1545', type=str, help='Name of model ckpts')
parser.add_argument('--in_name', default='input_1', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_name', default='norm_dense_1/Softmax', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

args = parser.parse_args()



with tf.Session() as sess:
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    tf.get_default_graph()

    Model.save(
        os.path.join(args.weightspath, 'lucid', 'covid_net.pb'),
        image_shape=[args.input_size, args.input_size, 3],
        input_name = args.in_name,
        output_names = [args.out_name],
        image_value_range=[0,1],
    )