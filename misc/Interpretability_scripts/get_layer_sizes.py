import numpy as np
import tensorflow as tf
import os, argparse
import cv2
import pandas as pd

"""
Script to print sizes of different operations in COVID-Net
Used for exploring COVID-Net for interesting layers to visualize and
to identify bottleneck layers for TCAV
"""

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--weightspath', default='models/B', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-1545', type=str, help='Name of model ckpts')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')

args = parser.parse_args()
weightspath = args.weightspath
metaname = args.metaname
ckptname = args.ckptname

if __name__ == '__main__':

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
    saver.restore(sess, os.path.join(weightspath, ckptname))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)

    # layer_names = [
    #     'conv5_block3_out/add:0',
    #     'conv5_block2_out/add:0',
    #     'conv5_block1_out/add:0',
    #     'conv4_block6_out/add:0',
    #     'conv4_block5_out/add:0',
    #     'conv4_block4_out/add:0',
    #     'conv4_block3_out/add:0',
    #     'conv4_block2_out/add:0',
    #     'conv4_block1_out/add:0',
    #     'conv3_block4_out/add:0',
    #     'conv3_block3_out/add:0',
    #     'conv3_block2_out/add:0',
    #     'conv3_block1_out/add:0',
    #     'conv2_block3_out/add:0',
    #     'conv2_block2_out/add:0',
    #     'conv2_block1_out/add:0',
    #     args.out_tensorname
    # ]

    for name in layer_names:
        print(graph.get_tensor_by_name(name))

    with tf.gfile.GFile('models/B/lucid/covid_net.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    #import graph def

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    
    # for i in range(3):
    #     for op in graph.get_operations():
    #         if ('conv5_block3_' + str(i+1) in op.name):
    #             # print(op.name)
    #             print(op.name + '\t' + str(graph.get_tensor_by_name(op.name + ':0').shape))
    # for op in graph.get_operations():
    #     if('conv5_block3_out' in op.name):
    #         print(op.name + '\t' + str(graph.get_tensor_by_name(op.name + ':0').shape))
    #         print(op.name)
    # for op in graph.get_operations():
    #     if 'block3_2_conv/EConv2D_16/Conv2D' in op.name:
    #         print(op.name)
    #         print(op.outputs[0])
    for op in graph.get_operations():
        if 'Softmax' in op.name or 'softmax' in op.name or 'logit' in op.name:
            print(op.name)
            print(op.outputs[0])


    #candidates in order
    '''



    conv5_block3_2_conv/Econv2d_16/Conv2D
        input size: 15x15x408 from conv5_block3_2_xonv/EConv2D_16
        output size: 15x15x400 to conv5_block3_2_bn/FusedBatchNorm and conv5_block3_2_bn/cond/FusedBatchNorm/Switch
    
    it might work on the ones above this one though. keep an eye on that.

    '''