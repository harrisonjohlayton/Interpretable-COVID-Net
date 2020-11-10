import numpy as np
import tensorflow as tf
import os, argparse
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from data import crop_top, central_crop

"""
Performs the same operations as gradcam.py in home directory,
but outputs in a file structure that is easier to navigate when exploring
GradCams for interesting outputs.
"""

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

def process_image_file(filepath, top_percent, size):
    img = cv2.imread(filepath)
    img = crop_top(img, percent=top_percent)
    cropped_img = central_crop(img)
    new_img = cv2.resize(cropped_img, (size, size))
    return new_img, croped_img
  
class GradCAM:
    def __init__(self, graph, classes, outLayer, targetLayer=None):
        self.graph = graph
        self.classes = classes
        self.targetLayer = targetLayer
        self.outLayer = outLayer
        self.target = self.graph.get_tensor_by_name(self.targetLayer)


    def compute_grads(self):
        results = {} # grads of classes with keys being classes and values being normalized gradients
        for classIdx in self.classes:
            one_hot = tf.sparse_to_dense(classIdx, [len(self.classes)], 1.0)
            signal = tf.multiply(self.graph.get_tensor_by_name(self.outLayer),one_hot)
            loss = tf.reduce_mean(signal)

            grads = tf.gradients(loss, self.target)[0]

            norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads)))+tf.constant(1e-5))

            results[classIdx] = norm_grads

        return results


def generate_cam(conv_layer_out, grads_val, upsample_size):
    weights = np.mean(grads_val, axis=(0,1))
    cam = np.zeros(conv_layer_out.shape[0:2], dtype=np.float32)

    # Weight averaginng
    for i, w in enumerate(weights):
        cam += w*conv_layer_out[:,:,i]

    # Apply reLU
    cam = np.maximum(cam, 0)
    cam = cam/np.max(cam)
    cam = cv2.resize(cam, upsample_size)

    # Convert to 3D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3,[1,1,3])

    return cam3
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate GradCams for all input X-rays in the given test folder')
    parser.add_argument('--weights', default='models/B', type=str, help='Path to output folder')
    parser.add_argument('--meta', default='model.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckpt', default='model-1545', type=str, help='Name of model ckpts')
    parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
    parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
    parser.add_argument('--testfile', default='output/eval_results/testfile.txt', type=str, help='Name of testfile')
    parser.add_argument('--testfolder', default='data/test', type=str, help='Folder where test data is located')
    parser.add_argument('--outdir',default='./output/gradcam/', help="Output directory")
    parser.add_argument('--final_conv_tensor', default='conv5_block3_out/add:0', help='name of final convolutional tensor')
    parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
    parser.add_argument('--top_percent', default=0.0, type=float, help='percent to crop off top of image')

    args = parser.parse_args()

    sess = tf.Session()
    tf.get_default_graph()
    saver = tf.train.import_meta_graph(os.path.join(args.weights, args.meta))
    saver.restore(sess, os.path.join(args.weights, args.ckpt))

    graph = tf.get_default_graph()

    image_tensor = graph.get_tensor_by_name(args.in_tensorname)
    gradCam = GradCAM(graph=graph, classes = [0,1,2], outLayer=args.out_tensorname, targetLayer=args.final_conv_tensor)

    grads = gradCam.compute_grads()

    file = open(args.testfile, 'r')
    testfile = file.readlines()

    for i in range(len(testfile)):
        line = testfile[i].split()
        if (not os.path.isfile(os.path.join(args.testfolder, line[1]).replace('\\','/'))):
            print("MISSING: " + os.path.join(args.testfolder, line[1]).replace('\\','/'))
            continue
        print('testing ' + line[0])
        x, origin_im_cropped = process_image_file(os.path.join(args.testfolder, line[1]).replace('\\','/'), args.top_percent, args.input_size)
        img_arr = np.asanyarray(x)
        size_upsample = (origin_im_cropped.shape[1],origin_im_cropped.shape[0]) # (w, h)
        
        x = x.astype('float32') / 255.0

        for j in inv_mapping.keys():
            print('\t' + inv_mapping[j])
            output, grads_val = sess.run([gradCam.target, grads[j]], feed_dict={image_tensor: np.expand_dims(x, axis=0)})
            
            cam = generate_cam(output[0],grads_val[0],size_upsample)
            
            # Overlay cam on image
            cam = np.uint8(255*cam3)
            cam = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
            new_im = cam*0.3 + origin_im_cropped*0.5

            if (next_image_ground_truth == next_image_prediction):
                next_save_path = os.path.join(args.outdir, 'correct', 'ground_truth_' + next_image_ground_truth,
                    'heatmap_' + inv_mapping[j], next_image_name).replace('\\','/')
            else:
                next_save_path = os.path.join(args.outdir, 'incorrect', 'ground_truth_' + next_image_ground_truth,
                    'pred_' + next_image_prediction, 'heatmap_' + inv_mapping[j], next_image_name).replace('\\','/')
            cv2.imwrite(next_save_path,new_im)
    
    print("sorted GradCAM images saved in ", args.outdir)