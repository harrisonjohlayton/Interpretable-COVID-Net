import numpy as np
import tensorflow as tf
import os, argparse
import cv2
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

from lucid.misc.io import show
from lucid.modelzoo.vision_models import Model
from lucid_load_model import CovidNetB

#suppress warnings from tensorflow
from tensorflow.python.util import deprecation


parser = argparse.ArgumentParser(description='generate lucid visualisations for all channels in given layer')
parser.add_argument('--outpath', default='output/lucid/conv5/negative', type=str, help='path to output folder')
parser.add_argument('--layer', default='conv5_block3_out/add')
parser.add_argument('--depth', default=2048, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    model = CovidNetB()
    model.load_graphdef()

    for i in range(args.depth):
        param_f = lambda: tf.tile(tf.math.reduce_mean(param.image(128, fft=True), axis=3, keepdims=True), [1,1,1,3])
        obj = -objectives.channel("conv5_block3_out/add", i)

        deprecation._PRINT_DEPRECATION_WARNINGS = False
        tf.logging.set_verbosity(tf.logging.ERROR)

        print("\nProgress: " + str(i) + '/' + str(args.depth) + '\n')

        img = render.render_vis(model, obj, param_f, transforms = transform.standard_transforms, thresholds=[200,])
        img = np.uint8(255*(img[0]))
        cv2.imwrite(os.path.join(args.outpath, str(i) + '.png').replace('\\','/'),img[0])

