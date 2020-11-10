import cv2
import numpy as np
import os, argparse
import math

"""
Create sprite atlas for channel visualizations
"""

parser = argparse.ArgumentParser(description='concantenate all images in folder to spritemap')
parser.add_argument('--input_dir', default='output/lucid/conv5/negative/')
parser.add_argument('--output_dir', default='output/lucid/conv5/')
parser.add_argument('--depth', default=2048)
args = parser.parse_args()

HORIZONTAL_AXIS = 1
VERTICAL_AXIS = 0

channel = 0
# shape = None

def read_img():
    global shape, channel

    if (channel < args.depth):
        img = cv2.imread(os.path.join(args.input_dir, str(channel) + '.png'))
        channel += 1
        # if (shape == None):
        #     shape = img.shape()
    else:
        img = np.zeros((128,128,3), dtype=np.int8)

    return img

if __name__ == '__main__':
    row_size = math.ceil(math.sqrt(args.depth))
    col_size = math.ceil(args.depth / row_size)
    
    rows = []
    for i in range(col_size):
        img = read_img()

        for j in range(1, row_size):
            img = np.concatenate((img, read_img()), HORIZONTAL_AXIS)
        
        rows.append(img)
    
    img = rows[0]
    for i in range(1, col_size):
        img = np.concatenate((img, rows[i]), VERTICAL_AXIS)
    
    print('Done!')
    cv2.imwrite(os.path.join(args.output_dir, 'negative.png'), img)