from shutil import copy2
from random import shuffle
import numpy as np
import os
import cv2
import pandas as pd

"""
randomly select 50 examples from each output class to use
in TCAV
"""

COVID_LABEL = 'COVID-19'
PNUEMONIA_LABEL = 'pneumonia'
NORMAL_LABEL = 'normal'
IMAGE_DIR = 'data/train/'
OUTPUT_PNEUMONIA = 'data/tcav/pneumonia/'
OUTPUT_COVID = 'data/tcav/covid/'
OUTPUT_NORMAL = 'data/tcav/normal/'
LIST_FILE = 'data/lists/train_split.txt'

def read_csv(fd, image_no):
    normal = []
    pneumonia = []
    covid = []

    lines = fd.readlines()
    #randomize the lines
    shuffle(lines)
    i = 0
    while ((len(normal) < image_no) or (len(pneumonia) < image_no) or (len(covid) < image_no)):
        line = lines[i].split()
        i += 1
        label = line[2]
        image_name = line[1]
        if (os.path.isfile(os.path.join(IMAGE_DIR, image_name))):
            if ((label == COVID_LABEL) and (len(covid) < image_no)):
                covid.append(image_name)
            elif ((label == NORMAL_LABEL) and (len(normal) < image_no)):
                normal.append(image_name)
            elif ((label == PNUEMONIA_LABEL) and (len(pneumonia) < image_no)):
                pneumonia.append(image_name)

    return normal, pneumonia, covid

def copy_image(image_name, output_dir):
    copy2(os.path.join(IMAGE_DIR, image_name), output_dir)

if __name__ == "__main__":
    fd = open(LIST_FILE, "r")
    normal, pneumonia, covid = read_csv(fd, 50)
    for name in normal:
        print('copying normal pictures...')
        copy_image(name, OUTPUT_NORMAL)
    for name in pneumonia:
        print('copying pneumonia pictures...')
        copy_image(name, OUTPUT_PNEUMONIA)
    for name in covid:
        print('copying covid pictures...')
        copy_image(name, OUTPUT_COVID)
    print('done!')