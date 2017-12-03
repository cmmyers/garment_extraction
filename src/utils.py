import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from sklearn.cluster import KMeans

def rgb_to_bgr(image):
    r, g, b = cv2.split(image)
    return cv2.merge((b, g, r))

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge((r, g, b))


def get_background_one_image(filepath, num_points = 16):
    background_colors = []
    im = plt.imread(filepath)
    im = np.array(im, dtype=np.float64) / 255

    h, w, colors = im.shape

    for i in range(int(num_points/2)):
        row = np.random.choice(h)
        col = np.random.choice(xrange(int(w/8)))
        color_block = im[int(row-4):int(row+4), int(col-4):int(col+4)]
        ave_color = [avg_each_color(color_block, color) for color in ['r', 'g', 'b']]
        background_colors.append(ave_color)
    for i in range(int(num_points/2)):
        row = np.random.choice(h)
        col = np.random.choice(xrange(int((7*w/8)), w))
        color_block = im[int(row-4):int(row+4), int(col-4):int(col+4)]
        ave_color = [avg_each_color(color_block, color) for color in ['r', 'g', 'b']]
        background_colors.append(ave_color)

    return background_colors

def avg_each_color(color_block, color):
    color_dict = {'r':0, 'g':1, 'b':2}
    index = color_dict[color]
    all_colors = [color[index] for row in color_block for color in row]
    ave_color = sum(all_colors)/(len(all_colors) + 1)
    return ave_color
