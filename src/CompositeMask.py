import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from sklearn.cluster import KMeans
import utils


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

class CompositeMask():
    def __init__(self, path, threshold = 0.4, num_bg_colors = 16):
        self.load_and_resize_image(path)
        bg_pal = get_background_one_image(path, num_bg_colors)
        self.bg_color_list = [[c[2], c[1], c[0]] for c in bg_pal]
        self.threshold = threshold
        self.make_many_masks()
        self.composite_mask = sum(self.masks)
        self.make_masked_image()
        self.get_contours()

    def load_and_resize_image(self, path):
        im = plt.imread(path)
        size = im.shape
        r = 200.0 / im.shape[1]
        dim = (200, int(im.shape[0] * r))
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        im = rgb_to_bgr(resized)
        self.image = im

    def make_mask(self, three_colors):
        lower = np.array([(c - self.threshold*c)*255 for c in three_colors])
        upper = np.array([(c + self.threshold*c)*255 for c in three_colors])
        shapeMask = cv2.inRange(self.image, lower, upper)
        contoured_img, contours, hierarchy =                             cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contoured_img

    def make_many_masks(self):
        self.masks = []
        for three_colors in self.bg_color_list:
            self.masks.append(self.make_mask(three_colors))

    def plot_many_masks(self):
        num_bg_colors = len(self.masks)
        num_rows = int(num_bg_colors/4) + 1
        subplot_location = 0
        fig = plt.figure(figsize=(10, 12))
        for m in self.masks:
            subplot_location += 1
            ax = fig.add_subplot(num_rows, 4, subplot_location)
            ax.imshow(m)
        plt.tight_layout()
        plt.show()

    def make_masked_image(self):
        orig_shape = self.image.shape
        new_image = []
        flat_mask = [p for row in self.composite_mask for p in row]
        flat_image = [p for row in self.image for p in row]
        for m, p in zip(flat_mask, flat_image):
            if m != 0:
                new_image.append(np.array([57, 255, 20]))
            else: new_image.append(p)
        new_image = np.array(new_image)
        new_image = new_image.reshape(orig_shape)
        new_image = new_image.astype('uint8')
        new_image = bgr_to_rgb(new_image)
        self.masked_image = new_image

    def show_masked_image(self):
        im_rgb = self.masked_image
        plt.imshow(im_rgb)
        plt.show()

    def show_orig_image(self):
        im_rgb = bgr_to_rgb(self.image)
        plt.imshow(im_rgb)
        plt.show()

    def get_contours(self):
        lower = 0
        upper = 15
        shapeMask = cv2.inRange(self.composite_mask, lower, upper)
        self.contoured_img, self.contours, self.hierarchy =    \
                     cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    def plot_largest_contour(self):

        largest_contour = max(cv2.contourArea(cnt) for cnt in self.contours)
        temp_im = self.image.copy()
        for c in self.contours:
            # draw the contour and show it
            if cv2.contourArea(c) == largest_contour:
                cv2.drawContours(temp_im, [c], 0, (255, 255, 0), 3)
                plt.imshow(bgr_to_rgb(temp_im))
                break

        plt.show()
