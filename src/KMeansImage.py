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

class KMeansImage():

    def __init__(self, image, num_colors):
        self.image = image
        self.num_colors = num_colors
        self.make_kmeans_image()

    def flatten_image(self):
        im = np.array(self.image, dtype=np.float64) / 255
        w, h, colors = im.shape
        image_flat = np.reshape(im, (w * h, colors))
        return image_flat

    def kmeans_palette_one_image(self, sample_size = 500, seed = 17):
        image = self.flatten_image()
        if len(image) > sample_size:
            sample_size = len(image)
        random.seed(seed)
        sample = random.sample(image, sample_size)

        self.kmeans_fit = KMeans(n_clusters=self.num_colors, \
                                        random_state=seed).fit(sample)

    def assign_kmeans_clusters(self):
        image = self.flatten_image()
        self.predictions = self.kmeans_fit.predict(image)

    def assign_colors_to_predictions(self):
        self.kmeans_colors = np.array([self.kmeans_fit.cluster_centers_[ix] \
                                                for ix in self.predictions])

    def reshape_color_predictions(self):
        w, h, colors = self.image.shape
        self.kmeans_image = np.array(self.kmeans_colors).reshape(w, h, colors)

    def make_kmeans_image(self):
        self.kmeans_palette_one_image()
        self.assign_kmeans_clusters()
        self.assign_colors_to_predictions()
        self.reshape_color_predictions()

    def plot_kmeans_image(self):
        plt.imshow(bgr_to_rgb(self.kmeans_image))
        plt.show()
