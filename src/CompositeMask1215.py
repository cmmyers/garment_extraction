import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from sklearn.cluster import KMeans
import utils

path_to_opencv = '/Users/claremariemyers/Desktop/triage/opencv/'

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

def find_skin_palette(image_path, path_to_open_cv, num_points_to_check=4):
    image = plt.imread(image_path)

    face_colors = []

    face = find_face(image, path_to_open_cv)
    if face is not None:

        h, w, colors = face.shape
        face = np.array(face, dtype=np.float64) / 255
        for _ in range(num_points_to_check):
            col = np.random.randint(8,w-8)
            row = np.random.randint(8,h-8)
            color_block = face[row-4:row+4, col-4:col+4]

            ave_color = [avg_each_color(color_block, color) for color in ['r', 'g', 'b']]
            face_colors.append(ave_color)

    return face_colors

def find_face(image, path_to_opencv):
    '''
    uses OpenCV's face recognition tool to draw conclusions re: skin colors of models
    '''
    path = path_to_opencv + 'data/haarcascades/haarcascade_frontalface_default.xml'

    cascade = cv2.CascadeClassifier(path)



    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
    w, h = im_gray.shape
    min_size = (w/20, h/20)

    face = cascade.detectMultiScale(im_gray, 1.1, 5, minSize=min_size)

    if len(face) == 0:
            color_block = None
    else:
        for (x,y,w,h) in face:
            color_block = im_rgb[y:y+h, x:x+w]

    return color_block


class CompositeMask():
    def __init__(self, path, bg_threshold = 0.4, face_threshold = 0.1, \
                    num_bg_colors = 16, num_face_colors = 8):
        #resize image to 200x300
        self.load_and_resize_image(path)
        #sample from the background to get a background palette
        bg_pal = self.get_background_one_image(path, num_bg_colors)
        #change these colors to bgr
        self.bg_color_list = [[c[2], c[1], c[0]] for c in bg_pal]
        #set threshold for subtracting background colors
        self.bg_threshold = bg_threshold
        #make background masks
        self.make_many_masks_bg()
        #add all background masks together
        self.bg_composite_mask = sum(self.bg_masks)
        #lay mask over original image
        self.make_first_masked_image()

        #get the face palette
        self.face, face_pal = self.find_skin_palette(path, path_to_opencv,\
                                                    num_face_colors)
        #change the colors to bgr
        self.face_color_list = [[c[2], c[1], c[0]] for c in face_pal]
        #set the face threshold
        self.face_threshold = face_threshold
        self.make_many_masks_face()

        self.face_composite_mask = sum(self.face_masks)
        self.full_composite_mask = self.bg_composite_mask + \
                                            self.face_composite_mask

        self.get_contours()

    def load_and_resize_image(self, path):
        im = plt.imread(path)
        size = im.shape
        r = 200.0 / im.shape[1]
        dim = (200, int(im.shape[0] * r))
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        im = rgb_to_bgr(resized)
        self.image = im

    def get_background_one_image(self, path, num_points):
        background_colors = []
        im = plt.imread(path)
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

    def avg_each_color(self, color_block, color):
        color_dict = {'r':0, 'g':1, 'b':2}
        index = color_dict[color]
        all_colors = [color[index] for row in color_block for color in row]
        ave_color = sum(all_colors)/(len(all_colors) + 1)
        return ave_color

    def find_skin_palette(self, path, path_to_open_cv, num_points_to_check=4):
        image = plt.imread(path)

        face_colors = []

        face = find_face(image, path_to_open_cv)
        if face is not None:

            h, w, colors = face.shape
            face = np.array(face, dtype=np.float64) / 255
            for _ in range(num_points_to_check):
                col = np.random.randint(8,w-8)
                row = np.random.randint(8,h-8)
                color_block = face[row-4:row+4, col-4:col+4]

                ave_color = [avg_each_color(color_block, color) for color in ['r', 'g', 'b']]
                face_colors.append(ave_color)


        return face, face_colors

    def find_face(self, image, path_to_opencv):
        '''
        uses OpenCV's face recognition tool to draw conclusions re: skin colors of models
        '''
        path = path_to_opencv + 'data/haarcascades/haarcascade_frontalface_default.xml'

        cascade = cv2.CascadeClassifier(path)



        im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
        w, h = im_gray.shape
        min_size = (w/20, h/20)

        face = cascade.detectMultiScale(im_gray, 1.1, 5, minSize=min_size)

        if len(face) == 0:
                color_block = None
        else:
            for (x,y,w,h) in face:
                color_block = im_rgb[y:y+h, x:x+w]

        return color_block

    def make_mask_bg(self, three_colors):
        lower = np.array([(c - self.bg_threshold*c)*255 for c in three_colors])
        upper = np.array([(c + self.bg_threshold*c)*255 for c in three_colors])
        shapeMask = cv2.inRange(self.image, lower, upper)
        contoured_img, contours, hierarchy =                             cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contoured_img

    def make_mask_face(self, three_colors):
        lower = np.array([(c - self.face_threshold*c)*255  \
                                                    for c in three_colors])
        upper = np.array([(c + self.face_threshold*c)*255 \
                                                    for c in three_colors])
        shapeMask = cv2.inRange(self.first_masked_image, lower, upper)
        contoured_img, contours, hierarchy =   \
                                  cv2.findContours(shapeMask.copy(), \
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contoured_img

    def make_many_masks_bg(self):
        self.bg_masks = []
        for three_colors in self.bg_color_list:
            self.bg_masks.append(self.make_mask_bg(three_colors))

    def make_many_masks_face(self):
        self.face_masks = []
        for three_colors in self.face_color_list:
            self.face_masks.append(self.make_mask_face(three_colors))

    def plot_many_masks_bg(self):
        num_bg_colors = len(self.bg_masks)
        num_rows = int(num_bg_colors/4) + 1
        subplot_location = 0
        fig = plt.figure(figsize=(10, 12))
        for m in self.bg_masks:
            subplot_location += 1
            ax = fig.add_subplot(num_rows, 4, subplot_location)
            ax.imshow(m)
        plt.tight_layout()
        plt.show()

    def plot_many_masks_face(self):
        num_face_colors = len(self.face_masks)
        num_rows = int(num_face_colors/4) + 1
        subplot_location = 0
        fig = plt.figure(figsize=(10, 12))
        for m in self.face_masks:
            subplot_location += 1
            ax = fig.add_subplot(num_rows, 4, subplot_location)
            ax.imshow(m)
        plt.tight_layout()
        plt.show()



    def make_first_masked_image(self):
        orig_shape = self.image.shape
        new_image = []
        flat_mask = [p for row in self.bg_composite_mask for p in row]
        flat_image = [p for row in self.image for p in row]
        for m, p in zip(flat_mask, flat_image):
            if m != 0:
                new_image.append(np.array([57, 255, 20]))
            else: new_image.append(p)
        new_image = np.array(new_image)
        new_image = new_image.reshape(orig_shape)
        new_image = new_image.astype('uint8')
        new_image = bgr_to_rgb(new_image)
        self.first_masked_image = new_image

    def make_second_masked_image(self):
        orig_shape = self.image.shape
        new_image = []
        flat_mask = [p for row in self.face_composite_mask for p in row]
        flat_image = [p for row in self.first_masked_image for p in row]
        for m, p in zip(flat_mask, flat_image):
            if m != 0:
                new_image.append(np.array([57, 255, 20]))
            else: new_image.append(p)
        new_image = np.array(new_image)
        new_image = new_image.reshape(orig_shape)
        new_image = new_image.astype('uint8')
        new_image = bgr_to_rgb(new_image)
        self.second_masked_image = new_image

    def show_first_masked_image(self):
        im_rgb = self.first_masked_image
        plt.imshow(im_rgb)
        plt.show()

    def show_orig_image(self):
        im_rgb = bgr_to_rgb(self.image)
        plt.imshow(im_rgb)
        plt.show()

    def get_contours(self):
        lower = 0
        upper = 15
        shapeMask = cv2.inRange(self.full_composite_mask, lower, upper)
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
