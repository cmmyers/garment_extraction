import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import utils

path_to_opencv = '../../opencv'

def rgb_to_bgr(image):
    r, g, b = cv2.split(image)
    return cv2.merge((b, g, r))

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge((r, g, b))


class CompositeMask():
    def __init__(self, filename, bg_threshold = 0.4, num_bg_colors = 16, \
                                test_ims_dir = 'data/test/', \
                                seg_ims_dir = 'data/segmented/'):

        self.test_im_path = test_ims_dir + filename
        self.seg_im_path = seg_ims_dir + filename[:-4] + '-seg.jpg'
        #resize images to 200x300
        self.image = self.load_and_resize_image(self.test_im_path)
        self.seg_image = self.load_and_resize_image(self.seg_im_path)
        #sample from the background to get a background palette
        bg_pal = self.get_background_colors(num_bg_colors)
        #change these colors to decimal representation
        self.bg_color_list = [[c[0]/255, c[1]/255, c[2]/255] for c in bg_pal]
        self.bg_color_list_rgb = [c[::-1] for c in self.bg_color_list]
        #set threshold for subtracting background colors
        self.bg_threshold = bg_threshold
        #make background masks
        self.make_many_masks_bg()
        #add all background masks together
        self.bg_composite_mask = sum(self.bg_masks)
        #lay mask over original image
        self.make_first_masked_image()

    def load_and_resize_image(self, path):
        im = plt.imread(path)
        size = im.shape
        self.ratio = 200.0 / im.shape[1]
        dim = (200, int(im.shape[0] * self.ratio))
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        im = rgb_to_bgr(resized)
        return im

    def avg_each_color(self, color_block, color_name):
        color_dict = {'red':0, 'green':1, 'blue':2}
        index = color_dict[color_name]
        all_colors = [color[index] for row in color_block for color in row]
        ave_color = sum(all_colors)/(len(all_colors) + .001)
        return ave_color

    def get_background_colors(self, num_points):
        background_colors = []
        h, w, colors = self.image.shape
        points_per_side = int(num_points/2)

        self.left_coords = [(np.random.choice(h), \
                            np.random.choice(range(4, int(w/8)))) \
                            for i in range(points_per_side)]

        self.right_coords = [(np.random.choice(h), \
                            np.random.choice(range(int(7*w/8), w-4))) \
                            for i in range(points_per_side)]

        for row, col in self.left_coords + self.right_coords:
            color_block = self.image[int(row-4):int(row+4), \
                                    int(col-4):int(col+4)]
            ave_color = [self.avg_each_color(color_block, color) for color in ['red', 'green', 'blue']]
            background_colors.append(ave_color)

        return background_colors

    def make_mask_bg(self, three_colors):
        lower = np.array([(c - self.bg_threshold*c)*255 for c in three_colors])
        upper = np.array([(c + self.bg_threshold*c)*255 for c in three_colors])
        shapeMask = cv2.inRange(self.image, lower, upper)
        contoured_img, contours, hierarchy =                             cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contoured_img

    def make_many_masks_bg(self):
        self.bg_masks = []
        for three_colors in self.bg_color_list:
            self.bg_masks.append(self.make_mask_bg(three_colors))

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

    def make_reversed_first_masked_image(self):
        orig_shape = self.image.shape
        new_image = []
        flat_mask = [p for row in self.bg_composite_mask for p in row]
        flat_image = [p for row in self.image for p in row]
        for m, p in zip(flat_mask, flat_image):
            if m == 0:
                new_image.append(np.array([255, 57, 20]))
            else: new_image.append(p)
        new_image = np.array(new_image)
        new_image = new_image.reshape(orig_shape)
        new_image = new_image.astype('uint8')
        new_image = bgr_to_rgb(new_image)
        self.reversed_first_masked_image = new_image

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



    def find_skin_palette(self, path, path_to_open_cv, num_points_to_check=4):
        image = plt.imread(path)

        face_colors = []

        face = self.find_face(image, path_to_open_cv)
        if face is not None:

            h, w, colors = face.shape
            face = np.array(face, dtype=np.float64) / 255
            for _ in range(num_points_to_check):
                col = np.random.randint(8,w-8)
                row = np.random.randint(8,h-8)
                color_block = face[row-4:row+4, col-4:col+4]

                ave_color = [self.avg_each_color(color_block, color) for color in ['red', 'green', 'blue']]
                face_colors.append(ave_color)


        return face, face_colors

    def find_ratio_face_to_all_non_bg(self):
        #ratio of face to size to pixel count of skin, garment, and accessories
        face_size = self.face.shape[:2]
        scaled_face_size = np.prod(face_size)*1. * self.ratio**2
        body_pixels = self.make_boolean_mask_from_segs(['accessories', 'skin', 'garment'])
        return scaled_face_size / np.count_nonzero(body_pixels)

    def remove_bg_colors_from_face_pal(self):
        bg_colors = []
        for i, c1 in enumerate(self.face_color_list1):
            for j, c2 in enumerate(self.bg_color_list):
                e = euclidean_distances(np.array(c1).reshape(1, -1), np.array(c2).reshape(1, -1))
                if e < 0.2:
                    bg_colors.append(c1)
                break
        return [c for c in self.face_color_list1 if c not in bg_colors]


    def find_ratio_face_to_garment(self):
        #ratio of face to size to pixel count of garment
        face_size = self.face.shape[:2]
        scaled_face_size = np.prod(face_size)*1. * self.ratio**2
        body_pixels = self.make_boolean_mask_from_segs(['garment'])
        return scaled_face_size / np.count_nonzero(body_pixels)


    def make_boolean_mask_from_segs(self, list_of_segments_to_return_true):
        masks = []
        for seg in list_of_segments_to_return_true:
            low_bg = np.array(self.segments_dict[seg][0])
            high_bg = np.array(self.segments_dict[seg][1])
            masks.append(cv2.inRange(self.seg_image, low_bg, high_bg))
        return sum(masks) * -1


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
                cv2.drawContours(temp_im, [c], 0, (255, 255, 0), -1)
                self.im_largest_contoured = temp_im
                plt.imshow(bgr_to_rgb(temp_im))
                break

        plt.show()

    def make_boolean_mask_from_contours(self):
        masks = []
        low_bg = np.array([250, 250, 0])
        high_bg = np.array([255, 255, 5])
        self.contour_mask = cv2.inRange(self.im_largest_contoured, low_bg, high_bg)

    def make_contour_masked_image(self):
        orig_shape = self.image.shape
        new_image = []
        flat_mask = [p for row in self.contour_mask for p in row]
        flat_image = [p for row in self.image for p in row]
        for m, p in zip(flat_mask, flat_image):
            if m == 0:
                new_image.append(np.array([57, 255, 20]))
            else: new_image.append(p)
        new_image = np.array(new_image)
        new_image = new_image.reshape(orig_shape)
        new_image = new_image.astype('uint8')
        new_image = bgr_to_rgb(new_image)
        self.contour_masked_image = new_image

if __name__ == '__main__':
    pass
