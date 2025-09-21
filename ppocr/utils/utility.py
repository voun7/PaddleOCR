import os

import cv2
import numpy as np


def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)

    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def binarize_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img
