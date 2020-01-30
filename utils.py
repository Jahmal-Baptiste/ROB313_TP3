import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict


def define_ROI(event, x, y, flags, param):
    global r, c, w, h, ROI_DEFINED
    # if the left mouse button was clicked,
    # record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r = x
        c = y
        ROI_DEFINED = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        w = abs(r2 - r)
        h = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        ROI_DEFINED = True


def image_gradient(img, k=0):
    """
    calculate the gradient of an hsv_image in a kernel k.
    :return:
    """
    Dx = cv2.Sobel(img[:, :, k], cv2.CV_64F, 1, 0)
    Dy = cv2.Sobel(img[:, :, k], cv2.CV_64F, 0, 1)
    return Dx, Dy


def gradient_module(Dx, Dy):
    """
    Calculate the module of the gradient.
    :param Dx: The gradient on x direction.
    :param Dy: The gradient on y direction.
    :return:
    """
    return np.sqrt(Dx**2 + Dy**2)


def gradient_orientation(Dx, Dy):
    """
    Calculate the orientation of the gradient in degree.
    :param Dx:
    :param Dy:
    :return:
    """
    orientation = np.arctan2(Dy, Dx)
    orientation *= (180/np.pi)
    orientation = np.where(orientation < 0, orientation + 360, orientation)
    return orientation


def quantise_grad_orientation(orientation, angle_step=2):
    """
    Quantise the gradients orientation(in degree) with the angle step(in degree).
    :param orientation:
    :param angle_step:
    :return:
    """
    return np.floor(orientation/angle_step)


#def edges_image(img, k=2):
#    """
#    Find the edge of an image.
#    :param img:
#    :param k:
#    :return:
#    """
#    return cv2.Canny(img[:, :, k], 100, 200)


def image_ref_point(edges_img):
    """
    Find the reference point of an image.
    :param edges_img:
    :return:
    """
    edges_pos = np.argwhere(edges_img != 0)
    return np.mean(edges_pos, axis=0)


def complete_r_table(quantised_orient, obj_mask, ref_point):
    """
    Complete the R-Table of an object.
    :param quantised_orient:
    :param edges_img:
    :param ref_point:
    :return:
    """
    orient_num = np.unique(quantised_orient).shape[0]
    r_table    = OrderedDict()
    r_table[-1] = (0, 0)
    edges_pos  = np.argwhere(obj_mask != 0)
    for i, pos in enumerate(edges_pos):
        alpha = quantised_orient[tuple(pos)]
        r     = pos - ref_point
        if alpha not in list(r_table.keys()):
            r_table[alpha] = [r]
        else:
            r_table[alpha].append(r)
    return r_table


def construct_RTable(obj, gradient_threshold, channel=2):
    """
    Construct the R-Table of an hsv_object in a given channel.
    :param obj:
    :param k:
    :return:
    """
    # calculate the gradient of the obj at v level
    Dx, Dy = image_gradient(obj, channel)

    # calculate the gradient module
    grad_module = gradient_module(Dx, Dy)
    cv2.normalize(grad_module, grad_module, 0, 255, cv2.NORM_MINMAX)

    # calculate the mask of the image wrt the gradient module threshold
    grad_mask = cv2.inRange(grad_module, np.array([gradient_threshold]), np.array([255.]))

    # normalize the mask for future computation
    cv2.normalize(grad_mask, grad_mask, 0, 1, cv2.NORM_MINMAX)

    # calculate the orientation of the gradient in degree
    grad_orient = gradient_orientation(Dx, Dy)

    # find the edge of the obj
    #edges_img = edges_image(obj, k)

    # quantise the orientation to 180 direction (2 degree of step)
    quantised_grad_orient = quantise_grad_orientation(grad_orient, 2)

    # find the reference point of the obj
    ref_point = image_ref_point(grad_mask)

    # complete the R-Table
    r_table = complete_r_table(quantised_grad_orient, grad_mask, ref_point)

    return r_table


def compute_hough(image, grad_orient, image_grad_mask, obj, r_table, channel=2):
    hough, appearances = np.zeros_like(image[:, :, channel]), -1*np.ones_like(image[:, :, channel])
    positions = np.empty(image[:, :, channel].shape, dtype=object)

    # calculate the orientation of the gradient in degree
    image_grad_orient = grad_orient*(180/np.pi)
    image_grad_orient = np.where(image_grad_orient < 0, image_grad_orient + 360, image_grad_orient)

    # quantise the orientation to 180 direction (2 degree of step)
    image_quantised_grad_orient = quantise_grad_orientation(image_grad_orient, 2)

    image_edges_pos = np.nonzero(image_grad_mask)
    appearances[image_edges_pos[0], image_edges_pos[1]] = image_quantised_grad_orient[image_edges_pos[0], image_edges_pos[1]]
    positions[image_edges_pos[0], image_edges_pos[1]]   = 0 #image_edges_pos[0], image_edges_pos[1]

    def vote_func(pos):
        return len(r_table[appearances[pos]])
        
    vvote_func = np.vectorize(vote_func)

    test_list  = [(0, 0), (0, 1)]
    test_array = np.empty((2, 2), dtype=object)
    test_array[(0, 0), (0, 1)] = (0, 0)
    #hough = vvote_func(test_array)


    #for i, pos_i in enumerate(image_edges_pos):
    #    appearance_index = image_quantised_grad_orient[tuple(pos_i)]
    #    if appearance_index in list(r_table.keys()):
    #        for j, pos_j in enumerate(r_table[appearance_index]):
    #            pos = np.floor(pos_i + pos_j).astype(int)
    #            if pos[0]<0 or pos[0]>=image.shape[0] or pos[1]<0 or pos[1]>=image.shape[1]:
    #                continue
    #            hough[tuple(pos)] += 1
    
    return hough