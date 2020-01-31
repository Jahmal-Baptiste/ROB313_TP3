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
    for k in range(-1, 181):
        r_table[k] = [(0, 0)]

    edges_pos  = np.argwhere(obj_mask != 0)
    for i, pos in enumerate(edges_pos):
        alpha = quantised_orient[tuple(pos)]
        r     = pos - ref_point
        r_table[alpha].append(r)
        #if alpha not in list(r_table.keys()):
        #    r_table[alpha] = [r]
        #else:
        #    r_table[alpha].append(r)
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

    # quantise the orientation to 180 direction (2 degree of step)
    quantised_grad_orient = quantise_grad_orientation(grad_orient, 2)

    # find the reference point of the obj
    ref_point = image_ref_point(grad_mask)

    # complete the R-Table
    r_table = complete_r_table(quantised_grad_orient, grad_mask, ref_point)

    return r_table


def compute_hough(image, image_grad_orient, image_grad_mask, r_table, channel=2):
    # compute the orientation of the gradient in degrees
    grad_orient = image_grad_orient*(180./np.pi)
    grad_orient = np.where(grad_orient < 0, grad_orient + 360, grad_orient)

    # quantise the orientation to 180 direction (step of 2 degrees)
    quantised_grad_orient = quantise_grad_orientation(grad_orient, 2)

    # compute the appearances of all the edge points of the image
    edges_pos              = np.nonzero(image_grad_mask)
    appearances            = -1*np.ones_like(image[:, :, channel]) # appearance of -1 for the points that are not edges
    appearances[edges_pos] = quantised_grad_orient[edges_pos]
    
    # create an array of the positions of the image points (initialised at (0, 0) for all points)
    positions    = np.empty(image[:, :, channel].shape[0]*image[:, :, channel].shape[1], dtype=tuple)
    positions[:] = [(0, 0)]
    positions    = np.reshape(positions, image[:, :, channel].shape)

    # filling of the positions of the edge points alone    
    #tuple_edges_pos      = [(edges_pos[0][k], edges_pos[1][k]) for k in range(edges_pos[0].shape[0])]
    t_edges_pos          = np.transpose(edges_pos)
    tuple_edges_pos      = np.empty(edges_pos[0].shape[0], dtype=tuple)
    tuple_edges_pos[:]   = [tuple(pos) for pos in t_edges_pos] #np.apply_along_axis(tuple, 0, t_edges_pos)...
    positions[edges_pos] = tuple_edges_pos

    # definition and vectorization of the vote function
    vote_func = np.vectorize(lambda pos : len(r_table[appearances[pos]]))

    # hough computation
    hough = vote_func(positions) #Takes a lot of time...

    ### ALTERNATE HOUGH COMPUTATION ###
    ### (MUCH MORE TIME-CONSUMING)  ###
    #for i, pos_i in enumerate(image_edges_pos):
    #    appearance_index = image_quantised_grad_orient[tuple(pos_i)]
    #    if appearance_index in list(r_table.keys()):
    #        for j, pos_j in enumerate(r_table[appearance_index]):
    #            pos = np.floor(pos_i + pos_j).astype(int)
    #            if pos[0]<0 or pos[0]>=image.shape[0] or pos[1]<0 or pos[1]>=image.shape[1]:
    #                continue
    #            hough[tuple(pos)] += 1
    
    return hough