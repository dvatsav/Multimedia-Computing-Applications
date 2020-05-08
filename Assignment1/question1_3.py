"""
* https://stackoverflow.com/questions/19672315/how-to-apply-box-filter-on-integral-image-surf
* https://www.vision.ee.ethz.ch/~surf/eccv06.pdf
* https://www.ipol.im/pub/art/2015/69/article.pdf
"""

import argparse
import os
import cv2
import ast
from tqdm import tqdm
from skimage.transform import integral_image
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs
import numpy as np
import json
from numpy import matlib as mb
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix_det

def box_filter(itgl_img, x, y, r, c):
    (h, w) = itgl_img.shape[:2]

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
    
        toplx = np.minimum(x, h-1).astype(int)
        toply = np.minimum(y, w-1).astype(int)
        botrx = np.minimum(x+r, h-1).astype(int)
        botry = np.minimum(y+c, w-1).astype(int)

        A_zero_indices = (np.where(toplx<0), np.where(toply<0))
        B_zero_indices = (np.where(toplx<0), np.where(botry<0))
        C_zero_indices = (np.where(botrx<0), np.where(toply<0))
        D_zero_indices = (np.where(botrx<0), np.where(botry<0))

        toplx = np.maximum(toplx, 0).astype(int)
        toply = np.maximum(toply, 0).astype(int)
        botrx = np.maximum(botrx, 0).astype(int)
        botry = np.maximum(botry, 0).astype(int)

        #print (toplx, toply, botrx, botry)

        A = itgl_img[toplx, toply]
        B = itgl_img[toplx, botry]
        C = itgl_img[botrx, toply]
        D = itgl_img[botrx, botry]

        A[A_zero_indices[0]] = 0
        A[A_zero_indices[1]] = 0
        B[B_zero_indices[0]] = 0
        B[B_zero_indices[1]] = 0
        C[C_zero_indices[0]] = 0
        C[C_zero_indices[1]] = 0
        D[D_zero_indices[0]] = 0
        D[D_zero_indices[1]] = 0
        
        return (A + D - B - C).astype(np.float64)

    
    else:

        toplx = int(min(x, h-1))
        toply = int(min(y, w-1))
        botrx = int(min(x+r, h-1))
        botry = int(min(y+x, w-1))
        
        A = B = C = D = 0


        if toplx >= 0 and toply >= 0:
            A = itgl_img[toplx, toply]
        if toplx >= 0 and botry >= 0:
            B = itgl_img[toplx, botry]
        if botrx >= 0 and toply >= 0:
            C = itgl_img[botrx, toply]
        if botrx >= 0 and botry >= 0:
            D = itgl_img[botrx, botry]
        

    
        return float(A + D - B - C)

def get_filter_map(itgl_img, h, w, step, filter_size):
    
    normalization = 1 / (filter_size * filter_size)

    x = np.linspace(0, h-1, h)
    y = np.linspace(0, w-1, w)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    rows = xv.ravel() * step
    cols = yv.ravel() * step

    filter_border = int((filter_size-1)/2 + 1)
    filter_lobe = filter_size/3

    # black region weight -2, white region weight 1

    dxx = - box_filter(itgl_img, rows - filter_lobe, cols - filter_lobe//2 - 1, filter_border, filter_lobe) * 3 \
        + box_filter(itgl_img, rows - filter_lobe + 1, cols - filter_border + 1, filter_border - 1, filter_size - 2) 
        # black region, white region
    dyy = - box_filter(itgl_img, rows - filter_lobe//2 - 1, cols - filter_lobe, filter_lobe, filter_border) * 3 \
        + box_filter(itgl_img, rows - filter_border + 1, cols - filter_lobe + 1, filter_size - 2, filter_border - 1) 
        # black region, white region
    dxy = - box_filter(itgl_img, rows - filter_lobe - 1, cols, filter_lobe, filter_lobe) \
        - box_filter(itgl_img, rows, cols - filter_lobe - 1, filter_lobe, filter_lobe) \
        + box_filter(itgl_img, rows - filter_lobe - 1, cols - filter_lobe - 1, filter_lobe, filter_lobe) \
        + box_filter(itgl_img, rows, cols, filter_lobe, filter_lobe)
        # Right top, left bottom, left top, right bottom
    
    dxx = dxx.astype(np.float64) * normalization
    dyy = dyy.astype(np.float64) * normalization
    dxy = dxy.astype(np.float64) * normalization
    
    output_filter = np.multiply(dxx, dyy) - np.multiply(dxy, dxy) * 0.81
    output_laplace_sign = np.ones(output_filter.shape)
    dxxdyy = np.add(dxx, dyy)
    output_laplace_sign[np.where(dxxdyy < 0)] = -1
    
    return output_filter.reshape(h, w) 

def find_extrema(filtered_imgs, img):
    lmax = peak_local_max(filtered_imgs, threshold_abs=0.0, threshold_rel=0.1, footprint=np.ones((3,) * (img.ndim + 1)))

    return lmax

def haar_wavelet(itgl_img, row, col, size, axis):
    if axis == 0:
        return box_filter(itgl_img, row-size/2, col, size, size/2) - box_filter(itgl_img, row-size/2, col-size/2, size, size/2)
    else:
        return box_filter(itgl_img, row, col-size/2, size/2, size) - box_filter(itgl_img, row-size/2, col-size/2, size/2, size)

def get_gaussian_weight(x, y, sigma):
    return (1/(2*np.pi*sigma**2)) * \
        np.exp((-x**2 + y**2) / (2 * sigma**2))

def get_orientation(xc, yc, s, itgl_img):
    results_x = []
    results_y = []
    maximum = np.inf
    orientation = 0
    
    xi = np.linspace(-6*s, 6*s, 13)
    yi = np.linspace(-6*s, 6*s, 13)

    yi, xi = np.meshgrid(xi, yi)
    xi = xi.ravel()
    yi = yi.ravel()
    #print (x)
    #print (y)
    indx_keep = np.where(xi**2 + yi**2 < 36*s**2)[0]

    xi = xi[indx_keep]
    yi = yi[indx_keep]


    gw = get_gaussian_weight(xi, yi, 2*s)

    results_xs = gw*haar_wavelet(itgl_img, xi, yi, 4*s, 0)
    results_ys = gw*haar_wavelet(itgl_img, xi, yi, 4*s, 1)
    
    """
    for x in range(-6*s, 6*s+1, s):
        for y in range(-6*s, 6*s+1, s):
            if x**2 + y**2 < 36*s**2:
                gw = get_gaussian_weight(x, y, 2*s)
                results_x.append(gw*haar_wavelet(itgl_img, x, y, 4*s, 0))
                results_y.append(gw*haar_wavelet(itgl_img, x, y, 4*s, 1))
    """
    ang = 0
    while ang < 2*np.pi:
        sum_x = sum_y = 0

        for i in range(len(results_x)):
            angle = 0
            if results_x[i] - 0 < 0.002:
                angle = np.pi/2
            else:
                angle = np.arctan(results_y[i]/results_x[i])

            if ang < angle and angle < ang + np.pi/3:
                sum_x += results_x[i]
                sum_y += results_y[i]
    
        
        if (sum_x*sum_x + sum_y > maximum):
            maximum = sum_x*sum_x + sum_y*sum_y
            orientation = np.arctan(sum_y/sum_x)
        ang += 0.1
    return orientation


def get_descriptor(blobs, itgl_img):
    
    blobs = blobs.astype(np.int)
    scales = blobs[:, 2:].ravel()
    xs = blobs[:, 0:1].ravel()
    ys = blobs[:, 1:2].ravel()

    descriptor = np.zeros((xs.shape[0], 64))
    for _ in range(xs.shape[0]):
        x = xs[_]
        y = ys[_]
        s = scales[_]
        angle = get_orientation(x, y, s, itgl_img)
        vector = []
        for i in range(0, 20*s, 5*s):
            for j in range(0, 20*s, 5*s):
                dx = dy = adx = ady = 0

                # Choice of kernel with sigma 3.3s taken from https://www.ipol.im/pub/art/2015/69/article.pdf
                for k in range(i, i+5*s, s):
                    for l in range(j, j+5*s, s):
                        kr = k*np.cos(angle) - l*np.sin(angle)
                        lr = l*np.cos(angle) + k*np.sin(angle)
                        gw = get_gaussian_weight(kr-10*s, lr-10*s, 3.3*s)
                        haar_x = gw*haar_wavelet(itgl_img, kr, lr, 2*s, 0)
                        haar_y = gw*haar_wavelet(itgl_img, kr, lr, 2*s, 1)

                        dx += haar_x
                        dy += haar_y
                        adx += np.abs(haar_x)
                        ady += np.abs(haar_y)
                vector += [dx, dy, adx, ady]
        descriptor[_] = np.array(vector)
    return descriptor
    """
    window_sizes = 20 * scales
    subregion_sizes = 5 * scales
    descriptor = np.zeros((xs.shape[0], 64))
    for i in range(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        s = scales[i]

        ssx = np.linspace(-10*s+1, 10*s, 20).astype(int)
        ssy = np.linspace(-10*s+1, 10*s, 20).astype(int)

        ssy, ssx = np.meshgrid(ssx, ssy)
        shp = ssx.shape
        ssy = ssy.ravel()
        ssx = ssx.ravel()

        haar_x = haar_wavelet(itgl_img, x+ssx, x+ssy, 2*s, 0).reshape(shp[0], shp[1])
        haar_y = haar_wavelet(itgl_img, x+ssx, y+ssy, 2*s, 1).reshape(shp[0], shp[1])
        
        vector = []

        indices = [
            (0, 5, 0, 5), (0, 5, 5, 10), (0, 5, 10, 15), (0, 5, 15, 20),
            (5, 10, 0, 5), (5, 10, 5, 10), (5, 10, 10, 15), (5, 10, 15, 20),
            (10, 15, 0, 5), (10, 15, 5, 10), (10, 15, 10, 15), (10, 15, 15, 20),
            (15, 20, 0, 5), (15, 20, 5, 10), (15, 20, 10, 15), (15, 20, 15, 20),
        ]
        
        for idx in indices:
            dx = np.sum(haar_x[idx[0]:idx[1], idx[2]:idx[3]])
            dy = np.sum(haar_y[idx[0]:idx[1], idx[2]:idx[3]])
            adx = np.sum(np.abs(haar_x[idx[0]:idx[1], idx[2]:idx[3]]))
            ady = np.sum(np.abs(haar_y[idx[0]:idx[1], idx[2]:idx[3]]))
            
            vector += [dx, dy, adx, ady]
        descriptor[i] = np.array(vector)
    print (descriptor.shape)
    print (descriptor)
    """ 

    
def surf(img, is_test=False):
    itgl_img = integral_image(img)
    threshold = 0.0002
    num_octaves = 4
    octaves = np.array([
        [9, 15, 21, 27],
        [15, 27, 39, 51],
        [27, 51, 75, 99],
        [51, 99, 147, 195]
    ])
    filter_maps = [0] * 16
    (h, w) = itgl_img.shape[:2]
    step = 1
    sigma_list = np.zeros(octaves.shape[0] * octaves.shape[1])
    for i in range(octaves.shape[0]):
        for j in range(octaves.shape[1]):
            cur_filter = octaves[i, j]
            sigma_list[i * octaves.shape[1] + j] = cur_filter * (1.2/9)
            filter_maps[i*octaves.shape[1] + j] = hessian_matrix_det(img.astype(np.float64), sigma_list[i * octaves.shape[1] + j])

            
    filtered_imgs = np.stack(filter_maps, axis=-1)
    extremas = find_extrema(filtered_imgs, img)

    
    em = extremas.astype(np.float64)
    sigmas_of_peaks = sigma_list[extremas[:, -1]]
    sigmas_of_peaks = np.expand_dims(sigmas_of_peaks, axis=1)
    em = np.hstack([em[:, :-1], sigmas_of_peaks])
    overlap = 0.2
    blobs = _prune_blobs(em, overlap)
    # print (blobs.shape)
    if is_test:
        fig, ax = plt.subplots()
        nh,nw = img.shape
        count = 0
        ax.imshow(img, interpolation='nearest', cmap="gray")
        for blob in blobs:
            y,x,r = blob
            c = plt.Circle((x, y), r*1.414, color='red', linewidth=1.5, fill=False)
            ax.add_patch(c)
        ax.plot()  
        plt.show()
    #descriptor = get_descriptor(blobs, itgl_img)

    return blobs

def train(image_folder, save_folder):
    
    image_list = os.listdir(image_folder)
    keypoints = {}
    savefile_name = os.path.join(save_folder, "Surf_blobs.json")
    for img_filename in tqdm(image_list):
        #savefile_name = os.path.join(save_folder, img_filename[:-4]+".npy")

        img_path = os.path.join(image_folder, img_filename)

        img = cv2.imread(img_path, 0)
        scale_percent = 10
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        blobs = surf(img)

        points = []
        for blob in blobs:
            points.append({"coords":(blob[0], blob[1]), "radius":blob[2]*1.414})
        
        keypoints[img_filename] = points
        #print (keypoints)
        #print (descriptor)

        # with open(savefile_name, 'wb') as f:
        #     np.save(f, descriptor)
    with open(savefile_name, 'w') as f:
        json.dump(keypoints, f)

def test(query_folder, image_folder):
    query_list = os.listdir(query_folder)
    for query_file in query_list:
        query_file_path = os.path.join(query_folder, query_file)
        with open(query_file_path, 'r') as f:
            contents = f.readline().split(" ")
            img_filename, x, y, width, height = [contents[0][contents[0].find("_")+1:]+".jpg"] + list(map(lambda x : ast.literal_eval(x), contents[1:]))
        img_path = os.path.join(image_folder, img_filename)
        
        img = cv2.imread(img_path, 0)
        scale_percent = 30
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        surf(img, is_test=True)

def sanity_check():
    img = cv2.imread("images/all_souls_000000.jpg", 0)
    itgl_img = integral_image(img)
    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    surf(img, is_test=True)
    

def main():
    def check_mode(value):    
        if value != "train" and value != "test":
            raise argparse.ArgumentTypeError("%s is an invalid mode. Valid modes are train/test" % value)
        return value

    def check_path(value):
        if not os.path.exists(value):
            raise argparse.ArgumentTypeError("%s directory does not exist" % value)
        return value

    parser = argparse.ArgumentParser()
    
    parser.add_argument("mode",
                        help="train/test",
                        type=check_mode)

    parser.add_argument("image_folder",
                        help="Folder containing all images",
                        type=check_path)

    parser.add_argument("--feature_folder",
                        help="Folder to store generated autocorrelograms",
                        type=check_path)

    parser.add_argument("--query_folder",
                        help="Folder containing txt files with queries",
                        type=check_path)
    args = parser.parse_args()
    mode = args.mode

    query_folder = args.query_folder
    image_folder = args.image_folder
    save_folder = args.feature_folder

    if mode == "train":
        train(image_folder, save_folder)
        #sanity_check()
    else:
        test(query_folder, image_folder)

    

if __name__ == '__main__':
    main()