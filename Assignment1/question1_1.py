"""
* Deepak Srivatsav
* IIIT-Delhi
* Reference - http://www.cs.cornell.edu/~rdz/Papers/Huang-CVPR97.pdf
"""

import argparse
import os
import sys
import ast
import pickle
from operator import add, truediv
import time

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from tqdm import tqdm


def get_correlogram_distance(a1, a2):
    num_colors = a1.shape[0]
    d = a1.shape[1]
    assert a1.shape[0] == a2.shape[0]
    assert a1.shape[1] == a2.shape[1]
    distance = 0
    for color_ind in range(num_colors):
        for k in range(d):
            distance += (np.abs(a1[color_ind][k] - a2[color_ind][k]) / (1 + a1[color_ind][k] + a2[color_ind][k]))
    return np.abs(distance)

def gen_rgb_cube():
    cube_dimension = 256
    rgb_cube = np.ndarray((cube_dimension, cube_dimension, cube_dimension, 3),
                            dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            for k in range(256):
                position = color = (i, j, k)
                rgb_cube[position] = color
    rgb_cube = np.reshape(rgb_cube, (4096, 4096, 3))
    return rgb_cube

def get_cluster_model(num_color_clusters, cluster_model):
    print ("[*] Checking color quantization paths")
    if os.path.exists(cluster_model):
        with open(cluster_model, 'rb') as f:
            clt = pickle.load(f)
    else:
        print ("[*] Generating quantizations")
        rgb_cube = gen_rgb_cube()
        print ("[*] Generated RGB Cube")
        (h, w) = rgb_cube.shape[:2]
        img = cv2.cvtColor(rgb_cube, cv2.COLOR_BGR2LAB)
        img = img.reshape(-1, img.shape[-1])
        clt = MiniBatchKMeans(n_clusters=num_color_clusters)
        clt.fit(img)
        clt.cluster_centers_ = np.expand_dims(clt.cluster_centers_, axis=0)
        clt.cluster_centers_ = clt.cluster_centers_.astype("uint8")
        clt.cluster_centers_ = cv2.cvtColor(clt.cluster_centers_, cv2.COLOR_LAB2BGR)
        clt.cluster_centers_ = clt.cluster_centers_.reshape(-1, clt.cluster_centers_.shape[-1])

        with open(cluster_model, 'wb') as f:
            pickle.dump(clt, f)
    print ("[*] Got color quantizations")
    return clt


"""
* Perform color quantization to reduce number of unique colors, since RGB space can have
* up to 256*256*256 unique colors. Furthermore, the referenced paper states that m, the number
* of colors is deemed to be a constant. So preferably, we would want to reduce the number of 
* colors to a value such as 64.
"""
def quantize_image(img, clt):
    (h, w) = img.shape[:2]

    img = img.reshape(-1, img.shape[-1])
    
    labels = clt.predict(img)

    img = clt.cluster_centers_.astype("uint8")[labels]
    img = img.reshape((h, w, 3))


    return img


def get_neighbours(x, y, w, h, k):
    
    def check_valid(x, y, w, h):
        if 0 <= x < h and 0 <= y < w:
            return True
        else:
            return False

    neighbours = []
    left_corner = (x-k, y-k)
    if check_valid(left_corner[0], left_corner[1], w, h):
        neighbours.append(left_corner)
    for i in range(1, 2*k+1, 1):
        point = (left_corner[0]+i, left_corner[1])
        if check_valid(point[0], point[1], w, h):
            neighbours.append(point)
        point = (left_corner[0], left_corner[1]+i)
        if check_valid(point[0], point[1], w, h):
            neighbours.append(point)
    
    right_corner = (x+k, y+k)
    if check_valid(right_corner[0], right_corner[1], w, h):
        neighbours.append(right_corner)
    for i in range(1, 2*k, 1):
        point = (right_corner[0]-i, right_corner[1])
        if check_valid(point[0], point[1], w, h):
            neighbours.append(point)
        point = (right_corner[0], right_corner[1]-i)
        if check_valid(point[0], point[1], w, h):
            neighbours.append(point)

    return neighbours

"""
* Used a set of small values for distance, in order to implement the dynamic programming algorithm as explained
* in the reference paper
"""

def dp_method(img, num_colors, colors, d, num_colors_img, cluster_centers, color_count, h, w):
    autocorrelogram = np.zeros((num_colors, len(d)))
    color_neighbour_count = np.zeros(num_colors)

    inverse_index_map = {}

    for i in range(num_colors):
        inverse_index_map[tuple(cluster_centers[i])] = i
        if cluster_centers[i] in colors:
            autocorrelogram[i][0] = 1
    dlimit = max(d)
    horizontal_color_count = np.zeros((h, w, dlimit+1))
    vertical_color_count = np.zeros((h, w, dlimit+1))


    for k in range(dlimit+1):
        for x in range(h):
            for y in range(w):
                
                if k == 0:
                    horizontal_color_count[x][y][k] = 1
                    vertical_color_count[x][y][k] = 1
                else:
                    horizontal_color_count[x][y][k] = horizontal_color_count[x][y][k-1]
                    if y + k < w and np.array_equal(img[x][y], img[x][y+k]):
                        horizontal_color_count[x][y][k] += 1
                    vertical_color_count[x][y][k] = vertical_color_count[x][y][k-1]
                    if x + k < h and np.array_equal(img[x][y], img[x+k][y]):
                        vertical_color_count[x+k][y][k] += 1               

    for ki in range(len(d)):
        k = d[ki]
        for x in range(h):
            for y in range(w):
                color_ind = inverse_index_map[tuple(img[x][y])]
                if y-k >= 0:
                    if x+k < h:
                        autocorrelogram[color_ind][ki] += horizontal_color_count[x+k][y-k][min(2*k, dlimit)]
                     
                    if x-k >= 0:
                        autocorrelogram[color_ind][ki] += horizontal_color_count[x-k][y-k][min(2*k, dlimit)]

                    if x-k+1 >= 0 and x-k+1 < h:
                        autocorrelogram[color_ind][ki] += vertical_color_count[x-k+1][y-k][min(2*k-2, dlimit)]

                if y+k < w and x-k+1 >= 0 and x-k+1 < h:
                    autocorrelogram[color_ind][ki] += vertical_color_count[x-k+1][y+k][min(2*k-2, dlimit)]

    
        for color_ind in range(num_colors_img):
            autocorrelogram[color_ind][ki] /= (8*k*color_count[color_ind])

    return autocorrelogram

def nodp_method(img, num_colors, colors, d, num_colors_img, cluster_centers, color_count, h, w):
    
    autocorrelogram = np.zeros((num_colors, len(d)))
    color_neighbour_count = np.zeros(num_colors)

    inverse_index_map = {}
     
    
    for i in range(num_colors):
        inverse_index_map[tuple(cluster_centers[i])] = i

    for ki in range(len(d)):
        k = d[ki]
        for x in range(h):
            for y in range(w):
                color_ind = inverse_index_map[tuple(img[x][y])]
                neighbours = get_neighbours(x, y, w, h, k)
                
                for neighbour in neighbours:
                    cur = img[neighbour]
                    
                    if np.array_equal(cur, img[x, y]):
                        autocorrelogram[color_ind][ki] += 1
                color_neighbour_count[color_ind] += len(neighbours)
    
        for color_ind in range(num_colors):
            if color_neighbour_count[color_ind] > 0:
                autocorrelogram[color_ind][ki] /= (color_neighbour_count[color_ind])
            else:
                autocorrelogram[color_ind][ki] = 0
    return autocorrelogram


def generate_autocorrelogram(img, d, clt, use_dp=False):

    (h, w) = img.shape[:2]
    img = quantize_image(img, clt)
    colors, color_count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)

    cluster_centers = clt.cluster_centers_.astype("uint8")
    num_colors = cluster_centers.shape[0]
    num_colors_img = colors.shape[0]

    if num_colors != 64:
        print (num_colors)
        sys.exit()
    
    assert max(d) < h and max(d) < w

    if use_dp:
        autocorrelogram = dp_method(img, num_colors, colors, d, num_colors_img, cluster_centers, color_count, h, w)
    else:
        autocorrelogram = nodp_method(img, num_colors, colors, d, num_colors_img, cluster_centers, color_count, h, w)
    
                 
    return autocorrelogram

def train(image_folder, save_folder, cluster_model):
    num_color_clusters = 64
    d = [1, 3, 5, 7]
    cluster_model = get_cluster_model(num_color_clusters, cluster_model)

    image_list = os.listdir(image_folder)
    for img_filename in tqdm(image_list):
        savefile_name = os.path.join(save_folder, img_filename[:-4]+".npy")

        img_path = os.path.join(image_folder, img_filename)

        img = cv2.imread(img_path)
        scale_percent = 15
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        autocorrelogram = generate_autocorrelogram(img, d=d, clt=cluster_model, use_dp=True)

        with open(savefile_name, 'wb') as f:
            np.save(f, autocorrelogram)


def test(query_folder, image_folder, save_folder, cluster_model, gt_folder):
        
    query_list = os.listdir(query_folder)
    num_imgs_retrieved = [5, 10, 50, 100, 200, 400]

    avg_good = [0] * len(num_imgs_retrieved)
    avg_ok = [0] * len(num_imgs_retrieved)
    avg_junk = [0] * len(num_imgs_retrieved)
    
    max_precision = [-np.inf] * len(num_imgs_retrieved)
    min_precision = [np.inf] * len(num_imgs_retrieved)
    max_recall = [-np.inf] * len(num_imgs_retrieved)
    min_recall = [np.inf] * len(num_imgs_retrieved)
    max_f1 = [-np.inf] * len(num_imgs_retrieved)
    min_f1 = [np.inf] * len(num_imgs_retrieved)
    avg_recall = [0] * len(num_imgs_retrieved)
    avg_precision = [0] * len(num_imgs_retrieved)
    avg_f1 = [0] * len(num_imgs_retrieved)
    avg_k = 0
    mAP = 0
    avg_time = 0
    
    for indx, query_file in enumerate(query_list, start=1):
        query_file_path = os.path.join(query_folder, query_file)
        with open(query_file_path, 'r') as f:
            contents = f.readline().split(" ")
            img_filename, x, y, width, height = [contents[0][contents[0].find("_")+1:]+".jpg"] + \
                                                list(map(lambda x : ast.literal_eval(x), contents[1:]))
        img_path = os.path.join(image_folder, img_filename)
        
        gt_files = [os.path.join(gt_folder, query_file.replace("query", "good")), os.path.join(gt_folder, query_file.replace("query", "ok")), os.path.join(gt_folder, query_file.replace("query", "junk"))]
        filenames = []
        good = []
        ok = []
        junk = []
        for gt_file in gt_files:
            with open(gt_file, 'r') as f:
                if "good" in gt_file:
                    good = [line.rstrip() for line in f]
                    filenames += good 
                elif "ok" in gt_file:
                    ok = [line.rstrip() for line in f]
                    filenames += ok
                else:
                    junk = [line.rstrip() for line in f]
                    filenames += junk

        savefile_name = os.path.join(save_folder, img_filename[:-4]+".npy")
        if not os.path.exists(savefile_name):
            print ("[*] Generating features for query image")
            d = [1, 3, 5, 7]
            img = cv2.imread(img_path)
            scale_percent = 10
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            autocorrelogram = generate_autocorrelogram(img, d=d, clt=cluster_model)

        else:
            print ("[*] Retrieved features for query image")
            autocorrelogram = np.load(savefile_name)

        """
        * Calculate distances against images in database
        """

        feature_file_list = os.listdir(save_folder)
        distances = np.zeros(len(feature_file_list), dtype=np.float)
        ret_time = time.time()
        for idx, feature_file in enumerate(feature_file_list):
            feature_file_path = os.path.join(save_folder, feature_file)
            autocorrelogram_2 = np.load(feature_file_path) 
            dist = get_correlogram_distance(autocorrelogram, autocorrelogram_2)
            distances[idx] = dist
        results = np.argsort(distances)
        ret_time = time.time() - ret_time
        """
        * Calculate Statistics
        * mAP; recall@[5, 10, 50, 100, 200, 400], precision@[5, 10, 50, 100, 200, 400], F1@[5, 10, 50, 100, 200, 400] - mean, min and max
        """

        ngood = [0] * len(num_imgs_retrieved)
        nok = [0] * len(num_imgs_retrieved)
        njunk = [0] * len(num_imgs_retrieved)
        precision = ['-'] * len(num_imgs_retrieved)
        recall = ['-'] * len(num_imgs_retrieved)
        f1 = ['-'] * len(num_imgs_retrieved)
        
        positives_retrieved = 0
        ap = 0

        ngood_cur = nok_cur = njunk_cur = 0

        total_positives = len(filenames)
        for idx, result in enumerate(results, start=1):
            if feature_file_list[result][:-4] in good:
                ngood_cur += 1
                positives_retrieved += 1
                ap += (positives_retrieved/idx)
            
            if feature_file_list[result][:-4] in ok:
                nok_cur += 1
                positives_retrieved += 1
                ap += (positives_retrieved/idx)
            
            if feature_file_list[result][:-4] in junk:
                njunk_cur += 1
                positives_retrieved += 1
                ap += (positives_retrieved/idx)
            
            if idx in num_imgs_retrieved:
                stat_idx = num_imgs_retrieved.index(idx)
                tp = ngood_cur + nok_cur + njunk_cur
                fn = total_positives - tp
                fp = num_imgs_retrieved[stat_idx] - tp
                precision[stat_idx] = tp/(tp+fp)
                recall[stat_idx] = tp/(tp+fn)
                if tp != 0:
                    f1[stat_idx] = 2 * (precision[stat_idx]*recall[stat_idx]) / (precision[stat_idx] + recall[stat_idx])
                else:
                    f1[stat_idx] = 0
                ngood[stat_idx] = ngood_cur
                nok[stat_idx] = nok_cur
                njunk[stat_idx] = njunk_cur
            
            if positives_retrieved == total_positives:
                ap /= total_positives
                break

        print (
                "Query:", query_file[:-4], 
                "\nGood@"+str(num_imgs_retrieved)+":", ngood, 
                "\nOk@"+str(num_imgs_retrieved)+":", nok, 
                "\nJunk@"+str(num_imgs_retrieved)+":", njunk,
                "\nPrecision@"+str(num_imgs_retrieved)+":", precision,
                "\nRecall@"+str(num_imgs_retrieved)+":", recall,
                "\nF1@"+str(num_imgs_retrieved)+":", f1, 
                "\nAP@k (Average Precision):", ap,
                "\nTotal Retrieved (k):", idx, 
                "\nTotal Positives:", total_positives,
                "\nRetrieval Time (s):", ret_time,
                "\n-------------------------------------------------------------------------------------------------------------"
        )

        precision = list(map(lambda x: 0 if x == '-' else x, precision))
        recall = list(map(lambda x: 0 if x == '-' else x, recall))
        f1 = list(map(lambda x: 0 if x == '-' else x, f1))
        
        avg_good = list(map(add, avg_good, ngood))
        avg_ok = list(map(add, avg_ok, nok))
        avg_junk = list(map(add, avg_junk, njunk))

        avg_precision = list(map(add, avg_precision, precision))
        avg_recall = list(map(add, avg_recall, recall))
        avg_f1 = list(map(add, avg_f1, f1))

        max_precision = max(max_precision, precision)
        max_recall = max(max_recall, recall)
        max_f1 = max(max_f1, f1)

        min_precision = min(min_precision, precision)
        min_recall = min(min_recall, recall)
        min_f1 = min(min_f1, f1)

        mAP += ap
        avg_k += idx
        avg_time += ret_time

    
    avg_good = list(map(truediv, avg_good, [indx]*len(num_imgs_retrieved)))
    avg_ok = list(map(truediv, avg_ok, [indx]*len(num_imgs_retrieved)))    
    avg_junk = list(map(truediv, avg_junk, [indx]*len(num_imgs_retrieved)))
    avg_precision = list(map(truediv, avg_precision, [indx]*len(num_imgs_retrieved)))
    avg_recall = list(map(truediv, avg_recall, [indx]*len(num_imgs_retrieved)))
    avg_f1 = list(map(truediv, avg_f1, [indx]*len(num_imgs_retrieved)))
    mAP /= indx
    avg_k /= indx
    avg_time /= indx
    print ("*************************************************************************************************************")
    print ("Total Queries:", indx)

    print (
        "Mean Average Precision (mAP):", mAP,
        "\nAverage Retrievals (k):", avg_k,
        "\nAverage Time (s):", avg_time,
        "\nAvg Good imgs retrieved@"+str(num_imgs_retrieved)+":", avg_good,
        "\nAvg Ok imgs retrieved@"+str(num_imgs_retrieved)+":", avg_ok,
        "\nAvg Junk imgs retrieved@"+str(num_imgs_retrieved)+":", avg_junk,
        "\nAvg Precision@"+str(num_imgs_retrieved)+":", avg_precision,
        "\nAvg Recall@"+str(num_imgs_retrieved)+":", avg_recall,
        "\nAvg F1@"+str(num_imgs_retrieved)+":", avg_f1,
        "\nMax Precision@"+str(num_imgs_retrieved)+":", max_precision,
        "\nMax Recall@"+str(num_imgs_retrieved)+":", max_recall,
        "\nMax F1@"+str(num_imgs_retrieved)+":", max_f1,
        "\nMin Precision@"+str(num_imgs_retrieved)+":", min_precision,
        "\nMin Recall@"+str(num_imgs_retrieved)+":", min_recall,
        "\nMin F1@"+str(num_imgs_retrieved)+":", min_f1
    )

def sanity_check(path1, path2):
    num_color_clusters = 64
    d = [1, 3, 5, 7]
    cluster_model = get_cluster_model(num_color_clusters, "cluster_model.pkl")
    img1 = cv2.imread(path1)
    scale_percent = 10
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)

    img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
    img2 = cv2.imread(path2)
    scale_percent = 15
    width = int(img2.shape[1] * scale_percent / 100)
    height = int(img2.shape[0] * scale_percent / 100)
    dim = (width, height)

    img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
 
    a1 = generate_autocorrelogram(img1, d, cluster_model)

    a2 = generate_autocorrelogram(img2, d, cluster_model)
    #print (cluster_model.cluster_centers_)
    for i in range(len(d)):
        print (a1.T[i])
        print (a2.T[i])
        print (sum(a1.T[i]))
        print (sum(a2.T[i]))

    # print (a2)
    print (get_correlogram_distance(a1, a2))
    #a1 = np.load(path1)
    #a2 = np.load(path2)


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
    parser.add_argument("feature_folder",
                        help="Folder to store generated autocorrelograms",
                        type=check_path)

    parser.add_argument("cluster_model",
                        help="cluster_model",
                        type=str)

    parser.add_argument("--query_folder",
                        help="Folder containing txt files with queries",
                        type=check_path)

    parser.add_argument("--ground_truth",
                        help="Folder containing txt files with ground truth",
                        type=check_path)
    
    

    args = parser.parse_args()
    mode = args.mode

    query_folder = args.query_folder
    image_folder = args.image_folder
    save_folder = args.feature_folder
    gt_folder = args.ground_truth
    cluster_model = args.cluster_model

    if mode == "train":
        train(image_folder, save_folder, cluster_model)
    else:
        test(query_folder, image_folder, save_folder, cluster_model, gt_folder)
        #sanity_check("autocorrelogram_features/all_souls_000091.npy", "autocorrelogram_features/all_souls_000013.npy")
        #sanity_check("images/all_souls_000013.jpg", "images/all_souls_000220.jpg")
        

if __name__ == '__main__':
    main()