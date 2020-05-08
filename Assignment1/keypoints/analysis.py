import json
import numpy as np

for file in ["LoG_blobs.json", "Surf_blobs.json"]:
    with open(file) as jf:
        data = json.load(jf)
        avg_blobs = 0
        max_blobs = -np.inf
        min_blobs = np.inf
        max_img = 0
        min_img = 0
        for idx, img in enumerate(data, start=1):
            num_blobs = len(data[img])
            if num_blobs > max_blobs:
                max_blobs = num_blobs
                max_img = img
            if num_blobs < min_blobs:
                min_blobs = num_blobs
                min_img = img
            avg_blobs += num_blobs
        avg_blobs /= idx
        print (file)
        print ("Average Blobs:", avg_blobs, "\nMax Blobs:", max_blobs, "Max Img:", max_img, "\nMin Blobs:", min_blobs, "Min Img:", min_img, "\n")
