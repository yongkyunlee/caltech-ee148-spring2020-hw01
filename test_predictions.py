import json
import os

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

DATA_PATH = './data/RedLights2011_Medium'
PREDS_PATH = './data/hw01_preds'
OUTPUT_PATH = './data/output'
os.makedirs(OUTPUT_PATH, exist_ok=True) # create directory if needed 

def measure_score(file_name, points_arr, r_light1, r_light2):
    img = Image.open(os.path.join(DATA_PATH, file_name))
    img = np.array(img)

    r_light1_wh_ratio = r_light1.width / r_light1.height
    r_light2_wh_ratio = r_light2.width / r_light2.height

    score = 0

    # preds size is guaranteed to be the same as either the shape of
    # r_light1 and r_light2
    for points in points_arr:
        wh_ratio = (points[3] - points[1]) / (points[2] - points[0])
        if abs(r_light1_wh_ratio - wh_ratio) < abs(r_light2_wh_ratio - wh_ratio):
            r_light = r_light1
        else:
            r_light = r_light2
        r_light = r_light.resize((points[3] - points[1], points[2] - points[0]))
        
        r_light = np.asarray(r_light).reshape(-1)
        detected = img[points[0]:points[2],points[1]:points[3],:].reshape(-1)
        score += np.dot(r_light, detected)
    
    return score / len(points_arr)

def measure_overlap(points_arr):
    # use O(n^2) algorithm since it is assumed that there will not be many detections per image
    if len(points_arr) < 2:
        return 0
    
    overlap_area = 0
    for i in range(len(points_arr) - 1):
        for j in range(i+1, len(points_arr)):
            y0_1, x0_1, y1_1, x1_1 = points_arr[i]
            y0_2, x0_2, y1_2, x1_2 = points_arr[j]
            dx = min(x1_1, x1_2) - max(x0_1, x0_2)
            dy = min(y1_1, y1_2) - max(y0_1, y0_2)
            if dx >= 0 and dy >= 0:
                overlap_area += dx * dy
    
    total_area = sum(map(lambda x: (x[2] - x[0]) * (x[3] - x[1]), points_arr))
    return overlap_area / total_area

def measure_model_performance(file_names, preds, r_light1, r_light2):
    """ Measure performance of models
    - average detections per image
    - the average dot product 
    - overlapping area of the detected rectangles (likely to be a bad detection if there is a lot of overlap)
    """
    n_detections = 0
    score = 0
    overlap_ratio = 0
    fp = open(os.path.join(OUTPUT_PATH, 'detected_list.txt'), 'w')
    for file_name in tqdm(file_names):
        if len(preds[file_name]) == 0: # no detection at all
            continue
        n_detections += len(preds[file_name])
        score += measure_score(file_name, preds[file_name], r_light1, r_light2)
        overlap_ratio += measure_overlap(preds[file_name])
        fp.write(f'{file_name}: {len(preds[file_name])}\n')
    fp.close()
    return n_detections / len(file_names), score / len(file_names), overlap_ratio / len(file_names)

if __name__ == '__main__':
    preds_name = 'preds.json'

    file_names = sorted(os.listdir(DATA_PATH)) 
    file_names = [f for f in file_names if '.jpg' in f] 
    with open(os.path.join(PREDS_PATH, preds_name)) as fp:
        preds = json.load(fp)
    r_light1 = Image.open('red_light_single.jpg')
    r_light2 = Image.open('red_light_double.jpg')
    print(f'single red light size (w, h): ({r_light1.width}, {r_light1.height})')
    print(f'double red light size (w, h): ({r_light2.width}, {r_light2.height})')

    detection, match, overlap = measure_model_performance(file_names, preds, r_light1, r_light2)

    print(f'detection: {detection}')
    print(f'match: {match}')
    print(f'overlap: {overlap}')
