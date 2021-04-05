import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_PATH = './data/RedLights2011_Medium'
PREDS_PATH = './data/hw01_preds'
OUTPUT_PATH = './data/output'
os.makedirs(OUTPUT_PATH, exist_ok=True) # create directory if needed 

def get_match_score(img, sample):
    # I: np array, sample: Image
    sample = np.asarray(sample).reshape(-1)
    sample = sample / np.linalg.norm(sample)
    return np.dot(img, sample)


def process_overlap(I, points_arr, r_light1, r_light2):
    # I: np array, r_light1 and r_light2: Image
    remove_idx_arr = []
    for i in range(len(points_arr) - 1):
        for j in range(i+1, len(points_arr)):
            if i in remove_idx_arr or j in remove_idx_arr:
                continue
            y0_1, x0_1, y1_1, x1_1 = points_arr[i]
            y0_2, x0_2, y1_2, x1_2 = points_arr[j]
            dx = min(x1_1, x1_2) - max(x0_1, x0_2)
            dy = min(y1_1, y1_2) - max(y0_1, y0_2)
            if dx >= 0 and dy >= 0: # there is overlap
                # calcualte the match of the first box with r_light1
                r_light1_resized = r_light1.resize((x1_1 - x0_1, y1_1 - y0_1))
                score1 = get_match_score(I[y0_1:y1_1,x0_1:x1_1,:].reshape(-1), r_light1_resized)
                # calcualte the match of the first box with r_light2
                r_light2_resized = r_light2.resize((x1_1 - x0_1, y1_1 - y0_1))
                score2 = get_match_score(I[y0_1:y1_1,x0_1:x1_1,:].reshape(-1), r_light2_resized)
                score_i = max(score1, score2)
                
                # calculate the match of the second box with r_light1
                r_light1_resized = r_light1.resize((x1_2 - x0_2, y1_2 - y0_2))
                score1 = get_match_score(I[y0_2:y1_2,x0_2:x1_2,:].reshape(-1), r_light1_resized)
                # calculate the match of the second box with r_light2
                r_light2_resized = r_light2.resize((x1_2 - x0_2, y1_2 - y0_2))
                score2 = get_match_score(I[y0_2:y1_2,x0_2:x1_2,:].reshape(-1), r_light2_resized)
                score_j = max(score1, score2)

                # remove the smaller box
                if score_i < score_j:
                    remove_idx_arr.append(i)
                else:
                    remove_idx_arr.append(j)
    
    remove_idx_arr = sorted(remove_idx_arr, reverse=True)
    for idx in remove_idx_arr:
        del points_arr[idx]
    
    return points_arr

if __name__ == '__main__':
    preds_name = 'preds_th0.9_height.json'
    preds_processed_name = 'preds_th0.9_height_processed.json'

    file_names = sorted(os.listdir(DATA_PATH)) 
    file_names = [f for f in file_names if '.jpg' in f]
    with open(os.path.join(PREDS_PATH, preds_name)) as fp:
        preds = json.load(fp)
    r_light1 = Image.open('red_light_single.jpg')
    r_light2 = Image.open('red_light_double.jpg')

    preds_processed = {}
    for i, file_name in enumerate(file_names):
        if len(preds[file_name]) < 2:
            preds_processed[file_name] = preds[file_name]

        I = Image.open(os.path.join(DATA_PATH, file_names[i]))
        I = np.asarray(I)
        preds_processed[file_name] = process_overlap(I, preds[file_name], r_light1, r_light2)

    with open(os.path.join(PREDS_PATH, preds_processed_name), 'w') as fp:
        json.dump(preds_processed, fp)