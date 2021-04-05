import os
import numpy as np
import json
import math

from tqdm import tqdm
from PIL import Image

import process_preds

DATA_PATH = './data/RedLights2011_Medium'
PREDS_PATH = './data/hw01_preds'
OUTPUT_PATH = './data/output'
os.makedirs(OUTPUT_PATH, exist_ok=True) # create directory if needed 

def detect_red_light_single_fixed_size(I, r_light, threshold):
    bounding_boxes = []

    r_light = r_light.resize((r_light.width // 2, r_light.height // 2))
    r_light = np.asarray(r_light)
    r_light_height, r_light_width, _ = r_light.shape

    img_height, img_width, _ = I.shape

    r_light = r_light.reshape(-1)
    r_light = r_light / np.linalg.norm(r_light)
    # iterate through the centers
    stride_height = r_light_height // 10
    stride_width = r_light_width // 10

    for i in range(0, img_height - r_light_height, stride_height):
        for j in range(0, img_width - r_light_width, stride_width):
            # skip if out of range
            if i + r_light_height > img_height or j + r_light_width > img_width:
                continue
            # take the part of the image to compare to the red light
            x = I[i:i+r_light_height,j:j+r_light_width,:]
            x = x.reshape(-1) # reshape to 1d
            x = x / np.linalg.norm(x) # normalize
            match = np.dot(r_light, x)
            if match > threshold:
                bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])
    return bounding_boxes

def detect_red_light_single_var_size(I, r_light, threshold): 
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 

    img_height, img_width, _ = I.shape

    size_mult_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # multipliers for width and height
    for size_mult in size_mult_list:
        r_light_resized = r_light.resize((int(r_light.width * size_mult), int(r_light.height * size_mult)))
        r_light_resized = np.asarray(r_light_resized)
        r_light_height, r_light_width, _ = r_light_resized.shape
        r_light_resized = r_light_resized.reshape(-1)
        r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

        stride_height, stride_width = r_light_height // 5, r_light_width // 5

        for i in range(0, img_height - r_light_height, stride_height):
            for j in range(0, img_width - r_light_width, stride_width):
                # skip if out of range
                if i + r_light_height > img_height or j + r_light_width > img_width:
                    continue
                # take the part of the image to compare to the red light
                x = I[i:i+r_light_height,j:j+r_light_width,:]
                x = x.reshape(-1) # reshape to 1d
                x = x / np.linalg.norm(x) # normalize
                match = np.dot(r_light_resized, x)
                if match > threshold:
                    bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])

    return bounding_boxes

def detect_red_light_double_var_size(I, r_light1, r_light2, threshold): 
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 

    img_height, img_width, _ = I.shape

    size_mult_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # multipliers for width and height
    stride_ratio_list = [6, 6, 6, 7, 7, 7, 8, 8, 8]

    for idx, size_mult in enumerate(size_mult_list):
        r_light_resized = r_light1.resize((int(r_light1.width * size_mult), int(r_light1.height * size_mult)))
        r_light_resized = np.asarray(r_light_resized)
        r_light_height, r_light_width, _ = r_light_resized.shape
        r_light_resized = r_light_resized.reshape(-1)
        r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

        stride_height, stride_width = r_light_height // stride_ratio_list[idx], r_light_width // stride_ratio_list[idx]

        for i in range(0, img_height - r_light_height, stride_height):
            for j in range(0, img_width - r_light_width, stride_width):
                # skip if out of range
                if i + r_light_height > img_height or j + r_light_width > img_width:
                    continue
                # take the part of the image to compare to the red light
                x = I[i:i+r_light_height,j:j+r_light_width,:]
                x = x.reshape(-1) # reshape to 1d
                x = x / np.linalg.norm(x) # normalize
                match = np.dot(r_light_resized, x)
                if match > threshold:
                    bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])

    for i, size_mult in enumerate(size_mult_list):
        r_light_resized = r_light2.resize((int(r_light2.width * size_mult), int(r_light2.height * size_mult)))
        r_light_resized = np.asarray(r_light_resized)
        r_light_height, r_light_width, _ = r_light_resized.shape
        r_light_resized = r_light_resized.reshape(-1)
        r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

        stride_height, stride_width = r_light_height // stride_ratio_list[idx], r_light_width // stride_ratio_list[idx]

        for i in range(0, img_height - r_light_height, stride_height):
            for j in range(0, img_width - r_light_width, stride_width):
                # skip if out of range
                if i + r_light_height > img_height or j + r_light_width > img_width:
                    continue
                # take the part of the image to compare to the red light
                x = I[i:i+r_light_height,j:j+r_light_width,:]
                x = x.reshape(-1) # reshape to 1d
                x = x / np.linalg.norm(x) # normalize
                match = np.dot(r_light_resized, x)
                if match > threshold:
                    bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])
    
    return bounding_boxes

def detect_red_light(I, r_light1, r_light2, threshold): 
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 

    img_height, img_width, _ = I.shape

    size_mult_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # multipliers for width and height
    stride_ratio_list = [5, 5, 5, 6, 6, 6, 7, 7, 7]
    height_distortion_list = [0.8, 0.9, 1, 1.1, 1.2]

    for height_mult in height_distortion_list:    
        for idx, size_mult in enumerate(size_mult_list):
            r_light_resized = r_light1.resize((int(r_light1.width * size_mult), int(r_light1.height * size_mult * height_mult)))
            r_light_resized = np.asarray(r_light_resized)
            r_light_height, r_light_width, _ = r_light_resized.shape
            r_light_resized = r_light_resized.reshape(-1)
            r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

            stride_height, stride_width = r_light_height // stride_ratio_list[idx], r_light_width // stride_ratio_list[idx]

            for i in range(0, img_height - r_light_height, stride_height):
                for j in range(0, img_width - r_light_width, stride_width):
                    # skip if out of range
                    if i + r_light_height > img_height or j + r_light_width > img_width:
                        continue
                    # take the part of the image to compare to the red light
                    x = I[i:i+r_light_height,j:j+r_light_width,:]
                    x = x.reshape(-1) # reshape to 1d
                    x = x / np.linalg.norm(x) # normalize
                    match = np.dot(r_light_resized, x)
                    if match > threshold:
                        bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])

        for idx, size_mult in enumerate(size_mult_list):
            r_light_resized = r_light2.resize((int(r_light2.width * size_mult), int(r_light2.height * size_mult * height_mult)))
            r_light_resized = np.asarray(r_light_resized)
            r_light_height, r_light_width, _ = r_light_resized.shape
            r_light_resized = r_light_resized.reshape(-1)
            r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

            stride_height, stride_width = r_light_height // stride_ratio_list[idx], r_light_width // stride_ratio_list[idx]

            for i in range(0, img_height - r_light_height, stride_height):
                for j in range(0, img_width - r_light_width, stride_width):
                    # skip if out of range
                    if i + r_light_height > img_height or j + r_light_width > img_width:
                        continue
                    # take the part of the image to compare to the red light
                    x = I[i:i+r_light_height,j:j+r_light_width,:]
                    x = x.reshape(-1) # reshape to 1d
                    x = x / np.linalg.norm(x) # normalize
                    match = np.dot(r_light_resized, x)
                    if match > threshold:
                        bounding_boxes.append([i, j, i + r_light_height, j + r_light_width])
    
    return bounding_boxes

if __name__ == '__main__':
    # get sorted list of files: 
    file_names = sorted(os.listdir(DATA_PATH)) 

    # remove any non-JPEG files: 
    file_names = [f for f in file_names if '.jpg' in f]

    # load the red_light_single and red_light_double images
    r_light_single = Image.open('red_light_single.jpg')
    r_light_double = Image.open('red_light_double.jpg')
    print(f'r_light_single size: ({r_light_single.height}, {r_light_single.width})')

    preds = {}
    # file_names = ['RL-011.jpg']
    for i in tqdm(range(len(file_names))):
        # read image using PIL:
        I = Image.open(os.path.join(DATA_PATH,file_names[i]))
        
        # convert to numpy array:
        I = np.asarray(I)
        # preds[file_names[i]] = detect_red_light_single_var_size(I, r_light_single, 0.9)
        preds[file_names[i]] = detect_red_light_double_var_size(I, r_light_single, r_light_double, 0.9)
        # preds[file_names[i]] = detect_red_light(I, r_light_single, r_light_double, 0.9)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(PREDS_PATH,'preds_th0.9_double_var.json'),'w') as f:
        json.dump(preds,f)
    
    preds_processed = {}
    for i in tqdm(range(len(file_names))):
        file_name = file_names[i]
        if len(preds[file_name]) < 2:
            preds_processed[file_name] = preds[file_name]

        I = Image.open(os.path.join(DATA_PATH, file_names[i]))
        I = np.asarray(I)
        preds_processed[file_name] = process_preds.process_overlap(I, preds[file_name], r_light_single, r_light_double)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(PREDS_PATH,'preds_th0.9_double_var_processed.json'),'w') as f:
        json.dump(preds,f)
    
