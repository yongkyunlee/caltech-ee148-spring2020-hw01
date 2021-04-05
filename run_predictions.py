import os
import numpy as np
import json
import math

from tqdm import tqdm
from PIL import Image

def detect_red_light_random(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    
    box_height = 8
    box_width = 6
    
    num_boxes = np.random.randint(1,5) 
    
    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)
        
        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width
        
        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

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

    for i, size_mult in enumerate(size_mult_list):
        r_light_resized = r_light1.resize((int(r_light1.width * size_mult), int(r_light1.height * size_mult)))
        r_light_resized = np.asarray(r_light_resized)
        r_light_height, r_light_width, _ = r_light_resized.shape
        r_light_resized = r_light_resized.reshape(-1)
        r_light_resized = r_light_resized / np.linalg.norm(r_light_resized)

        stride_height, stride_width = r_light_height // stride_ratio_list[i], r_light_width // stride_ratio_list[i]

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

        stride_height, stride_width = r_light_height // stride_ratio_list[i], r_light_width // stride_ratio_list[i]

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

# set the path to the downloaded data: 
data_path = './data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

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
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    # preds[file_names[i]] = detect_red_light_single_fixed_size(I, r_light_single, 0.9)
    # preds[file_names[i]] = detect_red_light_single_var_size(I, r_light_single, 0.9)
    preds[file_names[i]] = detect_red_light_double_var_size(I, r_light_single, r_light_double, 0.87)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_th0.87.json'),'w') as f:
    json.dump(preds,f)
