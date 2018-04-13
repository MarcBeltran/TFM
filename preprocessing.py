from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def resize_images(input_folder, output_folder, ratio, image_format=".jpg"):
    for filename in os.listdir(input_folder):
        if filename.endswith(image_format):
            image = Image.open(os.path.join(input_folder, filename))
            width, height = image.size
            resized_image = image.resize((int(width*ratio), int(height*ratio)))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            resized_image.save(os.path.join(output_folder, filename))
            image.close()
    return

def is_valid_position(position, shape):
    width, height = shape[0], shape[1]
    return position[0] > 0 and position[0] < width and position[1] > 0 and position[1] < height

def calculate_boundaries(mask, pos, patch_size):
    width, height = mask.shape[0], mask.shape[1]
    
    x_min, x_max, y_min, y_max = -1, -1, -1, -1
    
    if pos[0] < patch_size//2 - 1:
        x_min = 0
        x_max = x_min + patch_size
    elif pos[0] > width - patch_size//2 + 1:
        x_max = width - 1
        x_min = x_max - patch_size

    if pos[1] < patch_size//2 - 1:
        y_min = 0
        y_max = y_min + patch_size
    elif pos[1] > height - patch_size//2 + 1:
        y_max = height - 1
        y_min = y_max - patch_size

    return x_min, x_max, y_min, y_max


def get_patch(mask, image, patch_size=32, containing_road=True):

    pos = tuple(np.random.choice(mask.shape[0], 2))
    print(pos)
    value = mask[pos]

    if containing_road:
        while value==0 and (not is_valid_position(pos, mask.shape)):
            pos = tuple(np.random.choice(mask.shape[0], 2))
            value = mask[pos]
        x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)
        return mask[x_min:x_max, y_min:y_max], image[x_min:x_max, y_min:y_max, :]
        
    else:
        x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)
        return mask[x_min:x_max, y_min:y_max], image[x_min:x_max, y_min:y_max, :]