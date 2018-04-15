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
    
    x_min = pos[0] - patch_size//2
    x_max = pos[0] + patch_size//2 - 1
    y_min = pos[1] - patch_size//2
    y_max = pos[1] + patch_size//2 - 1
    
    if x_min < 0:
        x_min = 0
        x_max = x_min + patch_size - 1
    elif x_max >= width:
        x_max = width - 1
        x_min = x_max - patch_size + 1

    if y_min < 0:
        y_min = 0
        y_max = y_min + patch_size - 1
    elif y_max >= height:
        y_max = height - 1
        y_min = y_max - patch_size + 1

    return x_min, x_max, y_min, y_max


def get_patch(mask, image, patch_size=32, containing_road=True):

    pos = tuple(np.random.choice(mask.shape[0], 2))
    value = mask[pos]

    if containing_road:
        while value==0 or (not is_valid_position(pos, mask.shape)):
            pos = tuple(np.random.choice(mask.shape[0], 2))
            value = mask[pos]
        x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)
        return mask[x_min:x_max, y_min:y_max], image[x_min:x_max, y_min:y_max, :]
        
    else:
        x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)
        return mask[x_min:x_max, y_min:y_max], image[x_min:x_max, y_min:y_max, :]

if __name__ == "__main__":
    print("Testing preprocessing.py")
    assert is_valid_position((-1, 1), (250, 250)) == False
    assert is_valid_position((251, 2), (250, 250)) == False 
    assert is_valid_position((2, 300), (250, 250)) == False 
    assert is_valid_position((2, -1), (250, 250)) == False 
    assert is_valid_position((2, 2), (250, 250)) == True 

    print("1/2 tests run successfully")

    M = np.zeros((10,10))
    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,2), 4)
    assert (x_min == 0 and x_max == 3 and y_min == 0 and y_max == 3)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,2), 4)
    assert (x_min == 3 and x_max == 6 and y_min == 0 and y_max == 3)
    
    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,2), 4)
    assert (x_min == 6 and x_max == 9 and y_min == 0 and y_max == 3)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,5), 4)
    assert (x_min == 6 and x_max == 9 and y_min == 3 and y_max == 6)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,9), 4)
    assert (x_min == 0 and x_max == 3 and y_min == 6 and y_max == 9)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,5), 4)
    assert (x_min == 0 and x_max == 3 and y_min == 3 and y_max == 6)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,8), 4)
    assert (x_min == 6 and x_max == 9 and y_min == 6 and y_max == 9)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,8), 4)
    assert (x_min == 3 and x_max == 6 and y_min == 6 and y_max == 9)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,5), 4)
    assert (x_min == 3 and x_max == 6 and y_min == 3 and y_max == 6)

    print("2/2 tests run successfully")