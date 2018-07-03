from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_image(img_num, dir):
  im_mask = Image.open(dir+str(img_num)+'_mask.png')
  im_sat = Image.open(dir+str(img_num)+'_sat.jpg')
  return im_mask, im_sat

#Loads a single image given the filename in .jpg form
def load_file_road(filenamejpg):
  im_mask = Image.open(filenamejpg.replace('sat','osm'))
  im_sat = Image.open(filenamejpg)
  return im_mask, im_sat

#Loads a set of n images from a directory
def load_images_road(num_images, dir, details=False):
  i=0
  images=[]
  for file in glob.glob(dir+'*_sat*.png'):
    print(file)
    i+=1
    if details: print(i, file)
    images.append(load_file(file))
    if i==num_images:
      break
  return images

#Loads a single image given the filename in .jpg form
def load_file(filenamejpg):
  im_mask = Image.open(filenamejpg[:-8]+'_mask.png')
  im_sat = Image.open(filenamejpg)
  return im_mask, im_sat

#Loads a set of n images from a directory
def load_images(num_images, dir, details=False):
  i=0
  images=[]
  for file in glob.glob(dir+'*.jpg'):
    i+=1
    if details: print(i, file)
    images.append(load_file(file))
    if i==num_images:
      break
  return images

#Given a input folder and an output folder, resize all images in the former and save to the later
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

#Check if a given position is inside a square
def is_valid_position(position, shape):
    width, height = shape[0], shape[1]
    return position[0] > 0 and position[0] < width and position[1] > 0 and position[1] < height

#Compute boundaries for a mask, starting at a given point and taking a size into account
def calculate_boundaries(mask, pos, patch_size):
    width, height = mask.shape[0], mask.shape[1]
    
    x_min = pos[0] - patch_size//2
    x_max = pos[0] + patch_size//2
    y_min = pos[1] - patch_size//2
    y_max = pos[1] + patch_size//2
    
    if x_min < 0:
        x_min = 0
        x_max = x_min + patch_size
    elif x_max >= width:
        x_max = width
        x_min = x_max - patch_size

    if y_min < 0:
        y_min = 0
        y_max = y_min + patch_size
    elif y_max >= height:
        y_max = height
        y_min = y_max - patch_size

    return x_min, x_max, y_min, y_max

#Given an image and its corresponding mask, calculate a patch from that image of the demanded size
def get_patch(mask, image, patch_size=64, containing_road=True):

    black_corner = True
    while black_corner:

        pos = tuple(np.random.choice(mask.shape[0], 2))
        value = mask[pos]

        if containing_road:
            while value==0 or (not is_valid_position(pos, mask.shape)):
                pos = tuple(np.random.choice(mask.shape[0], 2))
                value = mask[pos]
            x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)
            
        else:
            x_min, x_max, y_min, y_max = calculate_boundaries(mask, pos, patch_size)

        black_corner = is_black(image[x_min, y_min, :]) or is_black(image[x_min, y_max-1, :]) or is_black(image[x_max-1, y_min, :]) or is_black(image[x_max-1, y_max-1, :])

    return mask[x_min:x_max, y_min:y_max], image[x_min:x_max, y_min:y_max, :]

#Return true if a pixel color is full black. False otherwise
def is_black(pixel):
    return np.array_equal(np.array([0,0,0]), pixel)

#Get patch sampling of an image and some rotated versions of the same image
def sample_image(mask, sat , patch_size=64, num_patches=100, rotations=[-15,-10,-5,0,5,10,15], min_road_ratio=.5):
    mask_patches = []
    sat_patches = []

    mask_rotations = [mask.rotate(r) for r in rotations]
    sat_rotations = [sat.rotate(r) for r in rotations]

    for mask, sat in zip(mask_rotations, sat_rotations):
        mask_matrix = np.array(mask.convert("L"))
        sat_matrix = np.array(sat.convert("RGB"))
        partition = int(num_patches*min_road_ratio)
        
        for i in range(partition):
            mask_patch, sat_patch = get_patch(mask_matrix, sat_matrix, patch_size=patch_size)
            mask_patches.append(mask_patch)
            sat_patches.append(sat_patch)
        for i in range(partition, num_patches):
            mask_patch, sat_patch = get_patch(mask_matrix, sat_matrix, patch_size=patch_size, containing_road=False)
            mask_patches.append(mask_patch)
            sat_patches.append(sat_patch)
    
    return np.array(mask_patches), np.array(sat_patches)

#Evaluate how good a predicted mask is
def accuracy(original, predicted):
    difference = np.absolute(original/255. - predicted)
    accuracy = 1. - np.sum(difference)/(difference.shape[0]*difference.shape[1])
    return accuracy

if __name__ == "__main__":
    print("Testing preprocessing.py")
    assert is_valid_position((-1, 1), (250, 250)) == False
    assert is_valid_position((251, 2), (250, 250)) == False 
    assert is_valid_position((2, 300), (250, 250)) == False 
    assert is_valid_position((2, -1), (250, 250)) == False 
    assert is_valid_position((2, 2), (250, 250)) == True 

    print("1/3 tests run successfully")

    M = np.zeros((10,10))
    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,2), 4)
    assert (x_min == 0 and x_max == 4 and y_min == 0 and y_max == 4)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,2), 4)
    assert (x_min == 3 and x_max == 7 and y_min == 0 and y_max == 4)
    
    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,2), 4)
    assert (x_min == 6 and x_max == 10 and y_min == 0 and y_max == 4)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,5), 4)
    assert (x_min == 6 and x_max == 10 and y_min == 3 and y_max == 7)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,9), 4)
    assert (x_min == 0 and x_max == 4 and y_min == 6 and y_max == 10)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (2,5), 4)
    assert (x_min == 0 and x_max == 4 and y_min == 3 and y_max == 7)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (8,8), 4)
    assert (x_min == 6 and x_max == 10 and y_min == 6 and y_max == 10)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,8), 4)
    assert (x_min == 3 and x_max == 7 and y_min == 6 and y_max == 10)

    x_min, x_max, y_min, y_max = calculate_boundaries(M, (5,5), 4)
    assert (x_min == 3 and x_max == 7 and y_min == 3 and y_max == 7)

    print("2/3 tests run successfully")

    assert (is_black(np.array([0,0,0])))
    assert (not is_black(np.array([0,0,1])))

    print("3/3 tests run successfully")