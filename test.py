import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
from glob import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from lesson_functions import *
from sklearn.model_selection import train_test_split


def test_hog_extraction():
    cars = []
    notcars = []
    print('Extracting features from the dataset...')

    [cars.append(y) for x in os.walk('C:/Users/LPC/Documents/GitHub/kitti/vehicles') for y in glob(os.path.join(x[0], '*.png'))]
    [notcars.append(y) for x in os.walk('C:/Users/LPC/Documents/GitHub/kitti/non-vehicles') for y in glob(os.path.join(x[0], '*.png'))]
    
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image_car = mpimg.imread(cars[ind])
    image_notcar = mpimg.imread(notcars[ind])

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11  # HOG orientations
    pix_per_cell = 16 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()


    # Call our function with vis=True to see an image output
    features, hog_image_car = get_hog_features(image_car[:,:,2], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
                        
    features, hog_image_not_car = get_hog_features(image_notcar[:,:,2], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
                        
    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image_car, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image_car, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image_notcar, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image_not_car, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
    
def main():
    test_hog_extraction()
    
if __name__ == "__main__":
    main()   
