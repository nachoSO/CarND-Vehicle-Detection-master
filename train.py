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

def train():
    cars = []
    notcars = []
    print('Extracting features from the dataset...')

    [cars.append(y) for x in os.walk('./kitti/vehicles') for y in glob(os.path.join(x[0], '*.png'))]
    [notcars.append(y) for x in os.walk('./kitti/non-vehicles') for y in glob(os.path.join(x[0], '*.png'))]
    
    #Accuracy: 0.988457207207 = YUV
    #Accuracy: 0.991272522523 = YCrCb
    
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
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    
   
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    print('Training the dataset')
    svc.fit(X_train, y_train)
    
    print("Accuracy: "+str(svc.score(X_test, y_test)))
    
    filename = 'svm_model.p'
    pickle.dump(svc, open(filename, 'wb'))
    
    filename = 'X_scaler.p'
    pickle.dump(X_scaler, open(filename, 'wb'))


    
def main():
    train()
    
if __name__ == "__main__":
    main()   
