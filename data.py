## Casey Astiz & Nick Cogswell 701
## file to read in data from camera on car and split into train/validation data


import numpy as np
import cv2

def load_train_data():
    """
    Returns video data as (train_data, valid_data): train_data and
    valid_data includes tuples of (x, y) where x is a pixels x 1 numpy array
    representing the image the car is seeing, and y is the direction of the car.
    The data comes in as rgb, but convert to gray scale images to better the
    model.
    """
    train_data = []
    valid_data = []
    return (train_data, valid_data)
