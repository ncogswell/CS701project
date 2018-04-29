## Casey Astiz & Nick Cogswell
## CS701 project
## Spring 2018

## Skeleton Code for initial implementation
from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
import cv2

## Reference: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

def read_data():
    """take in all of the picture files, process them, split data information
    and return a list of data points"""

    img_dir = "images"
    data_path = os.path.join(img_dir,'*.bmp')
    files = glob.glob(data_path)
    data = []
    labels = []

    for file in files:
        img = cv2.imread(file)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp = np.array(gray_image)
        #print(temp.shape)
        flattened = temp.flatten().tolist()
        #print(flattened.shape)
        dist_index = file.find("-") + 1
        dist = file[dist_index: dist_index + 4]
        direction1 = file[dist_index+5:dist_index+6]
        direction2 = file[(dist_index+8):dist_index+9]

        data.append([flattened, dist])
        labels.append(direction1 + direction2)

    return data, labels


#basic set up for the neural net

def main():
    """Main function for basic implementation"""
    ##data is going to come from recordings from the car -> openCV

    # data = [[0., 0.], [1., 1.]]
    # ##training_data = [80% of data]
    # labels = [0, 1, 2, 3]
    # ## representing 4 basic states (stop, forward, right, left)
    #
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                      hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(data, labels) ## fit on train
    # return clf.predict([[2., 2.], [-1., -2.]]) ## predict on test data



read_data()
