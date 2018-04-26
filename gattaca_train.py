## Casey Astiz and Nick Cogswell
## CS701 project
## Spring 2018

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from tqdm import tqdm
import numpy as np
import cv2
import os
import glob
import keras


def read_data():
    """take in all of the picture files, process them, split data information
    and return a list of data points"""

    img_dir = "/data/scratch/castiz/training_images"
    data_path = os.path.join(img_dir,'*.bmp')
    files = glob.glob(data_path)
    data = []
    labels = []

    for file in tqdm(files[:1000]):
        img = cv2.imread(file)
        if img is None:
            pass
        else:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            temp = np.array(gray_image)
            ##print(temp.shape)
            flattened = temp.flatten().tolist()
            #print(flattened.shape)
            data.append(flattened)
            #find directions in file name
            label_index = file.find('.bmp')
            direction = file[label_index-3] + file[label_index-1] 
            labels.append(direction)
   

    #make categorical labels
    unique_labels = list(set(labels))
    newlabels = []
    for label in labels:
        newlabel = unique_labels.index(label)
        newlabels.append(newlabel)     

    categorical_labels = keras.utils.to_categorical(np.array(newlabels), num_classes=4)

    return data, categorical_labels

def baseline_model():
    """Create a deep neural network as a baseline"""
    
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=307200))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model



def convnet_model():
    """Create a CNN model."""

    #return model


def train_test():
    """Read data, train model, test model, save model"""
    data, labels = read_data()

    dataArray = np.array(data)
    labelsArray = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(dataArray, labelsArray, test_size=0.33, random_state=42)

    model = baseline_model() #choose baseline_model() or convnet_model()

    model.fit(x_train, y_train, epochs = 10)

    score = model.evaluate(x_test, y_test)
    print('Score = ')
    print(score)

    save_model()


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


train_test()
##############
# Current standings
# 0% with array issue
# 23% with baseline model with 10 epochs
# 56% with baeline model with more layers
