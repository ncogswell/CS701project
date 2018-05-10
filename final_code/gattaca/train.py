## Casey Astiz and Nick Cogswell
## CS701 project
## Spring 2018

"""
train.py takes the highest performing NN configurations from previous tests and trains
them with the whole data set. All the models are saved in the folder final_models and
accuracy data is recorded in the file results.txt
"""

#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
#from keras.models import model_from_json
#from keras.models import Model
#from keras.layers import Dense, Activation, Flatten
#from keras.utils import np_utils
#from keras.layers import Dense, Dropout
#from keras.layers import Embedding
#from keras.layers import Input, Convolution2D, MaxPooling2D
#from keras.optimizers import SGD
from tqdm import tqdm
import numpy as np
import cv2
import os
import glob
import keras
from time import time

def read_data():
    """
    take in all of the picture files, process them, split data information
    and return a list of data points
    """

    img_dir = "/data/scratch/castiz/training_images"
    data_path = os.path.join(img_dir,'*.bmp')
    files = glob.glob(data_path)
    data = []
    straight_data = []
    labels = []
    straight_labels = []
    lab_dict = {'ws':0, 'wa':1, 'wd':2, 'es':3}

    print("Reading and processing data")
    #for file in tqdm(files[:20]):
    for file in tqdm(files):
        img = cv2.imread(file)
        if img is None:
            pass
        else:
            edge_image = cv2.Canny(img, 100, 200)
            temp = np.array(edge_image)
            flattened = temp.flatten().tolist()
            label_index = file.find('.bmp')
            direction = file[label_index-3] + file[label_index-1] 

            if direction == 'ws':
                straight_data.append(flattened)
                straight_labels.append(lab_dict[direction])
            else:
                data.append(flattened)
                labels.append(lab_dict[direction])
  
    #Currently about 60% of our data is straight, so making our classes less skewed
    index = int(len(straight_data)*.25)
    data = data + straight_data[:index]
    labels = labels + straight_labels[:index]

    print("Processing labels")
    categorical_labels = keras.utils.to_categorical(np.array(labels), num_classes=4)

    return data, categorical_labels

def build_model1(x,y):
    """
    Return neural network with two hidden layers containing x and y neurons
    """
 
    model = Sequential()
    model.add(Dense(x, activation='relu', input_dim=307200))
    model.add(Dense(y, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def build_model2(x,y,z):
    """
    Return neural network with three hidden layers containing x, y, and z neurons
    """
 
    model = Sequential()
    model.add(Dense(x, activation='relu', input_dim=307200))
    model.add(Dense(y, activation='relu'))
    model.add(Dense(z, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def build_model3(x,y,z,a):
    """
    Return neural network with four hidden layers containing x, y, z, and a neurons
    """
 
    model = Sequential()
    model.add(Dense(x, activation='relu', input_dim=307200))
    model.add(Dense(y, activation='relu'))
    model.add(Dense(z, activation='relu'))
    model.add(Dense(a, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def train_test():
    """
    Read data, train models and test models, save models and accuracy
    """

    epoch_limit = 20 
    directory = 'final_models'

    models1 = [
        (256,64),
        (256,256),
        (512,64)]
    models2 = [
        (256,128,64),
        (256,128,128),
        (256,256,64),
        (512,128,128),
        (512,512,64),
        (512,512,256),
        (512,512,512)]
    models3 = [
        (256,256,256,256),
        (512,128,128,128),
        (512,256,256,128),
        (1024,256,128,128),
        (1024,256,256,128),
        (1024,512,512,128),
        (1024,1024,256,256),
        (1024,1024,512,256),
        (1024,1024,512,512),
        (1024,1024,1024,256),
        (1024,1024,1024,512),
        (1024,1024,1042,1024)]
    
    data, labels = read_data()
    dataArray = np.array(data)
    labelsArray = np.array(labels)
    
    x_train, x_test, y_train, y_test = train_test_split(dataArray, labelsArray, test_size=0.20, random_state=42)

    x_train = x_train / np.max(x_train) # Normalise data
    x_test = x_train / np.max(x_test) # Normalise data
    y_train = y_train / np.max(y_train) # Normalise data
    y_test = y_train / np.max(y_test) # Normalise data

    file = open('results.txt', 'w')
    file.write('x\ty\tz\ta\ttime\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14\t15\t16\t17\t18\t19\t20\n') 

    for param in models1:
        x,y = param
        model = build_model1(x, y)
        acc = []
        start_time = time()
        for i in range (epoch_limit):
            model.fit(x_train, y_train, epochs = 1)
            score = model.evaluate(x_test, y_test)
            acc.append(str(round(score[1], 3)))
            save_model(model, directory + '/' + str(round(score[1], 3)) + '_' + str(i) + '_' + str(x) + 'x' + str(y))
        end_time = time()
        line = '\t'.join([str(x), str(y), '\t', str(round(end_time - start_time, 1))]+acc)
        print(line)
        file.write(line + '\n')
    
    for param in models2:
        x,y,z = param
        model = build_model2(x, y, z)
        acc = []
        start_time = time()
        for i in range (epoch_limit):
            model.fit(x_train, y_train, epochs = 1)
            score = model.evaluate(x_test, y_test)
            acc.append(str(round(score[1], 3)))
            save_model(model, directory + '/' + str(round(score[1], 3)) + '_' + str(i) + '_' + str(x) + 'x' + str(y) + 'x' + str(z))
        end_time = time()
        line = '\t'.join([str(x), str(y), str(z), '', str(round(end_time - start_time, 1))]+acc)
        print(line)
        file.write(line + '\n')
    
    for param in models3:
        x,y,z,a = param
        model = build_model3(x, y, z, a)
        acc = []
        start_time = time()
        for i in range(epoch_limit):
            model.fit(x_train, y_train, epochs = 1)
            score = model.evaluate(x_test, y_test)
            acc.append(str(round(score[1], 3)))
            save_model(model, directory + '/' + str(round(score[1], 3)) + '_' + str(i) + '_' + str(x) + 'x' + str(y) + 'x' + str(z) + 'x' + str(a))
        end_time = time()
        line = '\t'.join([str(x), str(y), str(z), str(a), str(round(end_time - start_time, 1))]+acc)
        print(line)
        file.write(line + '\n')

def save_model(model, name):
    """
    Takes a model and name, and saves that model with that name
    """
    model_json = model.to_json()
    with open(name + '.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights(name + ".h5")
    print("Saved model to disk")

train_test()
