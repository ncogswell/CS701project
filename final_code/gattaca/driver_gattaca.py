## Casey Astiz & Nick Cogswell
## CS701 project
## Spring 2018

"""
Constantly runs input sent from the pi through the specified model and
sends the results back to the pi
"""

from subprocess import call
import threading
from time import sleep
#from time import time
#from sklearn.preprocessing import LabelEncoder
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.models import Sequential
from keras.models import model_from_json
#from keras.layers import Dense, Activation, Flatten
#from keras.utils import np_utils
#from keras.layers import Dense, Dropout
#from keras.layers import Embedding
#from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
#from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import SGD
#from tqdm import tqdm
import numpy as np
import cv2
import os
#import glob
import keras

model_name = 'models/0.63'

def open_model():
    """
    Loads the model as specified by model_name
    """
    # load json and create model
    print('Loading Model from Disk')
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+'.h5')
    print("Model Loaded") 
    loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    return loaded_model

def send_dir(dir):
    """ Sends a direction to the pi """
    dir = dir_dict[dir]
    print("Sending Direction:", dir_translate[dir])
    call(['ssh', '-p', '2222', 'pi@127.0.0.1', 'rm ~/CS701car/direction/* && touch ~/CS701car/direction/'+dir])

def usr_control():
    """ Waits for the user to quit the program """
    global user_in
    while user_in != 'q':
        user_in = input()
        sleep(1)

def get_dir():
    """
    Reads the image in directory img/
    Processes the image
    Runs the image through the model
    Sends prediction to the pi
    """
    if len(os.listdir('img')) > 0:
        for file in os.listdir('img'):
            img = cv2.imread("img/"+file)
            if img is None:
                sleep(.1)
            else:
                edge_image = cv2.Canny(img, 100, 200)
                temp = np.array(edge_image)
                flattened = temp.flatten().tolist()
                prediction = model.predict_proba(np.array(flattened).reshape((1,307200)), batch_size=1)
                send_dir(prediction[0].tolist().index(1))
    else:
        sleep(.1)
    

def main():
    """ Opens model and runs on pi input until user quit """
    global model
    model = open_model()
    while user_in != 'q':
        get_dir()

# Global variables and threads
user_in = ''
dir_dict = {0:'ws', 1:'wa', 2:'wd', 3:'es'}
dir_translate = {'ws':'Straight', 'wa':'Left', 'wd':'Right', 'es':'Stop', 'q':'Quit'}
model = None

thread = threading.Thread(target=usr_control, name="usr_control")
thread.start()
main()

# reset
print("Quit and Clean")
call(['ssh', '-p', '2222', 'pi@127.0.0.1', 'rm ~/CS701car/direction/* && touch ~/CS701car/direction/q'])
sleep(2)
call(['ssh', '-p', '2222', 'pi@127.0.0.1', 'rm ~/CS701car/direction/* && touch ~/CS701car/direction/es'])
