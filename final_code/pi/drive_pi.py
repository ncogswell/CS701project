## Nick Cogswell and Casey Astiz
## CS701 Spring 2018

"""
Drives Wall-E by initiating pi GPIO pins, constantly collecting images and sending to
gattaga, and reading directions in directory directions/ until quit command from gattaca.
Emergency Stop feature untested on account of range sensor difficulties, so distance data
is hardcoded and dist() is never called.

Motor Controls:
    Drive: w- forward, x- backward, e- stop
    Turn: a- left, d- right, s- stop
    z- full stop, q- quit
"""

import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import threading
import csv
import os
from subprocess import call

# GPIO Initialization
GPIO.setmode(GPIO.BOARD)
 
Motor1A = 16
Motor1B = 18
Motor1E = 22
 
Motor2A = 19
Motor2B = 21
Motor2E = 23
 
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1E,GPIO.OUT)
 
GPIO.setup(Motor2A,GPIO.OUT)
GPIO.setup(Motor2B,GPIO.OUT)
GPIO.setup(Motor2E,GPIO.OUT)

TRIG = 3
ECHO = 5

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

# Driving Methods
def set_state_drive(a):
    """ sets drive portion of current state tuple """
    global current_state
    x,y = current_state
    current_state = a,y

def set_state_turn(a):
    """ sets turn portion of current state tuple """
    global current_state
    x,y = current_state
    current_state = x,a

def drive(dir):
    """ makes car drive forward, backward, or stops driving """
    set_state_drive(dir)
    if dir == "w":
        GPIO.output(Motor2A,GPIO.HIGH)
        GPIO.output(Motor2B,GPIO.LOW)
        GPIO.output(Motor2E,GPIO.HIGH)
    elif dir == "x":
        GPIO.output(Motor2A,GPIO.LOW)
        GPIO.output(Motor2B,GPIO.HIGH)
        GPIO.output(Motor2E,GPIO.HIGH)
    else: 
        GPIO.output(Motor2E,GPIO.LOW)
        GPIO.output(Motor2A,GPIO.LOW)
        GPIO.output(Motor2B,GPIO.LOW)

def turn(dir):
    """ makes car turn left, right, or stops turning """
    set_state_turn(dir)
    if dir == "s":
        GPIO.output(Motor1E,GPIO.LOW)
    elif dir == "a":
        GPIO.output(Motor2E,GPIO.LOW)
        time.sleep(.2)
        GPIO.output(Motor1A,GPIO.LOW)
        GPIO.output(Motor1B,GPIO.HIGH)
        GPIO.output(Motor1E,GPIO.HIGH)
        time.sleep(.4)
        GPIO.output(Motor2E,GPIO.HIGH)
    else:
        GPIO.output(Motor2E,GPIO.LOW)
        time.sleep(.2)
        GPIO.output(Motor1A,GPIO.HIGH)
        GPIO.output(Motor1B,GPIO.LOW)
        GPIO.output(Motor1E,GPIO.HIGH)
        time.sleep(.4)
        GPIO.output(Motor2E,GPIO.HIGH)

def full_stop():
    """ stops all motors by setting all GPIO pins to low """
    GPIO.output(Motor1E,GPIO.LOW)
    GPIO.output(Motor2A,GPIO.LOW)
    GPIO.output(Motor2B,GPIO.LOW)
    GPIO.output(Motor2E,GPIO.LOW)

# Range Sensor Methods
def dist():
    """ uses the ultrasonic range sensor to return distance of object in front of car """
    global emg_break
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time() 
    while GPIO.input(ECHO)==0:
        pulse_start = time.time() 

    pulse_end = time.time() 
    while GPIO.input(ECHO)==1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start

    distance = pulse_duration * 17150

    distance = round(distance, 2)

    if distance < 10:
        full_stop()
        emg_break = True
        current_state = 'e','s'
        dir = 'es'
        print("!!! EMERGENCY BREAK !!!", distance)
    else:
        emg_break = False

    return distance

# Webcam Methods
def take_picture():
    """takes a picture at the current moment to be labeled in main"""
    cam = cv2.VideoCapture(0)
    s, im = cam.read() 
    temp = np.array(im)
    flattened = temp.flatten()
    return flattened

# Data Collection Methods
def collect_data():
    """ constantly takes pictures and sends them to gattaga """
    global data_size
    global dist_array
    cam = cv2.VideoCapture(0)
    while dir != 'q':
        s, im = cam.read()
        a,b = current_state
        #d = dist()
        d = 76.35
        name = str(time.time()) + "-" + str(d) + "-" + a + "-" + b + ".bmp"
        cv2.imwrite(name, im)
        print("Sending Image")
        call(['sshpass', '-f', '.psw', 'ssh', 'ncogswell@gattaca.cs.middlebury.edu', "rm img/*.bmp"])
        call(['sshpass', '-f', '.psw', 'scp', name, 'ncogswell@gattaca.cs.middlebury.edu:img/'])
        call(['rm', name])
        dist_array.append([d,current_state])
        data_size = data_size + 1
        #time.sleep(.2)

def auto_drive():
    """ driving function that takes directions from gattaca and quits when user quite on gattaca """
    global dir
    while dir != 'q':
        if len(os.listdir('direction')) > 0:
            for newinput in os.listdir('direction'):
                if dir != newinput:
                    dir = newinput
                    print("New Direction from NN:", dir_translate[newinput])
                    if not emg_break and len(dir) > 1:
                        in_drive = newinput[0]
                        in_turn = newinput[1]
                        drive(in_drive)
                        turn(in_turn)
        time.sleep(.1)

def test_drive():
    """
    driving function where the user controls the car and the directions from gattaca are
    compared to the current user inputted driving state
    """
    global dir

    thread2 = threading.Thread(target=test_get_dir, name="test get direction")
    thread2.start()
    
    while dir != 'q':
        dir = input()
        if dir in {"w","e","x"}:
            drive(dir)
        elif dir in {"a","s","d"}:
            turn(dir)
        elif dir == "z":
            full_stop()

def test_get_dir():
    """ test driving function that compares current direction and direction from gattaca """
    global accuracy
    last = "es"
    while dir != 'q':
        if len(os.listdir('direction')) > 0:
            for newinput in os.listdir('direction'):
                if last != newinput:
                    last = newinput
                    print("New Direction from NN: ", newinput)
                    curr = current_state[0]+current_state[1]
                    test_data.append((curr, last))       
                    if curr == last:
                        accuracy = (accuracy[0]+1, accuracy[1]+1)
                        print("correct")
                    else:
                        accuracy = (accuracy[0], accuracy[1]+1)
                        print("incorrect")
                    print((curr, last))       
        time.sleep(.1)
    print("Test Data:\n", test_data)
    print("Accuracy: ", accuracy[0]/accuracy[1])


# Set Global Variables
current_state = 'e','s'
dir = "es"
emg_break = False
data = []
dist_array = []
data_size = 0
dir_translate = {'ws':'Straight', 'wa':'Left', 'wd':'Right', 'es':'Stop', 'q':'Quit'}

# Testing
accuracy = (0,0) # (#currect, #total)
test_data = []

# Start Data Collection Thread
print("Warming Up Data Collection")
thread = threading.Thread(target=collect_data, name="data_collection")
thread.start()

# Start Driving
#auto_drive()
test_drive()


# GPIO cleanup
full_stop()
print ("Full Stop")
GPIO.output(TRIG, False)
print ("Waiting for Range Sensor to Settle")
time.sleep(2)
print ("Done")

GPIO.cleanup()

