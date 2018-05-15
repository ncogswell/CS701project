Gattaca files:

- driver_gattaca.py: communicates with the Pi to send directions to the car
    Two threads: One constantly waits for user input to quit
    The other reads image sent from Pi, run model on input, and sends prediction to Pi
    
- train.py: trains the neural network and saves a model to be run with the car


Pi File:

- drive_pi.py: allows us to drive the car from neural network input, all computed on Gattaca
    Two threads: One constantly takes pictures and sends to Gattaca
    The other constantly checks for a new direction from Gattaca and drives accordingly

