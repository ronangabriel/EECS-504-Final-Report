read_data.py converts the original .dat measurements to .csv files. 
It also contains a function to convert all the files to test and train tensors of the appropriate size.

create_data.py splits the original data into test and train datasets.
This script adds Gaussian noise to each image in order to generate many artificial images.

cycleGanExp.py is the main program to run in order to reproduce results. This script was adapted from the horses2zebras tutorial: https://www.tensorflow.org/tutorials/generative/cyclegan

Note that the import pix2pix.py was changed to take in 1 channel input instead of 3 channel.
