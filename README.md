# MNIST Net Viewer #

This project is a python project that links to a network trained to recognize the MNIST dataset.  This project allows you visuallize serveral different layers within the network in real time as the webcam image is run through the network.

# Dependencies #

This project depends on the repo located here: https://github.com/davemers0160/mnist_dll

This repo compiles a netwrok built in the [dlib](dlib.net) framework into a .dll/.so file that can be run in other projects.  This library file is required by the code.

The python code requires the packages listed in the requirements.txt file.  To ensure the requirements are met run the following

'''
pip install -r requirements.txt
'''

# #


To run the code a little setup is required first.  

[1] Connect a webcam
[2] Create a space to write the numbers.  This requires that there be a black/dark background with a white square where the number can be written.

'''
python -m bokeh serve --show mnist_viewer.py
'''

The output should look similar to the following
![Example Viewer Output](example.png)
