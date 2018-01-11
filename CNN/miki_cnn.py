# Part 1 - Building the CNN

#Importing the Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
