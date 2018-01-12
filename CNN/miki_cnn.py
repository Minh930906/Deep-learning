# Part 1 - Building the CNN

#Importing the Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3),input_shape=(64, 64, 3), activation = 'relu'))#32 feature detector,-->3 rows and 3 columns,input_shape-->first 2 argument the dimension of the 2d arrays,and the 3rd argument is the number of the channel 
#we used rectifier('relu') because we want to increase nonlinearity in our image or in our CNN.Rectifier breaks up linearity

# Step 2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))#pool_size we take 2x2 in general,when we apply maxpooling on our feature maps which created after the first step,most of the time we take 2by2 because we dont wanna lose the information.
#With 2by2 we keep information and we're still being precise on where we have high number is in the feature mapthat detects some specific features of the input image

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))#output node

# Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics =['accuracy'] )#binary because we only have binary outcome.If we had more than 2 outcome then we will choose categorical cross entropy

# Step 2 Fit our CNN to our images