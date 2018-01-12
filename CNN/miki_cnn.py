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
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,#rescale pixel values between 0 and 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),#target size is the size of the images that is  in CNN model
                                                 batch_size=32,
                                                 class_mode='binary')#your independant variable has more than 2 categories

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,#first argument-->training set.   model.fit_generator we rename to classifier cuz our CNN model name is classifier
                         steps_per_epoch=8000,#number of images we have in our training_set
                         epochs=25,#number of epochs who wanna choose to train our CNN 
                         validation_data=test_set,#that correspond to the testset on which we want to evaluate the performance of our CNN 
                         validation_steps=2000)#correspond to number of images in our testset