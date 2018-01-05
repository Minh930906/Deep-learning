#Artificial Neural Network

#Installing Theano

#Installing Tensorflow

#Installing Keras
# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values #independent variables

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])#kategorizálja a országot
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])#kategorizálja a nemet
onehotencoder = OneHotEncoder(categorical_features = [1])#create dummy variables
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Make ANN!
#Import Keras libraries and packages
#we can put these imports to the beginning
import keras
from keras.models import Sequential #required to initialize our NN
from keras.layers import Dense #required to build the layers of our ANN

#initialize the ANN
classifier = Sequential()#dont need to input anyargument google define the layers step by step.Google start with the input layer and the first hidden layer and then will adds more hidden layers and then finally will add the output layer

#Adding first input layer and first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))#add what really does is add hidden layers.output_dim -->avg((numbers of independent variable+number of output layers))-->avg(11+1)=6=(11+1)/2=6(hidden layers)
#initialize the weights
#activation-->the activation function wanna choose in our hidden layer. relu=rectifier activation function
#add second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))#removed input_dim cuz we know what to expect cuz the first hidden layer was created
#add the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))#output_dim=1 cuz we only have one node at output layer...binary outcome.(mert a független változó az categorical változó bináris outcome-mal).activation='sigmoid' mert a lehetőséget vizsgáljuk

#Compiling ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the ANN to Training set
classifier.fit(X_train,y_train,batch_size = 10,epochs=100)#batch_size --> Batch size defines number of samples that going to be propagated through the network. epochs means how many times you go through your training set.

#Part 3 - Making the prediction and evaluating the model 

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Evaluating,improving,and tuning ANN

#Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential #required to initialize our NN
from keras.layers import Dense #required to build the layers of our ANN
def build_classifier():
    classifier = Sequential()#dont need to input anyargument google define the layers step by step.Google start with the input layer and the first hidden layer and then will adds more hidden layers and then finally will add the output layer   
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))#add what really does is add hidden layers.output_dim -->avg((numbers of independent variable+number of output layers))-->avg(11+1)=6=(11+1)/2=6(hidden layers)
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))#removed input_dim cuz we know what to expect cuz the first hidden layer was created
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))#output_dim=1 cuz we only have one node at output layer...binary outcome.(mert a független változó az categorical változó bináris outcome-mal).activation='sigmoid' mert a lehetőséget vizsgáljuk
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()