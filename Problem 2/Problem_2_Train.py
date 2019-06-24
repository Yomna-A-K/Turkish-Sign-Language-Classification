#Tensorflow
import tensorflow as tf
import tensorflow

#KERAS
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
from tensorflow.keras import backend as K

#sklearn
import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os
from numpy import *
import cv2


img_rows,img_cols = 100,100

DataSetPath = 'Turkish-Dataset\\'
NewDataSetPath = 'Turkish-Dataset-G\\'

for x in range(10):
    Path = DataSetPath + str(x)
    NewPath = NewDataSetPath + str(x)

    #for file in listing:
        #img = cv2.imread(Path + '\\' + file,2)
        #ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #cv2.imwrite(NewPath + '\\' + file,bw_img)

    listing = os.listdir(Path)
    for file in listing:
        im = Image.open(Path + '\\' + file)   
        img = im.resize((100,100))
        gray = img.convert('L')
        gray.save(NewPath + '\\' + file,"PNG")
    
    listing = os.listdir(NewPath)    
    if x==0:
        label=np.ones((size(listing),),dtype = int)
        label[0:size(listing)] = 0
        InitialImmatrix = array([array(Image.open(NewPath + '\\' + im2))
              for im2 in listing],object)
    else:
        NewLabel=np.ones((size(listing),),dtype = int)
        NewLabel[0:size(listing)] = x
        label = np.append(label,NewLabel)
        NewImmatrix = array([array(Image.open(NewPath + '\\' + im2))
              for im2 in listing],object)
        immatrix = np.vstack([InitialImmatrix,NewImmatrix])
        InitialImmatrix = immatrix
        
#make sure each index is correctly correspondant to its label
print(immatrix.shape)
print(size(label))

#Shuffle both arrays
indices = np.arange(immatrix.shape[0])
np.random.shuffle(indices)
label = label[indices]
immatrix = immatrix[indices]

train_data = [immatrix,label]

# number of output classes
nb_classes = 10
class_names=np.ones((nb_classes,),dtype = int)
for x in range(nb_classes):
    class_names[x] = x

#batch_size to train
batch_size = 200
# number of epochs to train
epochs = 70
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

#%%
(X, Y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(Y_train.shape)

# convert class vectors to hot one matrices
Y_train = tf.keras.utils.to_categorical(Y_train, nb_classes)
Y_test = tf.keras.utils.to_categorical(Y_test, nb_classes)
print(Y_train.shape)
print("label : ", Y_train[2,:])

#%%
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (nb_conv,nb_conv), padding = 'Same', activation = 'relu', input_shape = (img_rows,img_cols,1)))
model.add(MaxPool2D(pool_size = (nb_pool,nb_pool)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 16, kernel_size = (nb_conv,nb_conv), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (nb_pool,nb_pool)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (nb_conv,nb_conv), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (nb_pool,nb_pool)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (nb_conv,nb_conv), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (nb_pool,nb_pool)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#evaluation metrics
Y_pred = model.predict(X_test, batch_size=200, verbose=1)
Y_pred_bool = np.argmax(Y_pred, axis=1)
Y_test_bool = np.argmax(Y_test, axis=1)
print(sklearn.metrics.classification_report(Y_test_bool, Y_pred_bool))


#source: auto example on scikit-learn
from plot_confusion_matrix import plot_confusion_matrix

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test_bool, Y_pred_bool, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test_bool, Y_pred_bool, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#save model with its trained weights to be loaded later if needed
fname = "weights-Test-CNN2.hdf5"
model.save_weights(fname,overwrite=True)






    
        
        
      
       