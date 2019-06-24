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

import numpy as np
from PIL import Image
import os
from numpy import *

img_rows,img_cols = 100,100
Path = 'Test-Images'
NewPath = 'Test-Images-Pre-Processed'
listing = os.listdir(Path) 
num_samples=size(listing)
print(num_samples)

for file in listing:
    im = Image.open(Path + '\\' + file)   
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here           
    gray.save(NewPath +'\\' +  file, "PNG")

immatrix = array([array(Image.open(NewPath + '\\' + im2))
              for im2 in listing],object)
#%%
X_test = immatrix.reshape(immatrix.shape[0], img_rows, img_cols, 1)

X_test = X_test.astype('float32')

X_test /= 255

# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 5

#try new model
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

fname = "weights-Test-CNN2.hdf5"
model.load_weights(fname)
print("loaded")

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

predictions = model.predict_classes(X_test)

file = open('indexs.txt', 'w+')

for item in predictions:
  file.write("%s\n" % item)

file.close()
