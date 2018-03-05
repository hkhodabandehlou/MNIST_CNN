# MNIST_CNN
#MNIST classification  with convolutional neural networks
#The code gets 99.8% accuracy on the test data

from __future__ import print_function
import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GaussianNoise
from keras import backend as K

now = datetime.datetime.now

img_rows,img_cols=28,28
num_classes=10
pool_size=2
filters=32
kernel_size=3
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('###',x_train.shape)
x_train=x_train.reshape((x_train.shape[0],)+input_shape)
x_test=x_test.reshape((x_test.shape[0],)+input_shape)
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

print('$$$$$$$',x_train.shape,y_train.shape)

model=Sequential()
model.add(Conv2D(filters,kernel_size,padding='valid',activation='relu',input_shape=input_shape))
model.add(Conv2D(filters,kernel_size,activation='relu'))
model.add(Conv2D(filters,kernel_size,activation='relu'))
model.add(Conv2D(filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Conv2D(filters,kernel_size,activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(units=200,activation='tanh'))
model.add(GaussianNoise(0.02))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=128)

score=model.evaluate(x_test,y_test)
print(score)
