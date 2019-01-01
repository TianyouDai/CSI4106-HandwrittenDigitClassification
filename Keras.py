import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
import time


#setting parameters 
BATCH_SIZE = 64
LR = 0.001
np.random.seed(120) 

#loading database and converting to variables
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
 
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
# convert to one_hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
 

#building model
model = Sequential()

# Conv layer 1 output shape (6,28,28)
model.add(Conv2D(6, (5, 5),
	padding='same',  
	input_shape=(1,28,28),
	))
model.add(Activation('relu'))
 
# Pooling layer 1 (max pooling) output shape (6,14,14)
model.add(MaxPooling2D(
	pool_size=(2,2),
        strides=(2, 2),
	padding='same',
	))
 
# Conv layer 2 output shape (16,14,14)
model.add(Conv2D(16, (5, 5), padding="same"))
model.add(Activation('relu'))
 
# Pooling layer 2 (max pooling) output shape (16,7,7)
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
 
 
# Fully connected layer 1 input shape (16*7*7)=(784)
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
 
# Fully connected layer 2 to shape (120) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))
 
#Compile model
adam = Adam(lr=LR)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#train
def train(EPOCH):
    
    print("-------------------Training------------------------------------")
    model.fit(X_train, Y_train, epochs=EPOCH,batch_size=BATCH_SIZE,
              verbose=2,validation_data=(X_test, Y_test))
    
    print("-------------------Testing------------------------------------")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('loss: ', score[0])
    print('accuracy: ', score[1])
   
