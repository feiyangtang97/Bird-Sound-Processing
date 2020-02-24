import keras
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D,Convolution2D,Activation
from keras.utils import to_categorical

def createModel():
    #print("Am I restart?")
    model = Sequential()

      # add conv layer1
    model.add(Convolution2D(
        batch_input_shape=(None,int(467/2),int(352/2),3),
        filters=32,
        kernel_size=5,# 5*5
        strides=1,
        padding='same',     # Padding method
        data_format='channels_last',
    ))

    model.add(Activation('relu'))

    #pooling
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_last',
    ))
    #model.add(Dropout(0.5))
      #add conv layer2
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_last'))
    model.add(Activation('relu'))
    #max pooling
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
    #model.add(Dropout(0.5))
     #add conv layer3
    #model.add(Convolution2D(128, 5, strides=1, padding='same', data_format='channels_last'))
    #model.add(Activation('relu'))
    #max pooling
    #model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))
    #model.add(Dropout(0.5))
    #full connect nerual
    # layer1
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #layer2
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #ouput layer
    model.add(Dense(69))
    model.add(Activation('softmax'))
    model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model
