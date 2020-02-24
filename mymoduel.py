import keras
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Activation, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from keras.utils import to_categorical

def createModel():

    model = Sequential()
    #layer 1
    model.add(Conv2D(32, (5, 5), 
        padding='same', 
        activation='relu', 
        input_shape=(None,None,3))
    )
    #model.add(Dropout(0.5))
    #layer 2
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    #model.add(Dropout(0.5))
   
    #layer 3
    #model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add( GlobalAveragePooling2D())   
    #model.add(Dropout(0.25))
    # layer1
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    #layer2
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(69, activation='softmax'))   
    print("pass")
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #print(model.summary)
    return model
