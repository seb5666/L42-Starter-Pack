from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization

def get_cnn_model(p=0.5):
    
    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)
    
    nb_classes = 10
    hidden_size = 128
    hidden_size2 = 256
    initializer = 'he_uniform'

    inp = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer=initializer)(inp)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer)(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer)(conv2)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer)(conv3)
    max1 = MaxPooling2D(pool_size=(2,2))(conv4)
    drop1 = BatchNormalization()(max1)
    drop2 = Dropout(p)(drop1)
    flat = Flatten()(drop2)
    hidden1 = Dense(hidden_size, activation='relu', kernel_initializer=initializer)(flat)
    drop3 = BatchNormalization()(hidden1)
    drop4 = Dropout(p)(drop3)
    hidden2 = Dense(hidden_size2, activation='relu', kernel_initializer=initializer)(drop4)
    drop5 = BatchNormalization()(hidden2)
    drop6 = Dropout(p)(drop5)
    out = Dense(nb_classes, activation='softmax')(drop6)

    model = Model(inputs=inp, outputs=out)
    
    print(model.summary())
    return model
