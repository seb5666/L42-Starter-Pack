from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization

def get_cnn_model(p=0.5):
    
    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)
    
    nb_classes = 10
    hidden_size = 256
    
    inp = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)(inp)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
    max1 = MaxPooling2D(pool_size=(2,2))(conv3)
    drop1 = Dropout(p)(max1)
    #drop1 = BatchNormalization()(max1)
    flat = Flatten()(drop1)
    hidden1 = Dense(hidden_size, activation='relu')(flat)
    drop2 = Dropout(p)(hidden1)
    #drop2 = BatchNormalization()(hidden1)
    out = Dense(nb_classes, activation='softmax')(drop2)

    model = Model(inputs=inp, outputs=out)
    
    print(model.summary())
    return model
