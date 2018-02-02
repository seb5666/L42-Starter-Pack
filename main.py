# %load main.py
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop

from data import load_data
from model import get_model
from cnn_model import get_cnn_model

from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
nb_epoch = 30

# Load data
(X_train, y_train, X_test, y_test) = load_data()

# Load and compile model
model = get_cnn_model()

model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])

#model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy:", score[1])
