from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers, optimizers
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

batch_size = 256
num_classes = 10
epochs = 50
weight_decay=1e-4
alpha=0.03

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

mean = np.mean(x_train, axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0), input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay*5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(weight_decay*10)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

adam = optimizers.adam()

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.0001)

history = model.fit(x_train, y_train, batch_size=batch_size, callbacks=[reduce_lr], epochs=epochs, validation_data=(x_test, y_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy without image augmentation')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss without image augmentation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

model.save('model_no_augmentation.h5')

