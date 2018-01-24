from __future__ import print_function

from keras.layers import Dense
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from main.data import databank

import numpy as np
import os
import datetime


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


K.tensorflow_backend.set_session(tf.Session(
    config=tf.ConfigProto(device_count={'GPU': 1},
                          gpu_options=tf.GPUOptions(allow_growth=True))))

batch_size = 128
num_classes = 2
epochs = 100

db = databank()
x_train, y_train, x_test, y_test = db.dataset_fromCSV()

input_size = len(x_train[0])
x_train = x_train.reshape(len(x_train), input_size)
x_test = x_test.reshape(len(x_test), input_size)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

tmplist = list()
for i in range(len(y_train)):
    y_train[i] = sigmoid(y_train[i])
    tmplist.append(np.array([y_train[i], 1 - y_train[i]]))
y_train = np.array(tmplist)

tmplist = list()
for i in range(len(y_test)):
    y_test[i] = sigmoid(y_test[i])
    tmplist.append(np.array([y_test[i], 1 - y_test[i]]))
y_test = np.array(tmplist)

model = Sequential()
model.add(Dense(input_size, activation='tanh', input_shape=(input_size,)))
model.add(Dense(input_size, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mape'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test mape:', score[1])

if not(os.path.exists('../result')):
    os.mkdir('../result')

now = datetime.datetime.now()
path = '../result/result' + now.strftime("%y%m%d%H%M%S")
if not(os.path.exists(path)):
    os.mkdir(path)

with open(path + '/memo.txt', 'w') as f:
    f.write('batchsize:' + str(batch_size) + '\n')
    f.write('epochs:' + str(epochs) + '\n')
    f.write('loss:' + str(score[0]) + '\n')
    f.write('mape:' + str(score[1]) + '\n')

model.save(path + '/model.h5')

result = model.predict(x_train)
acc = list()
pre = list()
for i in range(len(y_train)):
    acc.append(y_train[i][0])
    pre.append(result[i][0])
plt.figure()
plt.plot(acc)
plt.plot(pre)

result = model.predict(x_test)
acc = list()
pre = list()
for i in range(len(y_test)):
    acc.append(y_test[i][0])
    pre.append(result[i][0])
plt.figure()
plt.plot(acc)
plt.plot(pre)
plt.title('test')
plt.xlabel('day')
plt.ylabel('price')
plt.grid()
plt.legend(['actual', 'predict'],
           loc='lower right')
plt.savefig(path + '/test.png')


# ## add to show graph
def plot_history(history):

    # plot mape
    plt.figure()
    plt.plot(history.history['mean_absolute_percentage_error'], marker='.')
    plt.plot(history.history['val_mean_absolute_percentage_error'], marker='.')
    plt.title('model mape')
    plt.xlabel('epoch')
    plt.ylabel('mean_absolute_percentage_error')
    plt.grid()
    plt.legend(['mean_absolute_percentage_error', 'val_mean_absolute_percentage_error'],
               loc='upper right')
    plt.savefig(path + '/model_mape.png')

    # plot loss
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(path + '/model_loss.png')


plot_history(history)

plt.figure(1)
plt.show()
