'''
Created on 2017/12/14

@author: AMO-TOYAMA
'''
from __future__ import print_function

import datetime
import os

import keras
from keras.engine.topology import Container
from keras.layers import Dense, Activation
from keras.models import load_model, Input, Model

import keras.backend as K
from main.data import databank
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


K.tensorflow_backend.set_session(tf.Session(
    config=tf.ConfigProto(device_count={'GPU': 1},
                          gpu_options=tf.GPUOptions(allow_growth=True))))


batch_size = 128
num_classes = 2
epochs = 100

db = databank()
print('\n===data set start===\n')
x_train, y_train, x_test, y_test = db.dataset_fromSQL()
print('\n===data set end===\n')
input_size = len(x_train[0]) * len(x_train[0][0])
x_train = x_train.reshape(len(x_train), input_size)
x_test = x_test.reshape(len(x_test), input_size)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

es_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.00001, patience=10, verbose=1, mode='min')

y_train = sigmoid(y_train)
y_train = np.transpose(np.array([y_train, 1 - y_train]))
y_test = sigmoid(y_test)
y_test = np.transpose(np.array([y_test, 1 - y_test]))

input_encode = Input(shape=(input_size,))

autoencoder = load_model('../model/autoencoder.h5',
                         custom_objects={'Container': Container})
encode_layer = autoencoder.get_layer('encode_container')
encode_layer.trainable = False

o = Activation(activation='sigmoid')(input_encode)
o = encode_layer(o)
o = Dense(100, activation='tanh')(o)
o = Dense(100, activation='tanh')(o)
o = Dense(2, activation='softmax')(o)

model = Model(input_encode, o)

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
autoencoder.save(path + '/autoencoder.h5')

result = model.predict(x_train)
plt.figure()
acc = list()
pre = list()
for i in range(len(y_train)):
    acc.append(y_train[i][0])
    pre.append(result[i][0])
plt.plot(acc)
plt.plot(pre)
plt.savefig(path + '/train.png')

result = model.predict(x_test)
plt.figure()
acc = list()
pre = list()
for i in range(len(y_test)):
    acc.append(y_test[i][0])
    pre.append(result[i][0])
plt.plot(acc)
plt.plot(pre)
plt.title('test')
plt.xlabel('day')
plt.ylabel('price')
plt.grid()
plt.legend(['actual', 'predict'],
           loc='lower right')
plt.savefig(path + '/test.png')


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
