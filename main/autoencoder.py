
import datetime
import os

from keras.engine.topology import Container
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model

import keras.backend as K
import main.data as data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def autoencoder(input_size, hidden_node, tag):
    input_encode = Input(shape=(input_size,))
    input_decode = Input(shape=(hidden_node,))

    noised = GaussianNoise(0.01)(input_encode)

    encoded = Dense(hidden_node, activation='sigmoid',
                    name='encode_layer' + tag)(noised)

    decoded = Dense(input_size, activation='sigmoid',
                    name='decode_layer' + tag)(encoded)

    autoencoder = Model(input_encode, decoded)

    encode_layer = autoencoder.get_layer('encode_layer' + tag)
    encoder = Model(input_encode, encode_layer(input_encode))

    decode_layer = autoencoder.get_layer('decode_layer' + tag)
    decoder = Model(input_decode, decode_layer(input_decode))

    autoencoder.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['mae'])

    return {'autoencoder': autoencoder, 'encoder': encoder, 'decoder': decoder}


def AEfit(model, x_train, x_test):
    ae = model['autoencoder']
    encoder = model['encoder']

    ae.summary()

    history = ae.fit(x_train, x_train,
                     epochs=100,
                     batch_size=128,
                     shuffle=True,
                     validation_data=(x_test, x_test))

    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)

    return history, encoded_train, encoded_test


K.tensorflow_backend.set_session(tf.Session(
    config=tf.ConfigProto(device_count={'GPU': 1},
                          gpu_options=tf.GPUOptions(allow_growth=True))))

db = data.databank()
print('===data set start===\n')
x_train, _, x_test, _ = db.dataset_fromSQL()
print('\n===data set end===\n')
input_size = len(x_train[0]) * len(x_train[0][0])
x_train = x_train.reshape(len(x_train), input_size)
x_test = x_test.reshape(len(x_test), input_size)
x_train = sigmoid(x_train)
x_test = sigmoid(x_test)
print(x_train.shape)
print(x_test.shape)

input_encode = Input(shape=(input_size,))
input_decode = Input(shape=(100,))

model_list = list()
model_list.append(autoencoder(input_size, 500, str(len(model_list) + 1)))
model_list.append(autoencoder(500, 300, str(len(model_list) + 1)))
model_list.append(autoencoder(300, 100, str(len(model_list) + 1)))

history1, encoded_train, encoded_test = AEfit(
    model_list[0], x_train, x_test)
history2, encoded_train, encoded_test = AEfit(
    model_list[1], encoded_train, encoded_test)
history3, encoded_train, encoded_test = AEfit(
    model_list[2], encoded_train, encoded_test)

model1 = model_list[0]['autoencoder']
model2 = model_list[1]['autoencoder']
model3 = model_list[2]['autoencoder']

encode_layer = list()
decode_layer = list()

for i in range(len(model_list)):
    encode_layer.append(model_list[i]['autoencoder'].get_layer(
        'encode_layer' + str(i + 1)))
    decode_layer.append(model_list[i]['autoencoder'].get_layer(
        'decode_layer' + str(i + 1)))

o = encode_layer[0](input_encode)
o = encode_layer[1](o)
o = encode_layer[2](o)
encode_layers = Container(input_encode, o, name='encode_container')

o = decode_layer[2](input_decode)
o = decode_layer[1](o)
o = decode_layer[0](o)
decode_layers = Container(input_decode, o, name='decode_container')

o = GaussianNoise(0.01)(input_encode)
o = encode_layers(o)
o = decode_layers(o)

stacked_autoencoder = Model(input_encode, o)
encoder_sa = Model(input_encode, encode_layers(input_encode))
decoder_sa = Model(input_decode, decode_layers(input_decode))

stacked_autoencoder.summary()

stacked_autoencoder.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['mae'])

history_fine = stacked_autoencoder.fit(x_train, x_train,
                                       epochs=100,
                                       batch_size=128,
                                       shuffle=True,
                                       validation_data=(x_test, x_test))

encoded_data = encoder_sa.predict(x_test)
decoded_data = decoder_sa.predict(encoded_data)

evaluated_data = np.mean(np.abs(x_test - decoded_data), 0)

if not(os.path.exists('../result_AE')):
    os.mkdir('../result_AE')

now = datetime.datetime.now()
path = '../result_AE/result' + now.strftime("%y%m%d%H%M%S")
if not(os.path.exists(path)):
    os.mkdir(path)

stacked_autoencoder.save(path + '/autoencoder.h5')

plt.imshow(evaluated_data.reshape(20, 36))
plt.colorbar()
plt.savefig(path + '/evaluate.png')

n = 8
plt.figure(figsize=(24, 6))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(20, 36), vmin=0, vmax=1)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encode
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_data[i].reshape(10, 10), vmin=0, vmax=1)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display decode
    ax = plt.subplot(3, n, i + 1 + n * 2)
    plt.imshow(decoded_data[i].reshape(20, 36), vmin=0, vmax=1)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(path + '/pict.png')


def plot_history(history, name):

    # plot mae
    plt.figure()
    plt.plot(history.history['mean_absolute_error'], marker='.')
    plt.plot(history.history['val_mean_absolute_error'], marker='.')
    plt.title('model MAE')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.grid()
    plt.legend(['mae', 'val_mae'],
               loc='upper right')
    plt.savefig(path + '/model_MAE_' + name + '.png')

    # plot loss
    plt.figure()
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(path + '/model_loss_' + name + '.png')


plot_history(history1, 'his1')
plot_history(history2, 'his2')
plot_history(history3, 'his3')
plot_history(history_fine, 'his_fine')
