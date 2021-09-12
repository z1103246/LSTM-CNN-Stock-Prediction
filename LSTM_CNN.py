import collections.abc

import os
import sys

import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import metrics


def is_listlike(obj):
    if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
        return True
    return False


def residual_block(layer_in, n_conv, n_filters, kernel_size, stride, pad):
    if n_conv > 1:
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for i in range(n_conv)]
        if isinstance(stride, int):
            stride = [stride for i in range(n_conv)]
        if isinstance(pad, int):
            pad = [pad for i in range(n_conv)]
        if is_listlike(kernel_size) and is_listlike(stride) and len(kernel_size) == n_conv and len(stride) == n_conv and len(pad) == n_conv:
            pass
        else:
            print(
                "Number of kernel size and stride does not match the number of conv layer.")
            sys.exit()

    shortcut = Conv2D(n_filters, (1, 1), max(stride), 'valid',
                      activation='relu', kernel_initializer='he_normal')(layer_in)

    for i in range(n_conv):
        if i == 0:
            padding = ZeroPadding2D(padding=(pad[i]))(layer_in)
        else:
            padding = ZeroPadding2D(padding=(pad[i]))(conv)
        conv = Conv2D(n_filters, kernel_size[i], stride[i], "valid",
                      activation='relu', kernel_initializer='he_normal')(padding)
        bn = BatchNormalization()(conv)

    layer_out = add([conv, shortcut])
    layer_out = Activation('relu')(layer_out)
    return layer_out


def construct_cnn(look_forward, w=112, h=112, d=3, fc=True, single_output=False):
    input = Input(shape=(w, h, d))
    pad = ZeroPadding2D(padding=(3, 3))(input)
    conv1 = Conv2D(32, (7, 7), padding='valid', strides=2)(pad)
    pad2 = ZeroPadding2D(padding=(1, 1))(conv1)
    max_pool = MaxPooling2D((3, 3), padding='valid', strides=2)(pad2)
    res_conv1 = residual_block(
        max_pool, 3, 128, [(1, 1), (3, 3), (1, 1)], 1, [0, 1, 0])
    res_conv2 = residual_block(res_conv1, 3, 256, [(
        1, 1), (3, 3), (1, 1)], [2, 1, 1], [0, 1, 0])
    res_conv3 = residual_block(res_conv2, 3, 512, [(
        1, 1), (3, 3), (1, 1)], [2, 1, 1], [0, 1, 0])
    avg_pool = AveragePooling2D((7, 7), padding='valid')(res_conv3)

    if fc:
        fc = Dense(500, activation='relu')(avg_pool)
        drop = Dropout(0.5)(fc)
        fc2 = Dense(100, activation='relu')(drop)
        drop2 = Dropout(0.5)(fc2)
        fc3 = Dense(25, activation='relu')(drop2)
        drop3 = Dropout(0.5)(fc3)
        if single_output:
            for i in range(look_forward):
                pred = Dense(1, activation='linear',
                             name='cnn_output_' + str(i))(drop3)
        else:
            pred = Dense(1, activation='linear',
                         name='cnn_output_t_' + str(look_forward))(drop3)
        model = Model(inputs=input, outputs=pred, name="cnn")
    else:
        model = Model(inputs=input, outputs=avg_pool, name="cnn")

    return model


def construct_lstm(look_forward, look_back, num_features, fc=True, single_output=False):
    input = Input(shape=(look_back, num_features))
    lstm1 = LSTM(3, return_sequences=True)(input)
    lstm2 = LSTM(6)(lstm1)

    if fc:
        fc = Dense(500, activation='relu')(lstm2)
        drop = Dropout(0.5)(fc)
        fc2 = Dense(100, activation='relu')(drop)
        drop2 = Dropout(0.5)(fc2)
        fc3 = Dense(25, activation='relu')(drop2)
        drop3 = Dropout(0.5)(fc3)
        if not single_output:
            for i in range(look_forward):
                pred = Dense(1, activation='linear',
                             name='lstm_output_' + str(i))(drop3)
        else:
            pred = Dense(1, activation='linear',
                         name='lstm_output_t_' + str(look_forward))(drop3)
        model = Model(inputs=input, outputs=pred, name="lstm")
    else:
        model = Model(inputs=input, outputs=lstm2, name="lstm")

    return model


def construct_lstm_cnn(look_forward, look_back=30, compile=True, single_output=False):
    cnn = construct_cnn(look_forward, fc=False)
    cnn_flatten = Flatten()(cnn.output)
    lstm = construct_lstm(look_forward, look_back, 2, fc=False)

    # Merged layer
    merged_outputs = []
    cnn_lstm = concatenate([cnn_flatten, lstm.output])
    fc_merged = Dense(500, activation='relu')(cnn_lstm)
    drop_merged = Dropout(0.5)(fc_merged)
    fc2_merged = Dense(100, activation='relu')(drop_merged)
    drop2_merged = Dropout(0.5)(fc2_merged)
    fc3_merged = Dense(25, activation='relu')(drop2_merged)
    drop3_merged = Dropout(0.5)(fc3_merged)
    if not single_output:
        for i in range(look_forward):
            pred_merged = Dense(1, activation='linear',
                                name='merged_output_' + str(i))(drop3_merged)
            merged_outputs.append(pred_merged)
    else:
        pred_merged = Dense(
            1, activation='linear', name='merged_output_t_' + str(look_forward))(drop3_merged)
        merged_outputs.append(pred_merged)

    # Auxiliary branch for cnn
    cnn_outputs = []
    fc_cnn = Dense(500, activation='relu')(cnn_flatten)
    drop_cnn = Dropout(0.5)(fc_cnn)
    fc2_cnn = Dense(100, activation='relu')(drop_cnn)
    drop2_cnn = Dropout(0.5)(fc2_cnn)
    fc3_cnn = Dense(25, activation='relu')(drop2_cnn)
    drop3_cnn = Dropout(0.5)(fc3_cnn)
    if not single_output:
        for i in range(look_forward):
            pred_cnn_aux = Dense(1, activation='linear',
                                 name='cnn_aux_output_' + str(i))(drop3_cnn)
            cnn_outputs.append(pred_cnn_aux)
    else:
        pred_cnn_aux = Dense(
            1, activation='linear', name='cnn_aux_output_t_' + str(look_forward))(drop3_cnn)
        cnn_outputs.append(pred_cnn_aux)

    # Auxiliary branch for lstm
    lstm_outputs = []
    fc_lstm = Dense(500, activation='relu')(lstm.output)
    drop_lstm = Dropout(0.5)(fc_lstm)
    fc2_lstm = Dense(100, activation='relu')(drop_lstm)
    drop2_lstm = Dropout(0.5)(fc2_lstm)
    fc3_lstm = Dense(25, activation='relu')(drop2_lstm)
    drop3_lstm = Dropout(0.5)(fc3_lstm)
    if not single_output:
        for i in range(look_forward):
            pred_lstm_aux = Dense(1, activation='linear',
                                  name='lstm_aux_output_' + str(i))(drop3_lstm)
            lstm_outputs.append(pred_lstm_aux)
    else:
        pred_lstm_aux = Dense(
            1, activation='linear', name='lstm_aux_output_' + str(look_forward))(drop3_lstm)
        lstm_outputs.append(pred_lstm_aux)

    # Final model with three branches
    model = Model(inputs=[cnn.input, lstm.input], outputs=(
        merged_outputs + cnn_outputs + lstm_outputs), name="lstm-cnn")
    if compile:
        if not single_output:
            loss_weights = [1 for i in range(
                look_forward)] + [0.2 for i in range(look_forward)] + [0.2 for i in range(look_forward)]
        else:
            loss_weights = [1, 0.2, 0.2]
        model.compile(optimizer='adam', loss=rmse_loss, loss_weights=loss_weights, metrics=[metrics.RootMeanSquaredError(name='rmse'),
                                                                                            metrics.MeanAbsolutePercentageError(
                                                                                                name='mape'),
                                                                                            metrics.MeanAbsoluteError(name='mae')])
    return model


def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def construct_inference_model(model):
    return Model(model.input, model.output[0])


def predict_price(model: Model, x):
    pred = model.predict(x)
    return np.exp(pred)


if __name__ == "__main__":
    cnn_model = construct_cnn(5)
    cnn_model.summary()

    lstm_model = construct_lstm(5, 30, 2)
    lstm_model.summary()

    lstm_cnn_model = construct_lstm_cnn(5, 30)
    lstm_cnn_model.summary()
