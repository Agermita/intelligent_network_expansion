# Data manipulation
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# Data Visualiation
import matplotlib.pyplot as plt
import seaborn as sns

# System
import os

# Deep Learning
import tensorflow
from typing import Dict, List, Tuple, Sequence

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, GRU

""" prameters to be optimized

type de model models. Sequential ou autre
layers.LSTM ou layers.GRU
units

kernel_regularizer=L1L2(l1=0.05, l2=0.05)
batch_size
epochs
"""
def initialize_model(input_shape: tuple, output_length) -> models:
    #=================================================================
    # Initialize the Neural Network with random weights

    # input_shape =(X.shape[1],X.shape[2]) : the shape of input datanumber of
        #days to use for training and number of features
    # output_length = y_train.shape[1] : the number of days to be predicted
    #=================================================================

    model = models.Sequential()
    ## 1.1 - Recurrent Layer
    model.add(layers.Masking(mask_value=-10, input_shape=input_shape))
    #
    model.add(layers.GRU(units=16,
                        activation='relu',
                        return_sequences = False,
                        kernel_regularizer=L1L2(l1=0.05, l2=0.05),
                        ))


    ## 1.2 - Predictive Dense Layers

    model.add(layers.Dense(output_length, activation='linear'))

    return model


def compile_model(model: models) -> models:

    #=================================================================
    # Compile the Neural Network

    # model : initialized model
    #=================================================================

    # 2 - Compiler
    # ======================
    initial_learning_rate = 0.01

    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=1000, decay_rate=0.5)

    adam = optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='mse', optimizer=adam, metrics=["mae"])
    return model

def train_model(
        model: models,
        X: np.ndarray,
        y: np.ndarray,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split_rate=0.3
    ) -> Tuple[tensorflow.keras.Model, dict]:

    #=================================================================
    # Fit the model and return a tuple (fitted_model, history)

    # model : Compiled model
    # X : X to be used for training
    # y : y to be used for training
    # patinece : number of epochs to wait after overfiting befor early stoping
    # validation_data :  #tuple of (X_val, y_val) to be used for model validation
    # validation_split_rate : rate of validation split if validation_split is used in stead of validation_data
    #=================================================================

    # early stop the fitting after patience if val_loss is becoming worst
    es = EarlyStopping(monitor = "val_loss",
                      patience = patience,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X, y,
                        validation_data=validation_data,

                        batch_size = 16,
                        epochs = 500,
                        callbacks = [es],
                        verbose = 1)

    return model, history
