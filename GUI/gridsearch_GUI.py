# IMPORTING IMPORTANT LIBRARIES
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Bidirectional
import seaborn as sb
import keras.backend as K
from keras.callbacks import LearningRateScheduler

def run(file = "/Users/jasonc/Desktop/code_FYP/database_airport.csv" ,split_ratio = 0.8, validation_split = 0.1, timestep = 2, layers = 3,
        layers_min = 1, neu_max = 100,neu_min = 10, interval = 5, iteration = 10,optimizer = 'Adamax', loss_fun = 'mse', activation = 'tanh',
        batch_size = 1, epochs = 125, regularizer = 0.5, dropout = 0.1):
    N_layer = layers
    N_layer_min = layers_min
    N_neuron = neu_max
    N_iteration = iteration
    interval = interval
    count = 0
    count_2 = 0
    No_of_layers = int((N_layer+1)-(N_layer_min))
    Neu_per_layer = int(((neu_max-neu_min)/interval)+1)
    results_train = np.zeros((No_of_layers,Neu_per_layer))
    results_test = np.zeros((No_of_layers, Neu_per_layer))

    MAE_train_temp = np.zeros(N_iteration)
    MAE_test_temp = np.zeros(N_iteration)

    # FOR REPRODUCIBILITY
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, :])
        return np.array(dataX), np.array(dataY)

    # IMPORTING DATASET
    dataset = pd.read_csv(file)
    # dataset = dataset['ave.']

    # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(dataset) + 1, 1)  # 输出[1 2 3 … 1663 1664]
    dataset = np.array(dataset).reshape(-1, 1)
    # print('dataset_shape:', dataset.shape)
    print('dataset\n', dataset)

    # PREPARATION OF TIME SERIES DATASET
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 定义scalar
    dataset = scaler.fit_transform(dataset)  # 1

    # SPLITTING THE DATA INTO TRAIN AND TEST SET
    train_size = round(len(dataset)*split_ratio)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # print('train,test shape:', train.shape, test.shape)
    # print('training set\n', train)
    # print('testing set\n', test)

    look_back = timestep
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
    # print('trainX\n', trainX)
    # print('trainY\n', trainY)
    # print('trainX, trainY shape after reshape:', trainX.shape, trainY.shape)
    # print('testX, testY shape after reshape:', testX.shape, testY.shape)


    for i in range(N_layer_min, N_layer + 1):
        for j in range(neu_min, N_neuron + 1, interval):
            for z in range(N_iteration):
                if i == 1:
                    model = Sequential()
                    model.add(LSTM(j, kernel_regularizer=regularizers.l2(regularizer), input_shape=(look_back, 1), activation=activation))
                    model.add(Dropout(dropout))
                    model.add(Dense(1))
                    model.compile(loss=loss_fun, optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
                    history = model.fit(trainX, trainY, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

                elif i == 2:
                    model = Sequential()
                    model.add(LSTM(j, kernel_regularizer=regularizers.l2(regularizer),
                                   input_shape=(look_back, 1), return_sequences=True, activation=activation))
                    model.add(Dropout(dropout))
                    model.add(LSTM(j,activation=activation))
                    model.add(Dropout(dropout))
                    model.add(Dense(1))
                    model.compile(loss=loss_fun, optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
                    history = model.fit(trainX, trainY, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

                else:
                    model = Sequential()
                    model.add(LSTM(j, kernel_regularizer=regularizers.l2(regularizer),
                                   input_shape=(look_back, 1), return_sequences=True, activation=activation))
                    model.add(Dropout(dropout))
                    model.add(LSTM(j, return_sequences=True, activation=activation))
                    model.add(Dropout(dropout))
                    model.add(LSTM(j, activation=activation))
                    model.add(Dropout(dropout))
                    model.add(Dense(1))
                    model.compile(loss=loss_fun, optimizer= optimizer, metrics=[tf.keras.metrics.MeanAbsoluteError()])
                    history = model.fit(trainX, trainY, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
                # ---------------------------------------------------------------------------------------------------------------------------
                train_predict = model.predict(trainX)
                trainY_predict = scaler.inverse_transform(train_predict)
                data = scaler.inverse_transform(trainY)
                trainY_measure = data

                MAE_train = np.mean(np.abs(trainY_measure - trainY_predict))

                MAE_train_temp[z] = MAE_train
                print("layers",i)
                print("neurons",j)

                print("MAE_train_temp:\n", MAE_train_temp)


                # ---------------------------------------------------------------------------------------------------------------------------
                test_predict = model.predict(testX)
                testY_predict = scaler.inverse_transform(test_predict)
                data_test = scaler.inverse_transform(testY)
                testY_measure = data_test

                MAE_test = np.mean(np.abs(testY_measure - testY_predict))

                MAE_test_temp[z] = MAE_test

                print("MAE_test_temp:\n", MAE_test_temp)

            # ---------------------------------------------------------------------------------------------------------------------------
            MAE_train_mean = np.mean(MAE_train_temp)

            print("MAE_train_mean:\n", MAE_train_mean)

            MAE_train_temp = np.zeros(N_iteration)

            neurons_position = count
            layers_position = count_2
            results_train[layers_position,neurons_position] = MAE_train_mean

            # ---------------------------------------------------------------------------------------------------------------------------
            MAE_test_mean = np.mean(MAE_test_temp)
            print("MAE_test_mean:\n", MAE_test_mean)


            MAE_test_temp = np.zeros(N_iteration)


            results_test[layers_position,neurons_position] = MAE_test_mean


            print("results_train:\n", results_train)
            print("results_test:\n", results_test)

            count = count + 1

        count = 0
        count_2 = count_2 +1
    # ---------------------------------------------------------------------------------------------------------------------------
    results_test = results_test.transpose()
    results_train = results_train.transpose()
    print("results_train(transpose):\n", results_train)
    print("results_test(transpose):\n", results_test)
    np.savetxt('Results_Train_GUI.csv', results_train, fmt='%.05f', delimiter=',')
    np.savetxt('Results_Test_GUI.csv', results_test, fmt='%.05f', delimiter=',')





