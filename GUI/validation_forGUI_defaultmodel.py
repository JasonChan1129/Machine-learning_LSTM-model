# IMPORTING IMPORTANT LIBRARIES
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Bidirectional
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

# # MAKE SURE THE INITIAL WEIGHT IS THE SAME EVERYTIME
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(10)
# rn.seed(10)
# tf.random.set_seed(10)

def run(year=5,measure_point=1):
    # FOR REPRODUCIBILITY
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, :])
        return np.array(dataX), np.array(dataY)

    # IMPORTING DATASET
    measure_point = measure_point
    dataset = pd.read_csv('database_airport.csv')
    if measure_point == '1':
        dataset = dataset['1']
    elif measure_point == '2':
        dataset = dataset['2']
    elif measure_point == '3':
        dataset = dataset['3']
    elif measure_point == '4':
        dataset = dataset['4']
    elif measure_point == '5':
        dataset = dataset['5']
    elif measure_point == '6':
        dataset = dataset['6']
    elif measure_point == '7':
        dataset = dataset['7']
    elif measure_point == '8':
        dataset = dataset['8']
    elif measure_point == '9':
        dataset = dataset['9']
    elif measure_point == '10':
        dataset = dataset['10']
    elif measure_point == '11':
        dataset = dataset['11']
    elif measure_point == '12':
        dataset = dataset['12']
    elif measure_point == '13':
        dataset = dataset['13']
    elif measure_point == '14':
        dataset = dataset['14']
    elif measure_point == '15':
        dataset = dataset['15']
    elif measure_point == '16':
        dataset = dataset['16']
    elif measure_point == '17':
        dataset = dataset['17']
    else:
        dataset = dataset['ave.']

    np.savetxt('dataset_GUI_predictmodel_default.csv',dataset, delimiter = ',')

    # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(dataset) + 1, 1)  # 输出[1 2 3 … 1663 1664]
    dataset = np.array(dataset).reshape(-1, 1)
    # print('dataset_shape:', dataset.shape)
    # print('dataset\n', dataset)

    # PREPARATION OF TIME SERIES DATASET
    scaler = MinMaxScaler(feature_range=(-1, 1))  # 定义scalar
    dataset = scaler.fit_transform(dataset)  # 1

    # SPLITTING THE DATA INTO TRAIN AND TEST SET
    train_size = 21
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # print('train,test shape:', train.shape, test.shape)
    # print('training set\n', train)
    # print('testing set\n', test)

    look_back = 2
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
    testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
    # print('trainX\n', trainX)
    # print('trainY\n', trainY)
    # print('trainX, trainY shape after reshape:', trainX.shape, trainY.shape)
    # print('testX, testY shape after reshape:', testX.shape, testY.shape)

    model = load_model('model_measured_ave_LSTM.h5')

    # COMPARING TRAIN AND PREDICTED VALUE
    train_predict = model.predict(trainX)
    train_predict = scaler.inverse_transform(train_predict)
    np.savetxt('train_predict_predictmodel_default.csv',train_predict, delimiter = ',')
    # data = scaler.inverse_transform(trainY)
    # trainPredictout = np.vstack((data, train_predict))
    # print('trainPredictout:\n', trainPredictout)

    # COMPARING TEST AND PREDICTED VALUE
    test_predict = model.predict(testX)
    test_predict_for_future = model.predict(testX)
    test_predict = scaler.inverse_transform(test_predict)
    np.savetxt('test_predict_predictmodel_default.csv',test_predict, delimiter = ',')
    # testY = scaler.inverse_transform(testY)
    # testPredictout = np.vstack((testY, test_predict))
    # print('testPredictout:\n', testPredictout)

    future = year-2019
    future_predict = np.zeros((future, 2, 1))
    future_predict[0:1, :, :] = test_predict_for_future[len(test_predict)-look_back:]
    predict = np.zeros((future, 1))
    for j in range(0, future):
        if j == future - 1:
            input_test = future_predict[j:j + 1, :, :]
            predict[j, :] = model.predict(input_test)
        else:
            input_test = future_predict[j:j + 1, :, :]
            predict[j, :] = model.predict(input_test)
            future_predict[j + 1, -1, :] = predict[j, :]
            future_predict[j + 1, 0, :] = future_predict[j, -1, :]

    futurePredictout = scaler.inverse_transform(predict)
    print('futurepredict:\n', futurePredictout)
    np.savetxt('future_predict_predictmodel_default.csv',futurePredictout, delimiter = ',')

    # PLOTTING
    # Visualization
    # trainPredictPlot = np.empty_like(dataset)
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    #
    # testPredictPlot = np.empty_like(dataset)
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(train_predict) + (look_back * 2):len(dataset), :] = test_predict
    #
    # z = np.zeros((len(dataset) + future, 1))
    # futurePredictPlot = np.empty_like(z)
    # futurePredictPlot[:, :] = np.nan
    # futurePredictPlot[len(dataset):, :] = futurePredictout
    #
    # plt.plot(scaler.inverse_transform(dataset), label='real settlement')
    # plt.plot(trainPredictPlot, label='predicted settlement(training set)')
    # plt.plot(testPredictPlot, label='predicted settlement(testing set)')
    # plt.plot(futurePredictPlot, label='predicted future settlement')
    # plt.title("Average Value")
    # plt.ylabel('Settlement(m)')
    # plt.xlabel('Year')
    # plt.legend()
    # plt.show()










