# IMPORTING IMPORTANT LIBRARIES
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Bidirectional, Dropout
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

# ------------------- MAKE SURE THE INITIAL WEIGHT IS THE SAME EVERYTIME --------------------
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(5)
rn.seed(5)
tf.random.set_seed(5)

# -------------------- FOR REPRODUCIBILITY --------------------------------------
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# ----------------- IMPORTING DATASET ------------------------------------------
dataset = pd.read_csv('HKA_data_interpolated.csv')
days = dataset["days"]

Measure_point = 1
if Measure_point == 1:
    dataset = dataset['1']
elif Measure_point == 2:
    dataset = dataset['2']
elif Measure_point == 3:
    dataset = dataset['3']
elif Measure_point == 4:
    dataset = dataset['4']
elif Measure_point == 5:
    dataset = dataset['5']

# ------------------- CREATING OWN INDEX FOR FLEXIBILITY-----------------------
obs = np.arange(1, len(dataset) + 1, 1)  # 输出[1 2 3 … 1663 1664]
dataset = np.array(dataset).reshape(-1, 1)
print('dataset_shape:', dataset.shape)
print('dataset\n', dataset)

# -------------------  DATA PREPROCESSING -------------------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

# --------------------  INITIALIZE THE DATA --------------------------------------
look_back = 2
trainX, trainY = create_dataset(dataset, look_back)
# testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
# testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
print('trainX\n', trainX)
print('trainY\n', trainY)
print('trainX, trainY shape after reshape:', trainX.shape, trainY.shape)
# print('testX, testY shape after reshape:', testX.shape, testY.shape)

# ----------------------------- LOAD MODEL ----------------------------------------
model = load_model('model_measured_ave_LSTM.h5')

# ------------------------- COMPARING TRAIN AND PREDICTED VALUE --------------------
train_predict = model.predict(trainX)
train_predict = scaler.inverse_transform(train_predict)
data = scaler.inverse_transform(trainY)
trainPredictout = np.vstack((data, train_predict))
mae_train = mean_absolute_error(data, train_predict)
print('trainPredictout:\n', trainPredictout)
print("mae_train:\n", mae_train)
np.savetxt('HKA_Train_Predict.csv', train_predict, fmt='%.05f', delimiter=',')

# ------------------------------------------------------------------------------------
future = 2
future_predict = np.zeros((future, look_back, 1))
future_predict[0:look_back - 1, :, :] = dataset[(len(dataset) - look_back):]
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

# ------------------------- PLOT THE GRAPH ----------------------------------------
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

z = np.zeros((len(dataset)+future,1))
futurePredictPlot = np.empty_like(z)
futurePredictPlot[:,:] = np.nan
futurePredictPlot[len(dataset):,:] = futurePredictout

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(days, scaler.inverse_transform(dataset), color='tab:green', label='measured settlement')
ax.scatter(days, trainPredictPlot, color='tab:orange', label='predicted settlement')
future_year = []
for i in range(203, 203+((len(dataset) - 1) * 20) + future * 20 +1,20):
    future_year.append(i)
ax.scatter(future_year, futurePredictPlot,color='tab:red',label='predicted future settlement')

if Measure_point == 1:
    ax.set_title("Measure point 1")
elif Measure_point == 2:
    ax.set_title("Measure point 2")
elif Measure_point == 3:
    ax.set_title("Measure point 3")
elif Measure_point == 4:
    ax.set_title("Measure point 4")
elif Measure_point == 5:
    ax.set_title("Measure point 5")
ax.set_ylabel('Settlement(m)')
ax.set_xlabel('Contract day')
ax.legend()
plt.show()
