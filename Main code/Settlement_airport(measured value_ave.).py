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
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
import keras.backend as K
from keras.models import load_model

#------------------- MAKE SURE THE INITIAL WEIGHT IS THE SAME EVERYTIME --------------------
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(5)
rn.seed(5)
tf.random.set_seed(5)

#-------------------- FOR REPRODUCIBILITY --------------------------------------
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return np.array(dataX), np.array(dataY)

#------------------- IMPORTING DATASET ----------------------------------------
dataset = pd.read_csv('database_airport.csv')
dataset = dataset['ave.']

#------------------- CREATING OWN INDEX FOR FLEXIBILITY-----------------------
obs = np.arange(1, len(dataset) + 1, 1) # 输出[1 2 3 … 1663 1664]
dataset = np.array(dataset).reshape(-1,1)
print('dataset_shape:', dataset.shape)
print('dataset\n', dataset)

#-------------------  DATA PREPROCESSING -------------------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)

#-------------------  SPLITTING THE DATA INTO TRAIN AND TEST SET -----------------
split_ratio = 0.8
train_size = round(len(dataset) * split_ratio)
test_size = len(dataset)-train_size
train,test = dataset[0:train_size,:],dataset[train_size:len(dataset),:]
print('train,test shape:', train.shape, test.shape)
print('training set\n', train)
print('testing set\n', test)

#--------------------  INITIALIZE THE DATA --------------------------------------
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
print('trainX\n', trainX)
print('trainY\n', trainY)
print('trainX, trainY shape after reshape:', trainX.shape, trainY.shape)
print('testX, testY shape after reshape:', testX.shape, testY.shape)

#----------------------- DEVELOPE THE LSTM MODEL --------------------------------------
# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
model = Sequential()
model.add(LSTM(64,kernel_regularizer=regularizers.l2(0.3), input_shape=(look_back, 1),
			   return_sequences=True, activation ='tanh'))
model.add(LSTM(64,activation ='tanh'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adamax', metrics=[tf.keras.metrics.MeanAbsoluteError()])
history = model.fit(trainX, trainY, validation_split = 0.1, epochs = 100, batch_size = 1)
print(model.summary())
print(history.history.keys())
# print(np.log10(history.history['val_loss']))

# ----------------------- DEVELOPE THE LSTM MODEL (with k-fold) --------------------------------------
# kf = KFold(n_splits=10)
# print(kf)
# fold = 1
# epoch = 100
# train_score = np.zeros(epoch)
# cross_val_score = np.zeros(epoch)
#
# for train_index, val_index in kf.split(trainX):
#     print("FOLD" + str(fold))
#     print("TRAIN:", train_index, "VALIDATION:", val_index)
#     X_train, X_train_validation = trainX[train_index], trainX[val_index]
#     y_train, y_train_validation = trainY[train_index], trainY[val_index]
#     # print("TRAIN_DATA\n",X_train)
#     # print("VALIDATION_DATA\n", X_train_validation)
#     # print("TRAIN_LABEL\n",y_train)
#     # print("VALIDATION_LABEL\n", y_train_validation)
#     model = Sequential()
#     model.add(LSTM(256, input_shape=(look_back, 1),
#                    return_sequences=True, activation='tanh'))  # kernel_regularizer=regularizers.l2(0.3),
#     # model.add(LSTM(16,return_sequences=True,activation ='tanh'))
#     model.add(LSTM(256, activation='tanh'))
#     model.add(Dense(1))
#     model.compile(loss='mse', optimizer='adamax', metrics=[tf.keras.metrics.MeanAbsoluteError()])
#     history = model.fit(X_train, y_train, validation_data=(X_train_validation, y_train_validation),
#                         epochs=epoch, batch_size=1)
#     if fold ==1:
#         print(model.summary())
#     cross_val_score_temp = np.log10(history.history['val_loss'])
#     print("cross_val_score_temp\n", cross_val_score_temp)
#     train_score_temp = np.log10(history.history["loss"])
#     print("train_score_temp\n", train_score_temp)
#     cross_val_score = cross_val_score + cross_val_score_temp
#     print("cross_val_score\n", cross_val_score)
#     train_score = train_score + train_score_temp
#     print("train_score\n", train_score)
#     fold = fold + 1
#
# cross_val_score_mean = cross_val_score / kf.get_n_splits(trainX)
# print("cross_val_score_mean\n", cross_val_score_mean)
# train_score_mean = train_score / kf.get_n_splits(trainX)
# print("train_score_mean\n", train_score_mean)
#
# plt.plot(train_score_mean)
# plt.plot(cross_val_score_mean)
# # plt.title('model loss')
# plt.ylabel('loss (in log10 value)')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()

#--------------------------- Visualization ----------------------------------
loss = np.log10(history.history['loss'])
val_loss = np.log10(history.history['val_loss'])
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()

#------------------------- SAVE MODEL --------------------------------------
# np.savetxt('loss_val.csv', history.history['val_loss'], delimiter = ',')
# np.savetxt('loss_train.csv', history.history['loss'], delimiter = ',')
model.save('model_measured_ave_LSTM.h5')
# model = load_model('model_measured_ave_LSTM.h5')

#------------------------- COMPARING TRAIN AND PREDICTED VALUE ------------------
train_predict = model.predict(trainX)
train_predict = scaler.inverse_transform(train_predict)
data = scaler.inverse_transform(trainY)
trainPredictout = np.vstack((data, train_predict))
print('trainPredictout:\n',trainPredictout)

#------------------------- COMPARING TEST AND PREDICTED VALUE ------------------
test_predict_transformed = model.predict(testX)
test_predict = scaler.inverse_transform(test_predict_transformed)
testY = scaler.inverse_transform(testY)
testPredictout = np.vstack((testY, test_predict))
print('testPredictout:\n',testPredictout)

#------------------------- PREDICT FUTURE VALUE --------------------------------
future = 2
future_predict = np.zeros((future, 2, 1))
future_predict[0:future, :, :] = test_predict_transformed[len(test_predict) - look_back:]
predict = np.zeros((future,1))
for j in range(0, future):
	if j == future-1:
		input_test = future_predict[j:j + 1, :, :]
		predict[j, :] = model.predict(input_test)
	else:
		input_test = future_predict[j:j + 1, :, :]
		predict[j, :] = model.predict(input_test)
		future_predict[j + 1, -1, :] = predict[j, :]
		future_predict[j+1,0,:] = future_predict[j,-1,:]

# futurePredictout = scaler.inverse_transform(predict)
# print('futurepredict:\n',futurePredictout)
futurePredictout = scaler.inverse_transform(predict)
print('futurepredict:\n',futurePredictout)

#------------------------- PLOT THE GRAPH ----------------------------------------
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2):len(dataset),:] = test_predict

z = np.zeros((len(dataset)+future,1))
futurePredictPlot = np.empty_like(z)
futurePredictPlot[:,:] = np.nan
futurePredictPlot[len(dataset):,:] = futurePredictout

year = []
for i in range(1,len(dataset)+1):
	year.append(i)
future_year = []
for i in range(1,len(dataset)+1+future):
	future_year.append(i)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(year,scaler.inverse_transform(dataset),label = 'real settlement')
ax.scatter(year,trainPredictPlot,label = 'predicted settlement(training set)')
ax.scatter(year,testPredictPlot,label = 'predicted settlement(testing set)')
ax.scatter(future_year,futurePredictPlot,label = 'predicted future settlement')
ax.set_ylabel('Settlement(m)')
ax.set_xlabel('Year')
ax.set_title("Measure Point (ave.)")
ax.legend()
plt.show()
