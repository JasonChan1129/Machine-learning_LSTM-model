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
from keras.layers import Dropout
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Bidirectional
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.models import load_model


#  MAKE SURE THE INITIAL WEIGHT IS THE SAME EVERYTIME
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(10)
# rn.seed(10)
# tf.random.set_seed(10)

def run(file = "/Users/jasonc/Desktop/code_FYP/database_airport.csv" , measure_point = "1", model=None ,
		optimizer = "adamax", epochs = 225, validation_split = 0.1, batch_size = 1,
		split_ratio = 0.8 , timestep = 2):

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
	dataset = dataset[measure_point]

	# -------------------------------------------------------------------------------------------------------------------------
	# CREATING OWN INDEX FOR FLEXIBILITY
	obs = np.arange(1, len(dataset) + 1, 1)  # 输出[1 2 3 … 1663 1664]
	dataset = np.array(dataset).reshape(-1, 1)
	# print('dataset_shape:', dataset.shape)
	# print('dataset\n', dataset)

	# PREPARATION OF TIME SERIES DATASET
	scaler = MinMaxScaler(feature_range=(-1, 1))  # 定义scalar
	dataset = scaler.fit_transform(dataset)  # 1

	# SPLITTING THE DATA INTO TRAIN AND TEST SET
	train_size = round(len(dataset) * split_ratio)
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
	print('trainX\n', trainX)
	print('trainY\n', trainY)
	print('trainX, trainY shape after reshape:', trainX.shape, trainY.shape)
	print('testX, testY shape after reshape:', testX.shape, testY.shape)

	# 搭建LSTM MODEL
	# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
	# return_sequences 意思是在每个时间点，要不要输出output，默认的是 false（最后一个时间点输出一个值）

	model = model
	model.compile(loss='mse', optimizer=optimizer, metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
	history = model.fit(trainX, trainY, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
	# print(model.summary())

	#  list all data in history
	print(history.history.keys())  # dict_keys(['loss', 'mean_absolute_percentage_error', 'val_loss', 'val_mean_absolute_percentage_error'])
	# visualization
	# loss = np.log10(history.history['loss'])
	# val_loss = np.log10(history.history['val_loss'])
	# plt.plot(loss)
	# plt.plot(val_loss)
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.show()

	# Save Model
	np.savetxt('loss_val_for_plot_HKA.csv', history.history['val_loss'], delimiter=',')
	np.savetxt('loss_train_for_plot_HKA.csv', history.history['loss'], delimiter=',')
	model.save('HKA_GUI.h5')

	# # COMPARING TRAIN AND PREDICTED VALUE
	train_predict = model.predict(trainX)
	train_predict = scaler.inverse_transform(train_predict)
	trainY = scaler.inverse_transform(trainY)
	np.savetxt('trainY_HKA.csv', trainY, delimiter=',')
	np.savetxt('train_predict_HKA.csv', train_predict, delimiter=',')

	# data = scaler.inverse_transform(trainY)
	trainPredictout = np.vstack((trainY, train_predict))
	print('trainPredictout:\n', trainPredictout)
	#
	# # COMPARING TEST AND PREDICTED VALUE
	test_predict = model.predict(testX)
	test_predict = scaler.inverse_transform(test_predict)
	testY = scaler.inverse_transform(testY)
	np.savetxt('testY_HKA.csv', testY, delimiter=',')
	np.savetxt('test_predict_HKA.csv', test_predict, delimiter=',')

	testPredictout = np.vstack((testY, test_predict))
	print('trainPredictout:\n', testPredictout)

	# fig = plt.figure()
	# ax = fig.add_subplot(1, 1, 1)
	# ax.scatter(days, scaler.inverse_transform(dataset), color='tab:green', label='measured settlement')
	# ax.scatter(days, trainPredictPlot, color='tab:blue', label='predicted settlement(training set)')
	# ax.scatter(days, testPredictPlot, color='tab:orange', label='predicted settlement(testing set)')
	#
	# if Measure_point == 1:
	# 	ax.set_title("Measure point 1")
	# elif Measure_point == 2:
	# 	ax.set_title("Measure point 2")
	# elif Measure_point == 3:
	# 	ax.set_title("Measure point 3")
	# elif Measure_point == 4:
	# 	ax.set_title("Measure point 4")
	# elif Measure_point == 5:
	# 	ax.set_title("Measure point 5")
	#
	# ax.set_ylabel('Settlement(m)')
	# ax.set_xlabel('Days')
	# ax.legend()
	# plt.show()




