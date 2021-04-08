import sys
import os, csv
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from make_model_import import Ui_make_model
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
import LSTM_model_import, validation_forGUI, validation_forGUI_HKA, validation_forGUI_defaultmodel, gridsearch_GUI, HKA_model_import,validation_forGUI_defaultmodel_HKA
from keras.models import load_model
import seaborn as sb

class ModelRunnerUi(QWidget, Ui_make_model):
    def __init__(self, parent=None):
        super(ModelRunnerUi, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Deep Learning Application for Settlement Prediction (Kansai Airport)")

        self.tab_3.setCurrentIndex(self.tab_3.indexOf(self.tab_1))
        self.tab_3.setTabText(self.tab_3.indexOf(self.tab_1),"Home Page")
        self.tab_3.setTabText(self.tab_3.indexOf(self.tab_2), "Model Configuration")
        self.tab_3.setTabText(self.tab_3.indexOf(self.special_three), "Train Model")
        self.tab_3.setTabText(self.tab_3.indexOf(self.tab_4),  "Model Result")
        self.tab_3.setTabText(self.tab_3.indexOf(self.tab_5),"Test Model")
        self.tab_3.setTabText(self.tab_3.indexOf(self.tab_6), "Test Model (Default Model)")

        self.pushButton_start.clicked.connect(self.start)
        self.pushButton_browse.clicked.connect(self.get_file)
        self.pushButton_browse_2.clicked.connect(self.get_file_2)
        self.pushButton_3.clicked.connect(self.run)
        self.pushButton_save.clicked.connect(self.save)
        self.pushButton_load.clicked.connect(self.load)
        self.pushButton_neu.clicked.connect(self.select_neurons)
        self.pushButton_5.clicked.connect(self.get_measure)
        self.pushButton_RM.clicked.connect(self.run_model)
        self.pushButton.clicked.connect(self.select_year)
        self.pushButton_4.clicked.connect(self.select_day)
        self.pushButton_predict.clicked.connect(self.predict)
        self.pushButton_predict_3.clicked.connect(self.predict_HKA)
        self.pushButton_predict_2.clicked.connect(self.predict2)
        self.pushButton_predict_4.clicked.connect(self.predict_HKA_2)
        self.pushButton_2.clicked.connect(self.select_year_2)
        self.pushButton_6.clicked.connect(self.select_day_2)

        self.pushButton_sf1.clicked.connect(self.save_fig)
        self.pushButton_sf2.clicked.connect(self.save_fig_2)
        self.pushButton_sf1_3.clicked.connect(self.save_fig_3)
        self.pushButton_sf1_2.clicked.connect(self.save_fig_4)
        self.pushButton_home1.clicked.connect(self.home)
        self.pushButton_home2.clicked.connect(self.home_2)
        self.pushButton_home1_3.clicked.connect(self.home_3)
        self.pushButton_home1_2.clicked.connect(self.home_4)
        self.pushButton_zoom1.clicked.connect(self.zoom)
        self.pushButton_zoom2.clicked.connect(self.zoom_2)
        self.pushButton_zoom1_3.clicked.connect(self.zoom_3)
        self.pushButton_zoom1_2.clicked.connect(self.zoom_4)
        self.pushButton_move.clicked.connect(self.move)
        self.pushButton_move2.clicked.connect(self.move_2)
        self.pushButton_move_3.clicked.connect(self.move_3)
        self.pushButton_move_2.clicked.connect(self.move_4)
        self.pushButton_7.clicked.connect(self.save_csv_loss)
        self.pushButton_8.clicked.connect(self.save_csv_train_result)
        self.pushButton_9.clicked.connect(self.save_csv_predict_KIA)
        self.pushButton_10.clicked.connect(self.save_csv_predict_HKA)
        self.pushButton_11.clicked.connect(self.save_csv_predict_default_KIA)
        self.pushButton_12.clicked.connect(self.save_csv_predict_default_HKA)

    def start(self):
        self.tab_3.setCurrentIndex(self.tab_3.indexOf(self.tab_2))

    def select_neurons(self):
        num, ok = QInputDialog.getInt(self, 'Neurons Dialog', 'Neurons:', min=10, max=600)
        if ok:
            self.line_neu.setText(str(num))

    def select_year(self):
        num, ok = QInputDialog.getInt(self, 'Year Input Dialog', 'Year:', min=2020, max=2100)
        if ok:
            self.lineEdit_PY.setText(str(num))

    def select_day(self):
        num, ok = QInputDialog.getInt(self, 'Day Input Dialog', 'Day:', min=1143, max=3143,step=20)
        if ok:
            self.lineEdit_PY_3.setText(str(num))

    def select_day_2(self):
        num, ok = QInputDialog.getInt(self, 'Day Input Dialog', 'Day:', min=1143, max=3143,step=20)
        if ok:
            self.lineEdit_PY_4.setText(str(num))

    def select_year_2(self):
        num, ok = QInputDialog.getInt(self, 'Year Input Dialog', 'Year:', min=2020, max=2100)
        if ok:
            self.lineEdit_PY_2.setText(str(num))
    def get_file(self):
        self.file_name, _ = QFileDialog.getOpenFileName(self , "Load Data File",'.',"CSV File (*.csv)")
        self.lineEdit_file_name.setText(self.file_name)

    def get_file_2(self):
        self.file_name_2, _ = QFileDialog.getOpenFileName(self, "Load Data File",'.',"CSV file (*.csv)")
        self.lineEdit_file_name_2.setText(self.file_name_2)


    def run(self):
        self.split_ratio_GRID = float(self.lineEdit_2.text())
        self.validation_split_GRID = float(self.lineEdit_3.text())
        self.timestep_GRID = int(self.lineEdit_4.text())
        self.hiddenlayers_GRID = int(self.lineEdit_16.text())
        self.hiddenlayers_min_GRID = int(self.lineEdit_20.text())
        self.neurons_max_GRID = int(self.lineEdit_5.text())
        self.neurons_min_GRID = int(self.lineEdit_7.text())
        self.interval_GRID = int(self.lineEdit_9.text())
        self.iteration_GRID = int(self.lineEdit_11.text())
        self.optimizer_GRID = str(self.comboBox.currentText())
        self.loss_function_GRID = str(self.comboBox_2.currentText())
        self.activation_GRID = str(self.comboBox_3.currentText())
        self.batch_size_GRID = int(self.lineEdit_10.text())
        self.epochs_GRID = int(self.lineEdit_18.text())
        self.regularizer_GRID = float(self.lineEdit.text())
        self.dropout_GRID = float(self.lineEdit_19.text())


        gridsearch_GUI.run(split_ratio = self.split_ratio_GRID, validation_split = self.validation_split_GRID,
                           timestep = self.timestep_GRID, layers = self.hiddenlayers_GRID, layers_min =self.hiddenlayers_min_GRID,
                           neu_max = self.neurons_max_GRID, neu_min = self.neurons_min_GRID, interval = self.interval_GRID, iteration = self.iteration_GRID,
                           optimizer = self.optimizer_GRID, loss_fun = self.loss_function_GRID, activation = self.activation_GRID,
                           batch_size = self.batch_size_GRID, epochs = self.epochs_GRID, file = self.file_name, regularizer = self.regularizer_GRID
                           , dropout = self.dropout_GRID)

        self.results_train = pd.read_csv("Results_Train_GUI.csv", header=None)
        self.results_test = pd.read_csv("Results_Test_GUI.csv", header=None)
        self.results_train = np.array(self.results_train)
        self.results_test = np.array(self.results_test)
        self.plot_train = self.results_train
        self.plot_test = self.results_test
        self.x_axis_label = []
        self.y_axis_label = []

        for i in range(self.hiddenlayers_min_GRID, self.hiddenlayers_GRID + 1):
            self.x_axis_label.append(i)
        for j in range(self.neurons_min_GRID, self.neurons_max_GRID + 1, self.interval_GRID):
            self.y_axis_label.append(j)

        self.heatmap_train.canvas.axes.clear()
        sb.heatmap(self.plot_train, xticklabels=self.x_axis_label, yticklabels=self.y_axis_label, cbar= None,
                annot=True,fmt=".3f", linewidths=2, linecolor="white",ax=self.heatmap_train.canvas.axes) #cbar = true
        self.heatmap_train.canvas.axes.set_xlabel("Layers", labelpad= 1)
        self.heatmap_train.canvas.axes.set_ylabel("Neurons")
        self.heatmap_train.canvas.axes.set_title("MAE_training set")
        self.heatmap_train.canvas.draw()

        self.heatmap_test.canvas.axes.clear()
        sb.heatmap(self.plot_test, xticklabels=self.x_axis_label, yticklabels=self.y_axis_label, cbar= None,
                   annot=True, fmt=".3f", linewidths=2, linecolor="white", ax=self.heatmap_test.canvas.axes)
        self.heatmap_test.canvas.axes.set_xlabel("Layers",labelpad= 1)
        self.heatmap_test.canvas.axes.set_ylabel("Neurons")
        self.heatmap_test.canvas.axes.set_title("MAE_testing set")
        self.heatmap_test.canvas.draw()

        self.results_test_min = round(self.results_test.min(),3)
        self.results_train_min = round(self.results_train.min(),3)

        self.lineEdit_14.setText(str(self.results_train_min))
        self.lineEdit_15.setText(str(self.results_test_min))

        self.Min_test_index = np.argwhere(self.results_test == self.results_test.min())
        # self.Min_train_index = np.argwhere(self.results_train == self.results_train_min)
        print("Min_test_index:\n", self.Min_test_index)
        self.min_test_col = self.hiddenlayers_min_GRID + self.Min_test_index[0, 1]
        self.min_test_row = self.neurons_min_GRID + self.Min_test_index[0, 0] * self.interval_GRID
        # self.min_train_col = self.Min_train_index[0, 1] + 1
        # self.min_train_row = self.neurons_min_GRID + self.Min_train_index[0, 0] * self.interval_GRID

        self.lineEdit_12.setText(str(self.min_test_col))
        self.lineEdit_13.setText(str(self.min_test_row))

    def get_measure(self):
        measures = ("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","ave.")
        measure, ok = QInputDialog.getItem(self, "Select a measure point", "Enter measure point:",measures,0,False)
        if ok:
            self.lineEdit_8.setText(str(measure))

    def run_model(self):
        #Build model
        self.measure_point = str(self.lineEdit_8.text())
        self.hiddenlayers = int(self.lineEdit_hl.text())
        self.neurons = int(self.line_neu.text())
        self.activation = str(self.lineEdit_af.text())
        self.regularizer = float(self.lineEdit_6.text())
        self.dropout = float(self.lineEdit_17.text())

        self.temp_model = Sequential()
        for i in range(self.hiddenlayers):
            nou = self.neurons
            if i == 0:
                self.temp_model.add(
                    LSTM(nou, kernel_regularizer=regularizers.l2(self.regularizer), input_shape=(2, 1),
                         return_sequences=True, activation=self.activation))
                self.temp_model.add(Dropout(self.dropout))
            elif i == int(self.hiddenlayers - 1):
                self.temp_model.add(LSTM(nou, activation=self.activation))
                self.temp_model.add(Dropout(self.dropout))
            else:
                self.temp_model.add(LSTM(nou, return_sequences=True, activation=self.activation))
                self.temp_model.add(Dropout(self.dropout))
        self.temp_model.add(Dense(1))
        # self.temp_model.compile(loss='mse', optimizer=self.optimier, metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
        print(self.temp_model.summary())

        # Run model
        self.optimizer = str(self.lineEdit_opt.text())
        self.epochs = int(self.line_epochs.text())
        self.validation_split = float(self.lineEdit_vs.text())
        self.batch_size = int(self.line_bs.text())
        self.split_ratio = float(self.lineEdit_sr.text())
        self.timestep = int(self.lineEdit_ts.text())

        if self.radioButton_KSA.isChecked():
            LSTM_model_import.run(file=self.file_name_2, measure_point = self.measure_point, model=self.temp_model,
                                  split_ratio=self.split_ratio, timestep=self.timestep, optimizer=self.optimizer,
                                  epochs=self.epochs, validation_split=self.validation_split, batch_size=self.batch_size)
            self.tab_3.setCurrentIndex(self.tab_3.indexOf(self.tab_4))
            # Plot graph on GUI
            self.val_loss = pd.read_csv('loss_val_for_plot.csv', header=None)
            self.loss = pd.read_csv('loss_train_for_plot.csv', header=None)
            self.val_loss = np.log10(self.val_loss)
            self.loss = np.log10(self.loss)
            self.loss_graph.canvas.axes.clear()
            self.loss_graph.canvas.axes.plot(self.loss, label='Train')
            self.loss_graph.canvas.axes.plot(self.val_loss, label='Validation')
            self.loss_graph.canvas.axes.set_ylabel('Loss (in log10 scale)', labelpad=10)
            self.loss_graph.canvas.axes.set_xlabel('Epochs')
            self.loss_graph.canvas.axes.xaxis.set_label_coords(1.05, 0)
            self.loss_graph.canvas.axes.legend()
            self.loss_graph.canvas.axes.set_title('Model Loss')
            self.loss_graph.canvas.draw()

            self.train_predict = pd.read_csv('train_predict.csv', header=None)
            self.test_predict = pd.read_csv('test_predict.csv', header=None)
            self.data = pd.read_csv('database_airport.csv')
            self.data = self.data[self.measure_point]
            self.year = []
            for i in range(1994, 1994 + len(self.data)):
                self.year.append(i)
            self.data = np.array(self.data).reshape(-1, 1)
            self.trainPredictPlot = np.empty_like(self.data)
            self.trainPredictPlot[:, :] = np.nan
            self.trainPredictPlot[self.timestep:len(self.train_predict) + self.timestep, :] = self.train_predict
            self.testPredictPlot = np.empty_like(self.data)
            self.testPredictPlot[:, :] = np.nan
            self.testPredictPlot[len(self.train_predict) + (self.timestep * 2):len(self.data), :] = self.test_predict
            self.predict_graph.canvas.axes.clear()
            self.predict_graph.canvas.axes.scatter(self.year, self.data, label='Measured Settlement', color='#448ee4')
            self.predict_graph.canvas.axes.scatter(self.year,self.trainPredictPlot, label='Predicted Settlement(training set)',color="orange")
            self.predict_graph.canvas.axes.scatter(self.year,self.testPredictPlot, label='Predicted Settlement(testing set)',color='green')
            self.predict_graph.canvas.axes.set_ylabel('Settltment(m)', labelpad=10)
            self.predict_graph.canvas.axes.set_xlabel('Years')
            self.predict_graph.canvas.axes.xaxis.set_label_coords(1.05, 0)
            self.predict_graph.canvas.axes.legend()
            self.predict_graph.canvas.axes.set_title('Model Result ' + self.measure_point)
            self.predict_graph.canvas.draw()

            self.trainY = pd.read_csv('trainY_KSA.csv',header=None)
            self.testY = pd.read_csv('testY_KSA.csv',header=None)
            self.mae_train = round(mean_absolute_error(self.trainY,self.train_predict),2)
            self.mae_test = round(mean_absolute_error(self.testY, self.test_predict),2)
            self.lineEdit_21.setText(str(self.mae_train))
            self.lineEdit_22.setText(str(self.mae_test))

        elif self.radioButton_HKA.isChecked():
            HKA_model_import.run(file=self.file_name_2, measure_point = self.measure_point,
                                 model=self.temp_model, split_ratio=self.split_ratio,
                                  timestep=self.timestep, optimizer=self.optimizer, epochs=self.epochs,
                                  validation_split=self.validation_split, batch_size=self.batch_size)

            self.tab_3.setCurrentIndex(self.tab_3.indexOf(self.tab_4))
            # Plot graph on GUI
            self.val_loss = pd.read_csv('loss_val_for_plot_HKA.csv', header=None)
            self.loss = pd.read_csv('loss_train_for_plot_HKA.csv', header=None)
            self.val_loss = np.log10(self.val_loss)
            self.loss = np.log10(self.loss)
            self.loss_graph.canvas.axes.clear()
            self.loss_graph.canvas.axes.plot(self.loss, label='Train')
            self.loss_graph.canvas.axes.plot(self.val_loss, label='Validation')
            self.loss_graph.canvas.axes.set_ylabel('Loss', labelpad=10)
            self.loss_graph.canvas.axes.set_xlabel('Epochs')
            self.loss_graph.canvas.axes.xaxis.set_label_coords(1.05, 0)
            self.loss_graph.canvas.axes.legend()
            self.loss_graph.canvas.axes.set_title('Model Loss')
            self.loss_graph.canvas.draw()

            self.train_predict = pd.read_csv('train_predict_HKA.csv', header=None)
            self.test_predict = pd.read_csv('test_predict_HKA.csv', header=None)
            self.dataset = pd.read_csv('HKA_data_interpolated.csv')
            self.days = self.dataset["days"]
            self.dataset = self.dataset[self.measure_point]

            self.dataset = np.array(self.dataset).reshape(-1, 1)
            self.trainPredictPlot = np.empty_like(self.dataset)
            self.trainPredictPlot[:, :] = np.nan
            self.trainPredictPlot[self.timestep:len(self.train_predict) + self.timestep, :] = self.train_predict
            self.testPredictPlot = np.empty_like(self.dataset)
            self.testPredictPlot[:, :] = np.nan
            self.testPredictPlot[len(self.train_predict) + (self.timestep * 2):len(self.dataset), :] = self.test_predict
            self.predict_graph.canvas.axes.clear()
            self.predict_graph.canvas.axes.scatter(self.days, self.dataset, label='Measured Settlement', color='#448ee4')
            self.predict_graph.canvas.axes.scatter(self.days,self.trainPredictPlot, label='Predicted Settlement(training set)', color='orange')
            self.predict_graph.canvas.axes.scatter(self.days, self.testPredictPlot, label='Predicted Settlement(testing set)', color='green')
            self.predict_graph.canvas.axes.set_ylabel('Settltment(m)', labelpad=10)
            self.predict_graph.canvas.axes.set_xlabel('Days')
            self.predict_graph.canvas.axes.xaxis.set_label_coords(1.05, 0)
            self.predict_graph.canvas.axes.legend()
            self.predict_graph.canvas.axes.set_title('Model Result ' + self.measure_point)
            self.predict_graph.canvas.draw()

            self.trainY = pd.read_csv('trainY_HKA.csv',header=None)
            self.testY = pd.read_csv('testY_HKA.csv',header=None)
            self.mae_train = round(mean_absolute_error(self.trainY,self.train_predict),3)
            self.mae_test = round(mean_absolute_error(self.testY, self.test_predict),3)
            self.lineEdit_21.setText(str(self.mae_train))
            self.lineEdit_22.setText(str(self.mae_test))

    def save_csv_loss(self):
        if self.radioButton_KSA.isChecked():
            self.epochs = int(self.line_epochs.text())
            self.index = []
            for i in range(1,self.epochs+1):
                self.index.append(i)
            self.val_loss = pd.read_csv('loss_val_for_plot.csv', header=None)
            self. loss = pd.read_csv('loss_train_for_plot.csv', header=None)
            self.val_loss = np.array(self.val_loss).reshape(-1, 1)
            self.loss = np.array(self.loss).reshape(-1, 1)
            self.df1 = pd.DataFrame(self.val_loss, columns=['a'], index=self.index)
            self.df2 = pd.DataFrame(self.loss, columns=['b'], index=self.index)
            self.loss_combine = pd.concat([self.df1, self.df2], axis=1)
            self.loss_combine = np.array(self.loss_combine).reshape(-1,2)
            filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
            if filename[0] == "":
                pass
            else:
                with open(filename[0], 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['loss_validation', 'loss_train'])
                    writer.writerows(self.loss_combine)

        elif self.radioButton_HKA.isChecked():
            self.epochs = int(self.line_epochs.text())
            self.index = []
            for i in range(1, self.epochs + 1):
                self.index.append(i)
            self.val_loss = pd.read_csv('loss_val_for_plot_HKA.csv', header=None)
            self.loss = pd.read_csv('loss_train_for_plot_HKA.csv', header=None)
            self.val_loss = np.array(self.val_loss).reshape(-1, 1)
            self.loss = np.array(self.loss).reshape(-1, 1)
            self.df1 = pd.DataFrame(self.val_loss, columns=['a'], index=self.index)
            self.df2 = pd.DataFrame(self.loss, columns=['b'], index=self.index)
            self.loss_combine = pd.concat([self.df1, self.df2], axis=1)
            self.loss_combine = np.array(self.loss_combine).reshape(-1, 2)
            filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
            if filename[0] == "":
                pass
            else:
                with open(filename[0], 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['loss_validation', 'loss_train'])
                    writer.writerows(self.loss_combine)

    def save_csv_train_result(self):
        if self.radioButton_KSA.isChecked():
            self.train_predict = pd.read_csv('train_predict.csv', header=None)
            self.test_predict = pd.read_csv('test_predict.csv', header=None)
            self.index_train = []
            for i in range(len(self.train_predict)):
                self.index_train.append(i)
            self.index_test = []
            for i in range(len(self.test_predict)):
                self.index_test.append(i)
            self.train_predict = np.array(self.train_predict).reshape(-1,1)
            self.test_predict = np.array(self.test_predict).reshape(-1, 1)
            self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
            self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
            self.train_result_combine = pd.concat([self.df1, self.df2], axis=1)
            self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 2)
            filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
            if filename[0] == "":
                pass
            else:
                with open(filename[0], 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['train_predict', 'test_predict'])
                    writer.writerows(self.train_result_combine)

        elif self.radioButton_HKA.isChecked():
            self.train_predict = pd.read_csv('train_predict_HKA.csv', header=None)
            self.test_predict = pd.read_csv('test_predict_HKA.csv', header=None)
            self.index_train = []
            for i in range(len(self.train_predict)):
                self.index_train.append(i)
            self.index_test = []
            for i in range(len(self.test_predict)):
                self.index_test.append(i)
            self.train_predict = np.array(self.train_predict).reshape(-1,1)
            self.test_predict = np.array(self.test_predict).reshape(-1, 1)
            self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
            self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
            self.train_result_combine = pd.concat([self.df1, self.df2], axis=1)
            self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 2)
            filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
            if filename[0] == "":
                pass
            else:
                with open(filename[0], 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['train_predict', 'test_predict'])
                    writer.writerows(self.train_result_combine)

    def save(self):
        self.split_ratio_save = float(self.lineEdit_2.text())
        self.validation_split_save = float(self.lineEdit_3.text())
        self.timestep_save = int(self.lineEdit_4.text())
        self.hiddenlayers_save = int(self.lineEdit_12.text())
        self.neurons_save = int(self.lineEdit_13.text())
        self.optimizer_save = str(self.comboBox.currentText())
        self.activation_save = str(self.comboBox_3.currentText())
        self.batch_size_save = int(self.lineEdit_10.text())
        self.epochs_save = int(self.lineEdit_18.text())
        self.regularizer_save = float(self.lineEdit.text())
        self.dropout_save = float(self.lineEdit_19.text())

    def load(self):
        self.lineEdit_hl.setText(str(self.hiddenlayers_save))
        self.line_neu.setText(str(self.neurons_save))
        self.lineEdit_vs.setText(str(self.validation_split_save))
        self.lineEdit_sr.setText(str(self.split_ratio_save))
        self.lineEdit_ts.setText(str(self.timestep_save))
        self.lineEdit_opt.setText(str(self.optimizer_save))
        self.lineEdit_af.setText(str(self.activation_save))
        self.line_bs.setText(str(self.batch_size_save))
        self.line_epochs.setText(str(self.epochs_save))
        self.lineEdit_6.setText(str(self.regularizer_save))
        self.lineEdit_17.setText(str(self.dropout_save))

    def predict(self):
        # Run model
        self.future = int(self.lineEdit_PY.text())
        self.measure_point = str(self.comboBox_MP.currentText())
        self.split_ratio = float(self.lineEdit_sr.text())
        self.timestep = int(self.lineEdit_ts.text())
        validation_forGUI.run(year=self.future, measure_point=self.measure_point,split_ratio=self.split_ratio,
                              timestep = self.timestep)

        # Load Graph on GUI
        self.data = pd.read_csv('dataset_GUI_predictmodel.csv',header=None)
        self.train_predict_predictmodel = pd.read_csv('train_predict_predictmodel.csv', header=None)
        self.test_predict_predictmodel = pd.read_csv('test_predict_predictmodel.csv', header=None)
        self.future_predict_predictmodel = pd.read_csv('future_predict_predictmodel.csv', header=None)
        self.data = np.array(self.data).reshape(-1, 1)
        self.year_predict = []
        for i in range(1994, 1994 + len(self.data)):
            self.year_predict.append(i)
        self.future_year_predict = []
        for i in range(1994, self.future + 1):
            self.future_year_predict.append(i)

        self.trainPredictPlot = np.empty_like(self.data)
        self.trainPredictPlot[:, :] = np.nan
        self.trainPredictPlot[self.timestep:len(self.train_predict_predictmodel) + self.timestep, :] = self.train_predict_predictmodel

        self.testPredictPlot = np.empty_like(self.data)
        self.testPredictPlot[:, :] = np.nan
        self.testPredictPlot[len(self.train_predict_predictmodel) + (self.timestep * 2):len(self.data), :] = self.test_predict_predictmodel

        self.z = np.zeros((len(self.data) + len(self.future_predict_predictmodel), 1))
        self.futurePredictPlot = np.empty_like(self.z)
        self.futurePredictPlot[:, :] = np.nan
        self.futurePredictPlot[len(self.data):, :] = self.future_predict_predictmodel

        self.predict_graph_predict.canvas.axes.clear()
        self.predict_graph_predict.canvas.axes.scatter(self.year_predict, self.data, label='Measured Settlement', color='#448ee4')
        self.predict_graph_predict.canvas.axes.scatter(self.year_predict, self.trainPredictPlot,
                                               label='Predicted Settlement(training set)', color="orange")
        self.predict_graph_predict.canvas.axes.scatter(self.year_predict, self.testPredictPlot,
                                               label='Predicted Settlement(testing set)', color='green')
        self.predict_graph_predict.canvas.axes.scatter(self.future_year_predict,self.futurePredictPlot, label='Predicted Future Settlement',
                                                    color="red")
        # self.predict_graph_predict.canvas.axes.plot(self.trainPredictPlot,label='Predicted Settlement(training set)',color="orange")
        # self.predict_graph_predict.canvas.axes.plot(self.testPredictPlot,label='Predicted Settlement(testing set)',color="green")
        # self.predict_graph_predict.canvas.axes.plot(self.futurePredictPlot, label='Predicted Future Settlement', color="red")
        # self.predict_graph_predict.canvas.axes.plot(self.data, label='Measured Settlement',color="#448ee4")
        self.predict_graph_predict.canvas.axes.set_ylabel('Settltment(m)')
        self.predict_graph_predict.canvas.axes.set_xlabel('Years')
        self.predict_graph_predict.canvas.axes.legend()
        self.predict_graph_predict.canvas.axes.set_title('Model Result')
        self.predict_graph_predict.canvas.draw()

    def save_csv_predict_KIA(self):
        self.train_predict = pd.read_csv('train_predict_predictmodel.csv', header=None)
        self.test_predict = pd.read_csv('test_predict_predictmodel.csv', header=None)
        self.future_predict = pd.read_csv('future_predict_predictmodel.csv', header=None)
        self.index_train = []
        for i in range(len(self.train_predict)):
            self.index_train.append(i)
        self.index_test = []
        for i in range(len(self.test_predict)):
            self.index_test.append(i)
        self.index_future = []
        for i in range(len(self.future_predict)):
            self.index_future.append(i)
        self.train_predict = np.array(self.train_predict).reshape(-1, 1)
        self.test_predict = np.array(self.test_predict).reshape(-1, 1)
        self.future_predict = np.array(self.future_predict).reshape(-1, 1)
        self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
        self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
        self.df3 = pd.DataFrame(self.future_predict, columns=['b'], index=self.index_future)
        self.train_result_combine = pd.concat([self.df1, self.df2,self.df3], axis=1)
        self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 3)
        filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
        if filename[0] == "":
            pass
        else:
            with open(filename[0], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['train_predict', 'test_predict', 'future_predict'])
                writer.writerows(self.train_result_combine)

    def save_csv_predict_HKA(self):
        self.train_predict = pd.read_csv('train_predict_predictmodel_HKA.csv', header=None)
        self.test_predict = pd.read_csv('test_predict_predictmodel_HKA.csv', header=None)
        self.future_predict = pd.read_csv('future_predict_predictmodel_HKA.csv', header=None)
        self.index_train = []
        for i in range(len(self.train_predict)):
            self.index_train.append(i)
        self.index_test = []
        for i in range(len(self.test_predict)):
            self.index_test.append(i)
        self.index_future = []
        for i in range(len(self.future_predict)):
            self.index_future.append(i)
        self.train_predict = np.array(self.train_predict).reshape(-1, 1)
        self.test_predict = np.array(self.test_predict).reshape(-1, 1)
        self.future_predict = np.array(self.future_predict).reshape(-1, 1)
        self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
        self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
        self.df3 = pd.DataFrame(self.future_predict, columns=['b'], index=self.index_future)
        self.train_result_combine = pd.concat([self.df1, self.df2,self.df3], axis=1)
        self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 3)
        filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
        if filename[0] == "":
            pass
        else:
            with open(filename[0], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['train_predict', 'test_predict', 'future_predict'])
                writer.writerows(self.train_result_combine)

    def predict_HKA(self):
        # Run model
        self.future = int(self.lineEdit_PY_3.text())
        self.measure_point = str(self.comboBox_MP_3.currentText())
        self.split_ratio = float(self.lineEdit_sr.text())
        self.timestep = int(self.lineEdit_ts.text())
        validation_forGUI_HKA.run(day=self.future, measure_point=self.measure_point,split_ratio=self.split_ratio,
                              timestep = self.timestep)

        # Load Graph on GUI
        self.data = pd.read_csv('dataset_GUI_predictmodel_HKA.csv',header=None)
        self.days= pd.read_csv('HKA_data_interpolated.csv')
        self.days = self.days["days"]
        self.train_predict_predictmodel = pd.read_csv('train_predict_predictmodel_HKA.csv', header=None)
        self.test_predict_predictmodel = pd.read_csv('test_predict_predictmodel_HKA.csv', header=None)
        self.future_predict_predictmodel = pd.read_csv('future_predict_predictmodel_HKA.csv', header=None)
        self.data = np.array(self.data).reshape(-1, 1)
        self.days = np.array(self.days).reshape(-1, 1)
        self.future_year = []
        for i in range(203, self.future + 1, 20):
            self.future_year.append(i)

        self.trainPredictPlot = np.empty_like(self.data)
        self.trainPredictPlot[:, :] = np.nan
        self.trainPredictPlot[self.timestep:len(self.train_predict_predictmodel) + self.timestep, :] = self.train_predict_predictmodel

        self.testPredictPlot = np.empty_like(self.data)
        self.testPredictPlot[:, :] = np.nan
        self.testPredictPlot[len(self.train_predict_predictmodel) + (self.timestep * 2):len(self.data), :] = self.test_predict_predictmodel

        self.z = np.zeros((len(self.data) + len(self.future_predict_predictmodel), 1))
        self.futurePredictPlot = np.empty_like(self.z)
        self.futurePredictPlot[:, :] = np.nan
        self.futurePredictPlot[len(self.data):, :] = self.future_predict_predictmodel

        self.predict_graph_predict.canvas.axes.clear()
        self.predict_graph_predict.canvas.axes.scatter(self.days, self.data, label='Measured Settlement', color='#448ee4')
        self.predict_graph_predict.canvas.axes.scatter(self.days, self.trainPredictPlot,
                                               label='Predicted Settlement(training set)', color="orange")
        self.predict_graph_predict.canvas.axes.scatter(self.days, self.testPredictPlot,
                                               label='Predicted Settlement(testing set)', color='green')
        self.predict_graph_predict.canvas.axes.scatter(self.future_year,self.futurePredictPlot, label='Predicted Future Settlement',
                                                    color="red")

        self.predict_graph_predict.canvas.axes.set_ylabel('Settltment(m)')
        self.predict_graph_predict.canvas.axes.set_xlabel('Contract day')
        self.predict_graph_predict.canvas.axes.legend()
        self.predict_graph_predict.canvas.axes.set_title('Model Result')
        self.predict_graph_predict.canvas.draw()

    def predict2(self):
        # Run model
        self.future = int(self.lineEdit_PY_2.text())
        self.measure_point = str(self.comboBox_MP_2.currentText())
        validation_forGUI_defaultmodel.run(year=self.future, measure_point=self.measure_point)

        # Load Graph on GUI
        self.data = pd.read_csv('dataset_GUI_predictmodel_default.csv',header=None)
        self.train_predict_predictmodel = pd.read_csv('train_predict_predictmodel_default.csv', header=None)
        self.test_predict_predictmodel = pd.read_csv('test_predict_predictmodel_default.csv', header=None)
        self.future_predict_predictmodel = pd.read_csv('future_predict_predictmodel_default.csv', header=None)
        self.data = np.array(self.data).reshape(-1, 1)
        self.year_predict_default = []
        for i in range(1994, 1994 + len(self.data)):
            self.year_predict_default.append(i)
        self.future_year_predict_default = []
        for i in range(1994, self.future +1):
            self.future_year_predict_default.append(i)

        self.trainPredictPlot = np.empty_like(self.data)
        self.trainPredictPlot[:, :] = np.nan
        self.trainPredictPlot[2:len(self.train_predict_predictmodel) + 2, :] = self.train_predict_predictmodel

        self.testPredictPlot = np.empty_like(self.data)
        self.testPredictPlot[:, :] = np.nan
        self.testPredictPlot[len(self.train_predict_predictmodel) + (2 * 2):len(self.data), :] = self.test_predict_predictmodel

        self.z = np.zeros((len(self.data) + len(self.future_predict_predictmodel), 1))
        self.futurePredictPlot = np.empty_like(self.z)
        self.futurePredictPlot[:, :] = np.nan
        self.futurePredictPlot[len(self.data):, :] = self.future_predict_predictmodel

        self.predict_graph_predict_default.canvas.axes.clear()
        self.predict_graph_predict_default.canvas.axes.scatter(self.year_predict_default, self.data, label='Measured Settlement',
                                                       color='#448ee4')
        self.predict_graph_predict_default.canvas.axes.scatter(self.year_predict_default, self.trainPredictPlot,
                                                       label='Predicted Settlement(training set)', color="orange")
        self.predict_graph_predict_default.canvas.axes.scatter(self.year_predict_default, self.testPredictPlot,
                                               label='Predicted Settlement(testing set)', color='green')
        self.predict_graph_predict_default.canvas.axes.scatter(self.future_year_predict_default, self.futurePredictPlot,
                                                       label='Predicted Future Settlement',
                                                       color="red")
        # self.predict_graph_predict_default.canvas.axes.plot(self.trainPredictPlot,label='Predicted Settlement(training set)',color="orange")
        # self.predict_graph_predict_default.canvas.axes.plot(self.testPredictPlot,label='Predicted Settlement(testing set)',color="green")
        # self.predict_graph_predict_default.canvas.axes.plot(self.futurePredictPlot, label='Predicted Future Settlement', color="red")
        # self.predict_graph_predict_default.canvas.axes.plot(self.data, label='Measured Settlement',color="#448ee4")
        self.predict_graph_predict_default.canvas.axes.set_ylabel('Settltment(m)')
        self.predict_graph_predict_default.canvas.axes.set_xlabel('Years')
        self.predict_graph_predict_default.canvas.axes.legend()
        self.predict_graph_predict_default.canvas.axes.set_title('Model Result')
        self.predict_graph_predict_default.canvas.draw()

    def predict_HKA_2(self):
        # Run model
        self.future = int(self.lineEdit_PY_4.text())
        self.measure_point = str(self.comboBox_MP_4.currentText())
        validation_forGUI_defaultmodel_HKA.run(day=self.future, measure_point=self.measure_point)

        # Load Graph on GUI
        self.data = pd.read_csv('dataset_GUI_predictmodel_HKA_default.csv',header=None)
        self.days= pd.read_csv('HKA_data_interpolated.csv')
        self.days = self.days["days"]
        self.train_predict_predictmodel = pd.read_csv('train_predict_predictmodel_HKA_default.csv', header=None)
        self.test_predict_predictmodel = pd.read_csv('test_predict_predictmodel_HKA_default.csv', header=None)
        self.future_predict_predictmodel = pd.read_csv('future_predict_predictmodel_HKA_default.csv', header=None)
        self.data = np.array(self.data).reshape(-1, 1)
        self.days = np.array(self.days).reshape(-1, 1)
        self.future_year = []
        for i in range(203, self.future + 1, 20):
            self.future_year.append(i)

        self.trainPredictPlot = np.empty_like(self.data)
        self.trainPredictPlot[:, :] = np.nan
        self.trainPredictPlot[2:len(self.train_predict_predictmodel) + 2, :] = self.train_predict_predictmodel

        self.testPredictPlot = np.empty_like(self.data)
        self.testPredictPlot[:, :] = np.nan
        self.testPredictPlot[len(self.train_predict_predictmodel) + (2* 2):len(self.data), :] = self.test_predict_predictmodel

        self.z = np.zeros((len(self.data) + len(self.future_predict_predictmodel), 1))
        self.futurePredictPlot = np.empty_like(self.z)
        self.futurePredictPlot[:, :] = np.nan
        self.futurePredictPlot[len(self.data):, :] = self.future_predict_predictmodel

        self.predict_graph_predict_default.canvas.axes.clear()
        self.predict_graph_predict_default.canvas.axes.scatter(self.days, self.data, label='Measured Settlement', color='#448ee4')
        self.predict_graph_predict_default.canvas.axes.scatter(self.days, self.trainPredictPlot,
                                               label='Predicted Settlement(training set)', color="orange")
        self.predict_graph_predict_default.canvas.axes.scatter(self.days, self.testPredictPlot,
                                               label='Predicted Settlement(testing set)', color='green')
        self.predict_graph_predict_default.canvas.axes.scatter(self.future_year,self.futurePredictPlot, label='Predicted Future Settlement',
                                                    color="red")

        self.predict_graph_predict_default.canvas.axes.set_ylabel('Settltment(m)')
        self.predict_graph_predict_default.canvas.axes.set_xlabel('Contract day')
        self.predict_graph_predict_default.canvas.axes.legend()
        self.predict_graph_predict_default.canvas.axes.set_title('Model Result')
        self.predict_graph_predict_default.canvas.draw()

    def save_csv_predict_default_KIA(self):
        self.train_predict = pd.read_csv('train_predict_predictmodel_default.csv', header=None)
        self.test_predict = pd.read_csv('test_predict_predictmodel_default.csv', header=None)
        self.future_predict = pd.read_csv('future_predict_predictmodel_default.csv', header=None)
        self.index_train = []
        for i in range(len(self.train_predict)):
            self.index_train.append(i)
        self.index_test = []
        for i in range(len(self.test_predict)):
            self.index_test.append(i)
        self.index_future = []
        for i in range(len(self.future_predict)):
            self.index_future.append(i)
        self.train_predict = np.array(self.train_predict).reshape(-1, 1)
        self.test_predict = np.array(self.test_predict).reshape(-1, 1)
        self.future_predict = np.array(self.future_predict).reshape(-1, 1)
        self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
        self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
        self.df3 = pd.DataFrame(self.future_predict, columns=['b'], index=self.index_future)
        self.train_result_combine = pd.concat([self.df1, self.df2, self.df3], axis=1)
        self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 3)
        filename = QFileDialog.getSaveFileName(self, "SAVE CSV", os.getenv('HOME'), "CSV Files (*.csv)")
        if filename[0] == "":
            pass
        else:
            with open(filename[0], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['train_predict', 'test_predict', 'future_predict'])
                writer.writerows(self.train_result_combine)

    def save_csv_predict_default_HKA(self):
        self.train_predict = pd.read_csv('train_predict_predictmodel_HKA_default.csv', header=None)
        self.test_predict = pd.read_csv('test_predict_predictmodel_HKA_default.csv', header=None)
        self.future_predict = pd.read_csv('future_predict_predictmodel_HKA_default.csv', header=None)
        self.index_train = []
        for i in range(len(self.train_predict)):
            self.index_train.append(i)
        self.index_test = []
        for i in range(len(self.test_predict)):
            self.index_test.append(i)
        self.index_future = []
        for i in range(len(self.future_predict)):
            self.index_future.append(i)
        self.train_predict = np.array(self.train_predict).reshape(-1, 1)
        self.test_predict = np.array(self.test_predict).reshape(-1, 1)
        self.future_predict = np.array(self.future_predict).reshape(-1, 1)
        self.df1 = pd.DataFrame(self.train_predict, columns=['a'], index=self.index_train)
        self.df2 = pd.DataFrame(self.test_predict, columns=['b'], index=self.index_test)
        self.df3 = pd.DataFrame(self.future_predict, columns=['b'], index=self.index_future)
        self.train_result_combine = pd.concat([self.df1, self.df2, self.df3], axis=1)
        self.train_result_combine = np.array(self.train_result_combine).reshape(-1, 3)
        filename = QFileDialog.getSaveFileName(self," SAVE CSV",os.getenv('HOME'), "CSV Files (*.csv)")
        if filename[0] == '':
            pass
        else:
            with open(filename[0], 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['train_predict', 'test_predict', 'future_predict'])
                writer.writerows(self.train_result_combine)

    def save_fig(self):
        self.loss_graph.canvas.toolbar.save_figure()

    def save_fig_2(self):
        self.predict_graph.canvas.toolbar.save_figure()

    def save_fig_3(self):
        self.predict_graph_predict_default.canvas.toolbar.save_figure()

    def save_fig_4(self):
        self.predict_graph_predict.canvas.toolbar.save_figure()

    def home(self):
        self.loss_graph.canvas.toolbar.home()

    def home_2(self):
        self.predict_graph.canvas.toolbar.home()

    def home_3(self):
        self.predict_graph_predict_default.canvas.toolbar.home()

    def home_4(self):
        self.predict_graph_predict.canvas.toolbar.home()

    def zoom(self):
        self.loss_graph.canvas.toolbar.zoom()

    def zoom_2(self):
        self.predict_graph.canvas.toolbar.zoom()

    def zoom_3(self):
        self.predict_graph_predict_default.canvas.toolbar.zoom()

    def zoom_4(self):
        self.predict_graph_predict.canvas.toolbar.zoom()

    def move(self):
        self.loss_graph.canvas.toolbar.pan()

    def move_2(self):
        self.predict_graph.canvas.toolbar.pan()

    def move_3(self):
        self.predict_graph_predict_default.canvas.toolbar.pan()

    def move_4(self):
        self.predict_graph_predict.canvas.toolbar.pan()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = ModelRunnerUi()
    m.show()
    sys.exit(app.exec_())
