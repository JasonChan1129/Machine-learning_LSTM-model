import sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import LSTM_model_import
from mpldatacursor import datacursor
# ---------------------------------------------------------------------------------------------------------
class MatplotlibWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        plt.style.use('dark_background')

        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.canvas,self,coordinates=False)
        self.toolbar.hide()

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        # vertical_layout.addWidget(self.toolbar)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        # self.cursor = Cursor(self.canvas.axes, useblit=True, color='red', linewidth=2)

# class MyMplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = Figure(figsize=(width, height), dpi=dpi) # 新建一个figure
#         self.axes = self.fig.add_subplot(111)  # 建立一个子图，如果要建立复合图，可以在这里修改
#
#
#         FigureCanvas.__init__(self, self.fig)
#         self.setParent(parent)
#
#         '''定义FigureCanvas的尺寸策略，这部分的意思是设置FigureCanvas，使之尽可能的向外填充空间。'''
#         FigureCanvas.setSizePolicy(self,
#                                    QSizePolicy.Expanding,
#                                    QSizePolicy.Expanding)
#         FigureCanvas.updateGeometry(self)
#
#     def load_csv(self):
#         self.val_loss = pd.read_csv('loss_val_for_plot.csv', header=None)
#         self.loss = pd.read_csv('loss_train_for_plot.csv', header=None)
#         self.val_loss = np.log10(self.val_loss)
#         self.loss = np.log10(self.loss)
#         return self.val_loss, self.loss
#
#     def load_csv_1(self):
#         self.train_predict = pd.read_csv('train_predict.csv', header=None)
#         self.test_predict = pd.read_csv('test_predict.csv', header=None)
#         self.data = pd.read_csv('database_airport.csv')
#         self.data = self.data['ave.']
#         self.data = np.array(self.data).reshape(-1, 1)
#
#         self.trainPredictPlot = np.empty_like(self.data)
#         self.trainPredictPlot[:, :] = np.nan
#         self.trainPredictPlot[2:len(self.train_predict) + 2, :] = self.train_predict
#
#         self.testPredictPlot = np.empty_like(self.data)
#         self.testPredictPlot[:, :] = np.nan
#         self.testPredictPlot[len(self.train_predict) + (2 * 2):len(self.data), :] = self.test_predict
#         return self.trainPredictPlot, self.testPredictPlot, self.data
#
#     def plot_graph(self):
#         self.load_csv()
#         self.fig.suptitle('Model Loss')
#         self.axes.plot(self.loss,label='Train')
#         self.axes.plot(self.val_loss,label='Validation')
#         self.axes.set_ylabel('Loss')
#         self.axes.set_xlabel('Epochs')
#         self.axes.legend()
#         self.axes.grid(True)
#         FigureCanvas.draw(self)
#
#
#
#     def plot_graph1(self):
#         self.load_csv_1()
#         self.fig.suptitle('Model Result (Average Value)')
#         self.axes.plot(self.data, label='Measured Settlement')
#         self.axes.plot(self.testPredictPlot,label='Predicted Settlement(testing set)')
#         self.axes.plot(self.trainPredictPlot,label='Predicted Settlement(training set)')
#         self.axes.set_ylabel('Settlement(m)')
#         self.axes.set_xlabel('Years')
#         self.axes.legend()
#         self.axes.grid(True)
#         FigureCanvas.draw(self)
#
#
# class MatplotlibWidget(QWidget):
#     def __init__(self, parent=None):
#         super(MatplotlibWidget, self).__init__(parent)
#         self.initUi()
#
#     def initUi(self):
#         self.layout = QVBoxLayout(self)
#         self.mpl = MyMplCanvas(self, width=5, height=4, dpi=100)
#         # self.mpl.plot_graph()
#         # self.mpl.plot_graph1()
#         # self.mpl_ntb = NavigationToolbar(self.mpl, self)  # 添加完整的 toolbar
#         self.layout.addWidget(self.mpl)
#         # self.layout.addWidget(self.mpl_ntb)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MatplotlibWidget()
    # ui.mpl.plot_graph()
    # ui.mpl.plot_graph1()
    ui.show()
    sys.exit(app.exec_())



