import os
import sys
import cv2
import datetime
import numpy as np
from time import sleep
from PyQt5 import QtCore, QtGui, QtWidgets

from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

PATH="images/t3.jpg"

class Ui_MainWindow(object):
    def __init__(self):
        print("Initializing ....")
        if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
            print ('training data is ready, classifier is loading...')
        else:
            print ('training data is being created...')
            open('training.data', 'w')
            color_histogram_feature_extraction.training()
            print ('training data is ready, classifier is loading...')

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 621, 441))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.lineEdit.setMaxLength(255)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 2)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)

        self.dialog = QtWidgets.QFileDialog(MainWindow)
        self.dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self.fname = None
        self.image = None

        self.pushButton.clicked.connect(self.pushButton_clicked)
        self.pushButton_2.clicked.connect(self.pushButton2_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Acne Detection and Classifier"))
        self.groupBox.setTitle(_translate("MainWindow", "Image Processing"))
        self.pushButton.setText(_translate("MainWindow", "Browse .."))
        self.lineEdit.setText(_translate("MainWindow", "File Location"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "<Processed Image>"))
        self.label.setText(_translate("MainWindow", "<Original Image>"))

    def pushButton_clicked(self):
        try:
            if self.dialog.exec_():
                self.showImage(self.dialog.selectedFiles())

        except Exception as error:
            print(error)

    def pushButton2_clicked(self):
        if self.pushButton_2.text() == "Start":
            self.pushButton_2.setText("Reset")
            try:
                color_histogram_feature_extraction.color_histogram_of_test_image(self.image)
                prediction = knn_classifier.main('training.data', 'test.data')
                print(prediction)
                self.pushButton_2.setText("Start")
            except Exception as error:
                print(error)
        else:
            self.pushButton_2.setText("Start")

    def showImage(self, files):
        for file in files:
            #print(file)
            self.fname = file
            self.lineEdit.setText(file)
            self.image = cv2.imread(file, cv2.IMREAD_COLOR)
            qimage = QtGui.QImage(file)
            pixmap = QtGui.QPixmap.fromImage(qimage)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
            self.label_2.setPixmap(pixmap)
            self.label_2.setScaledContents(True)
            self.label.show()
            self.label_2.show()

            #sleep(10)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("cleanlooks")
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    #mainWindow.showFullScreen()
    mainWindow.show()
    sys.exit(app.exec_())