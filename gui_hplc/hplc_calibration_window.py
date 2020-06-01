# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:44:12 2019

@author: AlMaMi
"""

# from gui_objects.plot_canvas import plot_canvas
# from gui_objects.image_canvas import image_canvas
from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget, QGridLayout,
                             QDesktopWidget, QLabel, QPushButton, QTextEdit,
                             QHBoxLayout, QVBoxLayout, QFileDialog, QLineEdit)

# from pyAnalytics.hplc_data import hplc_data


class hplc_calibration(QMainWindow):
    def __init__(self,parent):
        super().__init__()
        self.parent = parent
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

        # self.init_combos()
        
    def init_window(self):
        self.setGeometry(500,500,1200,900) #xPos,yPos,width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('HPLC calibration wizard')
        
        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)
        
        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)
        
    def define_widgets(self):
        self.wl_limits_label = QLabel('Wavelength limits')
        self.time_limits_label = QLabel('Time limits')
        self.calibration_concentrations_label = QLabel('Calibration concentrations')
        self.wl_limits_low_lineedit = QLineEdit()
        self.wl_limits_high_lineedit = QLineEdit()
        self.time_limits_low_lineedit = QLineEdit()
        self.time_limits_high_lineedit = QLineEdit()
        self.calibration_concentrations_lineedit = QLineEdit(readOnly=False)
        
    def position_widgets(self):

        self.calibration_limits_label_layout = QVBoxLayout()
        self.calibration_limits_label_layout.addWidget(self.wl_limits_label)
        self.calibration_limits_label_layout.addWidget(self.time_limits_label)
        self.calibration_limits_label_layout.addWidget(self.calibration_concentrations_label)
        
        self.wl_limits_lineedit_layout = QHBoxLayout()
        self.wl_limits_lineedit_layout.addWidget(self.wl_limits_low_lineedit)
        self.wl_limits_lineedit_layout.addWidget(self.wl_limits_high_lineedit)

        self.time_limits_lineedit_layout = QHBoxLayout()
        self.time_limits_lineedit_layout.addWidget(self.time_limits_low_lineedit)
        self.time_limits_lineedit_layout.addWidget(self.time_limits_high_lineedit)
        
        self.calibration_limits_lineedit_layout = QVBoxLayout()
        self.calibration_limits_lineedit_layout.addLayout(self.wl_limits_lineedit_layout)
        self.calibration_limits_lineedit_layout.addLayout(self.time_limits_lineedit_layout)
        self.calibration_limits_lineedit_layout.addWidget(self.calibration_concentrations_lineedit)

        self.calibration_limits_layout = QHBoxLayout()
        self.calibration_limits_layout.addLayout(self.calibration_limits_label_layout)
        self.calibration_limits_layout.addLayout(self.calibration_limits_lineedit_layout)

        # self.calibration_layout.addStretch(1)
        
        self.grid_container.addLayout(self.calibration_limits_layout, 0, 1, 1, 1)

    def connect_event_handlers(self):
        pass
        # self.calibration_data_textedit.textChanged.connect(self.update_processing_parameters)

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())