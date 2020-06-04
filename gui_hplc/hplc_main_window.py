# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:40:26 2019

@author: AlMaMi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import copy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QComboBox, QWidget,
                             QLineEdit, QFileDialog, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QTextEdit, QLabel, QToolTip, QAction,
                             qApp, QPushButton, QDesktopWidget, QCheckBox)

# import own modules ################
import pyAnalytics.raman_data as raman
#from gui_objects.plot_canvas import plot_canvas
from pyAnalytics.hplc_data import hplc_data
from hplc_import_window import hplc_import_window
from hplc_calibration_window import hplc_calibration_window
from hplc_visualization_window import hplc_visualization_window
#from spectra_fit_viewer import spectra_fit_viewer
#from pca_viewer import pca_viewer
#from options_viewer import options_viewer
#####################################


class main_window(QMainWindow):

    def __init__(self):
        super().__init__()  # constructor of for parent object class is called
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

        self.hplc_datasets = {}
        self.hplc_file_names = {}

    def init_window(self):
        self.setGeometry(500, 500, 1200, 100)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('HPLC data analysis')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.import_wizard_button = QPushButton('Import wizard')
        self.visualize_data_button = QPushButton('Visualize data')
        # Interesting preprocessing options: Baseline correction, normalize
        # with internal standard. Should both be integrated into hplc_data.
        self.open_preprocessing_button = QPushButton('Data preprocessing')
        self.open_calibration_button = QPushButton('Calibration wizard')
        self.export_wizard_button = QPushButton('Export wizard')
        self.analyze_data_button = QPushButton('Analyze data')

        self.dataset_selection_label = QLabel('<b>Active dataset</b>')
        self.dataset_selection_combo = QComboBox()
        # self.dataset_selection_combo.addItems(['import'])

    def position_widgets(self):
        self.dataset_selection_layout = QVBoxLayout()
        self.dataset_selection_layout.addWidget(self.dataset_selection_label)
        self.dataset_selection_layout.addWidget(self.dataset_selection_combo)

        self.hplc_windows_buttons_layout = QHBoxLayout()
        self.hplc_windows_buttons_layout.addWidget(self.import_wizard_button)
        self.hplc_windows_buttons_layout.addWidget(self.visualize_data_button)
        self.hplc_windows_buttons_layout.addWidget(self.open_preprocessing_button)
        self.hplc_windows_buttons_layout.addWidget(self.open_calibration_button)
        self.hplc_windows_buttons_layout.addWidget(self.export_wizard_button)
        self.hplc_windows_buttons_layout.addWidget(self.analyze_data_button)
        self.hplc_windows_buttons_layout.addLayout(self.dataset_selection_layout)



        self.grid_container.addLayout(self.hplc_windows_buttons_layout, 0, 1, 1, 1)

        # self.grid_container.addWidget(self.import_options_label, *(1, 1), 1, 1)
        # self.spectra_plot_limits_layout.addStretch(1)

    def connect_event_handlers(self):
        self.import_wizard_button.clicked.connect(self.open_import_window)
        self.open_calibration_button.clicked.connect(self.open_calibration_window)
        self.visualize_data_button.clicked.connect(self.open_visualization_window)

        self.dataset_selection_combo.currentIndexChanged.connect(self.update_windows)
        # self.button_import_path.clicked.connect(self.get_import_path)
        # self.button_start_import.clicked.connect(self.start_import)

    def open_calibration_window(self):
        self.hplc_calibration_window = hplc_calibration_window(self)
        self.hplc_calibration_window.show()

    def open_visualization_window(self):
        self.hplc_visualization_window = hplc_visualization_window(self)
        self.hplc_visualization_window.show()

    def open_import_window(self):
        self.hplc_import_window = hplc_import_window(self)
        self.hplc_import_window.show()

    def update_windows(self):
        try:
            self.hplc_visualization_window.set_active_dataset()
        except:
            pass
        try:
            self.hplc_calibration_window.set_active_dataset()
        except:
            pass

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


app = QApplication(sys.argv)
   
window = main_window()

window.show()
#app.exec_()
sys.exit(app.exec_())
   


