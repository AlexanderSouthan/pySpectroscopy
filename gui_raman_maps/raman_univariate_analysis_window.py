# -*- coding: utf-8 -*-

import copy
from PyQt5.QtWidgets import (QMainWindow,QComboBox,QWidget,QGridLayout,
                             QDesktopWidget,QLabel,QVBoxLayout,QLineEdit,
                             QPushButton,QHBoxLayout,QTextEdit,QFileDialog)

class raman_univariate_analysis_window(QMainWindow):
    def __init__(self, raman_data):
        self.update_data(raman_data)

        super().__init__()
        self.init_window()
        self.define_widgets()
        self.init_uva_parameters()
        self.set_option_values()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 600, 900) #xPos,yPos,width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('Univariate data analysis')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.imaging_label = QLabel('<b><h1>Univariate analysis options</h1></b>')
        self.int_at_point_label = QLabel('<b>Intensity at point</b>') 
        self.int_at_point_wavenumber_label = QLabel('wavenumber')
        self.int_at_point_wavenumber_lineedit = QLineEdit()

        self.sig_to_base_label = QLabel('<b>Signal to baseline/signal to axis</b>') 
        self.sig_to_base_lower_wavenumber_label = QLabel('Lower wavenumber limit')
        self.sig_to_base_lower_wavenumber_lineedit = QLineEdit()
        self.sig_to_base_upper_wavenumber_label = QLabel('Upper wavenumber limit')
        self.sig_to_base_upper_wavenumber_lineedit = QLineEdit()

        self.uva_selection_combo = QComboBox()
        self.uva_selection_combo.addItems(
            ['int_at_point', 'sig_to_base', 'sig_to_axis'])
        self.perform_uva_button = QPushButton('Perform univariate analysis')

        self.reset_button = QPushButton('Reset defaults')

    def position_widgets(self):
        self.int_at_point_layout = QHBoxLayout()
        self.int_at_point_layout.addWidget(self.int_at_point_wavenumber_label)
        self.int_at_point_layout.addWidget(self.int_at_point_wavenumber_lineedit)

        self.sig_to_base_lower_wn_layout = QHBoxLayout()
        self.sig_to_base_lower_wn_layout.addWidget(self.sig_to_base_lower_wavenumber_label)
        self.sig_to_base_lower_wn_layout.addWidget(self.sig_to_base_lower_wavenumber_lineedit)

        self.sig_to_base_upper_wn_layout = QHBoxLayout()
        self.sig_to_base_upper_wn_layout.addWidget(self.sig_to_base_upper_wavenumber_label)
        self.sig_to_base_upper_wn_layout.addWidget(self.sig_to_base_upper_wavenumber_lineedit)

        self.perform_uva_layout = QHBoxLayout()
        self.perform_uva_layout.addWidget(self.uva_selection_combo)
        self.perform_uva_layout.addWidget(self.perform_uva_button)

        self.imaging_layout = QVBoxLayout()
        self.imaging_layout.addWidget(self.imaging_label)
        self.imaging_layout.addWidget(self.int_at_point_label)
        self.imaging_layout.addLayout(self.int_at_point_layout)
        self.imaging_layout.addWidget(self.sig_to_base_label)
        self.imaging_layout.addLayout(self.sig_to_base_lower_wn_layout)
        self.imaging_layout.addLayout(self.sig_to_base_upper_wn_layout)
        self.imaging_layout.addLayout(self.perform_uva_layout)
        self.imaging_layout.addStretch(1)
        self.imaging_layout.addWidget(self.reset_button)

        self.grid_container.addLayout(self.imaging_layout, *(0,0),1,1)

    def connect_event_handlers(self):
        self.int_at_point_wavenumber_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.sig_to_base_lower_wavenumber_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.sig_to_base_upper_wavenumber_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.perform_uva_button.clicked.connect(
            self.perform_univariate_analysis)
        self.reset_button.clicked.connect(self.reset_defaults)

    def update_data(self, raman_data):
        self.raman_data = raman_data

    def init_uva_parameters(self):
        self.edit_args_dict = {
            'int_at_point': {'wn': 1600},
            'sig_to_base': {'wn': [1600, 1800]},
            'sig_to_axis': {'wn': [1600, 1800]}}

        self.edit_args_dict_default = copy.deepcopy(self.edit_args_dict)

    def set_option_values(self, mode='initial'):
        if mode == 'default':
            current_options_dict = self.edit_args_dict_default
        else:
            current_options_dict = self.edit_args_dict

        self.int_at_point_wavenumber_lineedit.setText(
            str(current_options_dict['int_at_point']['wn']))

        self.sig_to_base_lower_wavenumber_lineedit.setText(
            str(current_options_dict['sig_to_base']['wn'][0]))
        self.sig_to_base_upper_wavenumber_lineedit.setText(
            str(current_options_dict['sig_to_base']['wn'][1]))

    def perform_univariate_analysis(self):
        uva_type = self.uva_selection_combo.currentText()
        self.raman_data.univariate_analysis(
            uva_type, **self.edit_args_dict[uva_type])
        print('Current univariate analysis finished!')

    def reset_defaults(self):
        self.set_option_values(mode='default')

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_processing_parameters(self):
        self.edit_args_dict = {
            'int_at_point': {
                'wn':
                    float(self.int_at_point_wavenumber_lineedit.text())
                },
            'sig_to_base':{
                'wn':
                    [float(self.sig_to_base_lower_wavenumber_lineedit.text()),
                     float(self.sig_to_base_upper_wavenumber_lineedit.text())]
                    },
            'sig_to_axis':{
                'wn':
                    [float(self.sig_to_base_lower_wavenumber_lineedit.text()),
                     float(self.sig_to_base_upper_wavenumber_lineedit.text())]
                    }
                }
