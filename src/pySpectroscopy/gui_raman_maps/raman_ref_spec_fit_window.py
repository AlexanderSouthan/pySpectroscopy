# -*- coding: utf-8 -*-

import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QGridLayout, QDesktopWidget,
                             QLabel, QVBoxLayout, QPushButton, QHBoxLayout,
                             QTextEdit, QFileDialog)


class raman_ref_spec_fit_window(QMainWindow):
    def __init__(self, raman_data):
        self.update_data(raman_data)
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 600, 360) #xPos, yPos, width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('Reference spectrum fit')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.reference_spectra_label = QLabel('<b>Reference spectra</b>')
        self.add_reference_spectra_button = QPushButton(
            'Add reference spectrum path')
        self.clear_reference_spectra_button = QPushButton('Clear')
        self.reference_spectra_textedit = QTextEdit(readOnly=True)
        self.perform_ref_spec_fit_button = QPushButton(
            'Perform reference spectra fit')

    def position_widgets(self):
        self.reference_spectra_buttons_layout = QHBoxLayout()
        self.reference_spectra_buttons_layout.addWidget(
            self.add_reference_spectra_button)
        self.reference_spectra_buttons_layout.addWidget(
            self.clear_reference_spectra_button)

        self.imaging_layout = QVBoxLayout()
        self.imaging_layout.addWidget(self.reference_spectra_label)
        self.imaging_layout.addLayout(self.reference_spectra_buttons_layout)
        self.imaging_layout.addWidget(self.reference_spectra_textedit)
        self.imaging_layout.addWidget(self.perform_ref_spec_fit_button)
        self.imaging_layout.addStretch(1)

        self.grid_container.addLayout(self.imaging_layout, *(0, 0), 1, 1)

    def connect_event_handlers(self):
        self.add_reference_spectra_button.clicked.connect(
            self.add_reference_spectrum_path)
        self.clear_reference_spectra_button.clicked.connect(
            self.reference_spectra_textedit.clear)
        self.perform_ref_spec_fit_button.clicked.connect(
            self.perform_ref_spec_fit)

    def update_data(self, raman_data):
        self.raman_data = raman_data

    def add_reference_spectrum_path(self):
        import_files = QFileDialog.getOpenFileNames(
            self, "Select reference spectra")[0]
        for curr_file in import_files:
            self.reference_spectra_textedit.append(curr_file)

    def perform_ref_spec_fit(self):
        import_list = self.reference_spectra_textedit.toPlainText().split('\n')
        self.reference_spectra = np.empty(
            (len(import_list),
             len(self.raman_data.spectral_data_processed.columns)))
        for ii, current_file in enumerate(import_list):
           self.reference_spectra[ii, :] = np.fromfile(
               current_file, sep=' ')[1::2]

        self.raman_data.reference_spectra_fit(self.reference_spectra)

        print('Reference spectra fit finished!')

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
