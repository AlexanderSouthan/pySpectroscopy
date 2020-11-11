# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget, QLineEdit,
                             QFileDialog, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QTextEdit, QLabel, QPushButton,
                             QDesktopWidget)

# import own modules ################
#from gui_objects.plot_canvas import plot_canvas
from pyAnalytics.raman_data import raman_image
from pyAnalytics.spectroscopy_data import spectroscopy_data
# from hplc_calibration_window import hplc_calibration_window
# from hplc_visualization_window import hplc_visualization_window
#####################################


class raman_export_window(QMainWindow):

    def __init__(self, raman_data):
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.update_data(raman_data)
        self.connect_event_handlers()

    def init_window(self):

        self.setGeometry(500, 500, 600, 600)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('Raman data export wizard')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.export_options_label = QLabel('<b>Export options</b>')
        self.export_selection_combobox = QComboBox(self.container0)
        self.button_export_path = QPushButton(
            'Select export path', self.container0)
        self.lineedit_export_path = QLineEdit(self.container0)

        self.select_data_combo = QComboBox()
        self.select_column_combo = QComboBox()

        self.button_start_export = QPushButton("Start export", self.container0)

    def position_widgets(self):
        self.export_path_layout = QHBoxLayout()
        self.export_path_layout.addWidget(self.button_export_path)
        self.export_path_layout.addWidget(self.lineedit_export_path)

        self.monochrome_export_layout = QHBoxLayout()
        self.monochrome_export_layout.addWidget(self.select_data_combo)
        self.monochrome_export_layout.addWidget(self.select_column_combo)

        self.export_data_layout = QVBoxLayout()
        self.export_data_layout.addLayout(self.export_path_layout)
        self.export_data_layout.addWidget(self.export_options_label)
        self.export_data_layout.addWidget(self.export_selection_combobox)
        self.export_data_layout.addLayout(self.monochrome_export_layout)
        self.export_data_layout.addWidget(self.button_start_export)
        self.export_data_layout.addStretch(1)

        self.grid_container.addLayout(self.export_data_layout, 0, 1, 1, 1)

    def connect_event_handlers(self):
        self.button_export_path.clicked.connect(self.get_export_path)
        self.button_start_export.clicked.connect(self.start_export)
        self.select_data_combo.currentIndexChanged.connect(
            self.update_column_combo)
        self.export_selection_combobox.currentIndexChanged.connect(
            self.set_active_combos)

    def update_data(self, raman_data):
        self.raman_data = raman_data

        self.export_selection_combobox.clear()
        if type(self.raman_data) is raman_image:
            self.export_selection_combobox.addItems(
                ['preprocessed spectra', 'raw spectra', 'monochrome image',
                 'x-stack', 'y-stack', 'z-stack', 'projections'])
        elif type(self.raman_data) is spectroscopy_data:
            self.export_selection_combobox.addItems(
                ['preprocessed spectra', 'raw spectra'])

        self.select_data_combo.clear()
        self.select_data_combo.addItems(self.raman_data.monochrome_data.keys())
        self.select_data_combo.setCurrentIndex(0)

        self.set_active_combos()
        self.update_column_combo()

    def set_active_combos(self):
        if self.export_selection_combobox.currentText() in [
                'monochrome image', 'projections', 'x-stack', 'y-stack',
                'z-stack']:
            self.select_data_combo.setEnabled(True)
            self.select_column_combo.setEnabled(True)
        else:
            self.select_data_combo.setEnabled(False)
            self.select_column_combo.setEnabled(False)

    def update_column_combo(self):
        curr_data = self.select_data_combo.currentText()
        self.select_column_combo.clear()
        if curr_data != '':
            self.select_column_combo.addItems(
                self.raman_data.monochrome_data[curr_data].columns.astype(str).to_list())
            self.select_column_combo.setCurrentIndex(0)

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_export_path(self):
        self.export_path = str(
            QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.lineedit_export_path.setText(self.export_path)

    def start_export(self):
        export_selection = self.export_selection_combobox.currentText()
        if export_selection in ['preprocessed spectra', 'raw spectra']:
            if export_selection == 'preprocessed spectra':
                active_dataset = self.raman_data.spectral_data_processed
                file_name = 'raman_data_processed'
            elif export_selection == 'raw spectra':
                active_dataset = self.raman_data.spectral_data
                file_name = 'raman_data_raw'

            if type(self.raman_data) is raman_image:
                active_dataset = self.raman_data._confocal_data__decode_image_index(active_dataset)

            self.raman_data.export_spectra(self.export_path + '/',
                                           file_name,
                                           active_data=active_dataset)
        elif export_selection == 'monochrome image':
            dataset = self.select_data_combo.currentText()
            col_index = self.select_column_combo.currentText()
            self.raman_data.export_monochrome_image(dataset, col_index,
                self.export_path + '/', 'monochrome_image_' + dataset + '_' + col_index)
        elif export_selection == 'projections':
            dataset = self.select_data_combo.currentText()
            col_index = self.select_column_combo.currentText()
            self.raman_data.export_intensity_projections(
                dataset, col_index, self.export_path + '/')
        elif export_selection == 'x-stack':
            dataset = self.select_data_combo.currentText()
            col_index = self.select_column_combo.currentText()
            self.raman_data.export_stack(
                dataset, col_index, self.export_path + '/', axis='x')
        elif export_selection == 'y-stack':
            dataset = self.select_data_combo.currentText()
            col_index = self.select_column_combo.currentText()
            self.raman_data.export_stack(
                dataset, col_index, self.export_path + '/', axis='y')
        elif export_selection == 'z-stack':
            dataset = self.select_data_combo.currentText()
            col_index = self.select_column_combo.currentText()
            self.raman_data.export_stack(
                dataset, col_index, self.export_path + '/', axis='z')
