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


class raman_import_window(QMainWindow):

    def __init__(self, data_container):
        super().__init__()
        self.data_container = data_container
        self.init_window()
        self.define_widgets()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):

        self.setGeometry(500, 500, 600, 600)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('Raman data import wizard')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.scan_type_label = QLabel(self.container0)
        self.scan_type_label.setText('Data format')
        self.scan_type_combobox = QComboBox(self.container0)
        self.scan_type_combobox.addItems(
            ['independent spectra', 'z-scan', 'y-scan', 'x-scan',
             'volume scan'])
            # ['independent spectra', 'z-scan', 'y-scan', 'x-scan', 'volume scan',
            #  'Inline-IR', 'LSM'])
        self.add_import_data_button = QPushButton('Add data files for import')
        self.clear_import_data_button = QPushButton('Clear')
        self.import_data_textedit = QTextEdit(readOnly=True)
        self.import_dataset_name_label = QLabel('<b>Dataset name</b>')
        self.import_dataset_name_lineedit = QLineEdit('import')
        self.import_data_button = QPushButton('Import data')

    def position_widgets(self):
        self.scan_type_layout = QHBoxLayout()
        self.scan_type_layout.addWidget(self.scan_type_label)
        self.scan_type_layout.addWidget(self.scan_type_combobox)

        self.import_dataset_name_layout = QHBoxLayout()
        self.import_dataset_name_layout.addWidget(
            self.import_dataset_name_label)
        self.import_dataset_name_layout.addWidget(
            self.import_dataset_name_lineedit)

        self.import_data_layout = QVBoxLayout()
        self.import_data_layout.addLayout(self.scan_type_layout)
        self.import_data_layout.addWidget(self.add_import_data_button)
        self.import_data_layout.addWidget(self.clear_import_data_button)
        self.import_data_layout.addWidget(self.import_data_textedit)
        self.import_data_layout.addLayout(self.import_dataset_name_layout)
        self.import_data_layout.addWidget(self.import_data_button)

        self.grid_container.addLayout(self.import_data_layout, 0, 1, 1, 1)

    def connect_event_handlers(self):
        self.add_import_data_button.clicked.connect(self.add_import_files)
        self.clear_import_data_button.clicked.connect(
            self.import_data_textedit.clear)
        self.import_data_button.clicked.connect(self.import_data)

    def add_import_files(self):
        import_files = QFileDialog.getOpenFileNames(
            self, "Select data files for import")[0]
        for curr_file in import_files:
            self.import_data_textedit.append(curr_file)

    def import_data(self):
        selected_data_format = self.scan_type_combobox.currentText()

        if selected_data_format in ['z-scan', 'volume scan', 'x-scan',
                                    'y-scan', 'independent spectra']:
            data_source = 'import'
            spectral_data = None
            file_extension = 'txt'
            decimals_coordinates = 1
            if self.scan_type_combobox.currentText() == 'z-scan':
                measurement_type = 'Raman_z_scan'
            elif self.scan_type_combobox.currentText() == 'volume scan':
                measurement_type = 'Raman_volume'
            elif self.scan_type_combobox.currentText() == 'x-scan':
                measurement_type = 'Raman_x_scan'
            elif self.scan_type_combobox.currentText() == 'y-scan':
                measurement_type = 'Raman_y_scan'
            elif self.scan_type_combobox.currentText() == 'single spectrum':
                measurement_type = 'Raman_single_spectrum'
        # elif selected_data_format == 'Inline-IR':
        #     measurement_type = 'Inline_IR'
        #     data_source = 'import'
        #     spectral_data = None
        #     file_extension = 'spc'
        #     decimals_coordinates = 1
        # elif selected_data_format == 'LSM':
        #     data_source = 'import'
        #     spectral_data = None
        #     file_extension = 'tif'
        #     measurement_type = 'LSM'
        #     decimals_coordinates = 0
        else:
            raise ValueError('Unknown import type.')

        import_dataset_name = self.import_dataset_name_lineedit.text()
        import_file_names = self.import_data_textedit.toPlainText().split('\n')

        if selected_data_format in ['independent spectra']:
            self.data_container[import_dataset_name] = spectroscopy_data(
                data_source=data_source,
                file_names=import_file_names)
        elif selected_data_format in ['z-scan', 'volume scan', 'x-scan',
                                      'y-scan']:
            self.data_container[import_dataset_name] = raman_image(
                measurement_type=measurement_type,
                file_names=import_file_names,
                file_extension=file_extension, spectral_data=spectral_data,
                data_source=data_source,
                decimals_coordinates=decimals_coordinates)

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
