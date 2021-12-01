# -*- coding: utf-8 -*-

import numpy as np
from gui_objects.plot_canvas import plot_canvas
from PyQt5.QtWidgets import (QMainWindow,QComboBox,QWidget,QGridLayout,
                             QDesktopWidget,QLabel,QVBoxLayout)

from pyAnalytics.raman_data import raman_image


class reference_spectra_fit_viewer(QMainWindow):
    def __init__(self, raman_data):
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()

        self.plots_active = True
        self.update_data(raman_data)
        self.init_combo_boxes()
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 1200, 900)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('Spectrum fit viewer')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.spectrum_plot = plot_canvas(
            plot_title='Spectrum plot', x_axis_title='wavenumber [cm-1]')

        self.x_coord_label = QLabel('x-coordinate')
        self.x_coord_combo = QComboBox()
        self.y_coord_label = QLabel('y-coordinate')
        self.y_coord_combo = QComboBox()
        self.z_coord_label = QLabel('z-coordinate')
        self.z_coord_combo = QComboBox()

        self.sample_label = QLabel('Sample')
        self.sample_combo = QComboBox()

    def position_widgets(self):
        self.coord_combos_layout = QVBoxLayout()
        self.coord_combos_layout.addWidget(self.x_coord_label)
        self.coord_combos_layout.addWidget(self.x_coord_combo)
        self.coord_combos_layout.addWidget(self.y_coord_label)
        self.coord_combos_layout.addWidget(self.y_coord_combo)
        self.coord_combos_layout.addWidget(self.z_coord_label)
        self.coord_combos_layout.addWidget(self.z_coord_combo)
        self.coord_combos_layout.addWidget(self.sample_label)
        self.coord_combos_layout.addWidget(self.sample_combo)
        self.coord_combos_layout.addStretch(1)

        self.grid_container.addLayout(self.coord_combos_layout,*(0, 0), 1, 1)
        self.grid_container.addWidget(self.spectrum_plot, *(0, 1), 1, 1)

    def connect_event_handlers(self):
        self.x_coord_combo.currentIndexChanged.connect(self.plot_spectrum)
        self.y_coord_combo.currentIndexChanged.connect(self.plot_spectrum)
        self.z_coord_combo.currentIndexChanged.connect(self.plot_spectrum)
        self.sample_combo.currentIndexChanged.connect(self.plot_spectrum)

    def update_data(self, raman_data):
        self.raman_data = raman_data
        self.init_combo_boxes()
        self.plot_spectrum()

    def init_combo_boxes(self):
        # Disables the update_spectra_plots function.
        self.plots_active = False

        self.sample_combo.clear()
        self.x_coord_combo.clear()
        self.y_coord_combo.clear()
        self.z_coord_combo.clear()

        if type(self.raman_data) is raman_image:
            self.x_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.raman_data.get_coord_values('real', axis='x'),1)))

            self.y_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.raman_data.get_coord_values('real', axis='y'),1)))

            self.z_coord_combo.addItems(
                np.char.mod('%s',np.around(
                    self.raman_data.get_coord_values('real', axis='z'),1)))

            if len(self.raman_data.get_coord_values('coded', axis='x')) > 1:
                self.x_coord_combo.setEnabled(True)
            else:
                self.x_coord_combo.setEnabled(False)
            if len(self.raman_data.get_coord_values('coded', axis='y')) > 1:
                self.y_coord_combo.setEnabled(True)
            else:
                self.y_coord_combo.setEnabled(False)
            if len(self.raman_data.get_coord_values('coded', axis='z')) > 1:
                self.z_coord_combo.setEnabled(True)
            else:
                self.z_coord_combo.setEnabled(False)

            self.sample_combo.setEnabled(False)
        else:
            self.x_coord_combo.setEnabled(False)
            self.y_coord_combo.setEnabled(False)
            self.z_coord_combo.setEnabled(False)
            self.sample_combo.setEnabled(True)
            self.sample_combo.addItems(
                self.raman_data.spectral_data.index.values)
            self.sample_combo.setCurrentIndex(0)

        # Now the update_spectra_plots function will do something.
        self.plots_active = True

    def plot_spectrum(self):
        if type(self.raman_data) is raman_image:
            selected_spectrum_index = (
                self.get_coord(
                    'coded',
                    coord_value=float(self.x_coord_combo.currentText())),
                self.get_coord(
                    'coded',
                    coord_value=float(self.y_coord_combo.currentText())),
                self.get_coord(
                    'coded',
                    coord_value=float(self.z_coord_combo.currentText())))
        else:
            selected_spectrum_index = self.sample_combo.currentText()
        
        self.spectrum_plot.axes.clear()
        self.spectrum_plot.plot(
            self.raman_data.spectral_data_processed.columns,
            self.raman_data.spectral_data_processed.loc[
                selected_spectrum_index, :], pen='k')
        self.spectrum_plot.plot(
            self.raman_data.fitted_spectra.columns,
            self.raman_data.fitted_spectra.loc[
                selected_spectrum_index, :], pen='r')

    def get_coord(self, value_sort, axis='x', coord_value=None,
                  mode='raw_data'):

        if coord_value is not None:
            if value_sort == 'coded':
                return_value = round(
                    coord_value*self.raman_data.coord_conversion_factor)
            else:
                return_value = (coord_value/
                                self.raman_data.coord_conversion_factor)
        else:
            if mode == 'raw_data':
                if axis == 'x':
                    return_value = float(self.x_coord_combo.currentText())
                elif axis == 'y':
                    return_value = float(self.y_coord_combo.currentText())
                elif axis == 'z':
                    return_value = float(self.z_coord_combo.currentText())
            elif mode == 'processed':
                if axis == 'x':
                    return_value = float(
                        self.x_coord_combo_edited.currentText())
                elif axis == 'y':
                    return_value = float(
                        self.y_coord_combo_edited.currentText())
                elif axis == 'z':
                    return_value = float(
                        self.z_coord_combo_edited.currentText())

            if value_sort == 'coded':
                return_value = round(
                    return_value*self.raman_data.coord_conversion_factor)

        return return_value

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
