# -*- coding: utf-8 -*-

import numpy as np
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget,
                             QLineEdit, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QLabel, QToolTip,
                             QDesktopWidget, QPushButton)

from gui_objects.plot_canvas import plot_canvas
from pyAnalytics.spectroscopy_data import spectroscopy_data
from pyAnalytics.raman_data import raman_image


class raman_visualization_window(QMainWindow):

    def __init__(self, raman_data):
        super().__init__()
        self.init_window()
        self.define_widgets(raman_data)
        self.position_widgets(raman_data)

        self.spectra_plots_active = True
        self.replace_data(raman_data)

        self.connect_event_handlers()

    def init_window(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        self.setToolTip('Program for <b>confocal data</b> analysis')

        self.setGeometry(500, 500, 1200, 900)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('Raman data visualization')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self, raman_data):
        self.spectra_plot = plot_canvas(
            plot_title='Raman spectra',
            x_axis_title = 'wavenumber ' + '$\mathrm{[cm^{-1}]}$')
        self.spectra_plot_edited = plot_canvas(
            plot_title='Raman spectra (edited)',
            x_axis_title = 'wavenumber ' + '$\mathrm{[cm^{-1}]}$')

        self.show_spectra_combo = QComboBox()
        self.show_spectra_combo.addItems(['Single spectrum', 'All spectra'])
        self.show_baseline_combo = QComboBox()
        self.lower_wn_label = QLabel('Lower wavenumber limit', self.container0)
        self.lower_wn_lineedit = QLineEdit()
        self.upper_wn_label = QLabel('Upper wavenumber limit', self.container0)
        self.upper_wn_lineedit = QLineEdit()
        self.draw_vertical_lines_label = QLabel(
            'Draw vertical lines', self.container0)
        self.draw_vertical_lines_lineedit = QLineEdit(self.container0)

        self.x_coord_label = QLabel('x-coordinate')
        self.x_coord_edited_label = QLabel('x-coordinate')
        self.x_coord_combo = QComboBox()
        self.x_coord_combo.setEnabled(False)
        self.x_coord_combo_edited = QComboBox()
        self.x_coord_combo_edited.setEnabled(False)
        self.y_coord_label = QLabel('y-coordinate')
        self.y_coord_edited_label = QLabel('y-coordinate')
        self.y_coord_combo = QComboBox()
        self.y_coord_combo.setEnabled(False)
        self.y_coord_combo_edited = QComboBox()
        self.y_coord_combo_edited.setEnabled(False)
        self.z_coord_label = QLabel('z-coordinate')
        self.z_coord_edited_label = QLabel('z-coordinate')
        self.z_coord_combo = QComboBox()
        self.z_coord_combo.setEnabled(False)
        self.z_coord_combo_edited = QComboBox()
        self.z_coord_combo_edited.setEnabled(False)
        self.file_name_label = QLabel('File names')
        self.file_name_combo = QComboBox()
        self.file_name_combo.setEnabled(False)

        self.update_data_button = QPushButton('Update')

    def position_widgets(self, raman_data):
        self.grid_container.addWidget(self.spectra_plot, *(1, 0), 1, 1)
        self.grid_container.addWidget(self.spectra_plot_edited, *(2, 0), 1, 1)

        self.show_spectra_layout = QHBoxLayout()
        self.show_spectra_layout.addWidget(self.show_spectra_combo)
        self.show_spectra_layout.addWidget(self.show_baseline_combo)

        self.wavenumber_limits_labels_layout = QHBoxLayout()
        self.wavenumber_limits_labels_layout.addWidget(self.lower_wn_label)
        self.wavenumber_limits_labels_layout.addWidget(self.lower_wn_lineedit)

        self.wavenumber_limits_combos_layout = QHBoxLayout()
        self.wavenumber_limits_combos_layout.addWidget(self.upper_wn_label)
        self.wavenumber_limits_combos_layout.addWidget(self.upper_wn_lineedit)

        self.draw_vertical_lines_layout = QHBoxLayout()
        self.draw_vertical_lines_layout.addWidget(
            self.draw_vertical_lines_label)
        self.draw_vertical_lines_layout.addWidget(
            self.draw_vertical_lines_lineedit)

        self.plot_options_layout = QVBoxLayout()
        self.plot_options_layout.addLayout(self.show_spectra_layout)
        self.plot_options_layout.addLayout(
            self.wavenumber_limits_labels_layout)
        self.plot_options_layout.addLayout(
            self.wavenumber_limits_combos_layout)
        self.plot_options_layout.addLayout(self.draw_vertical_lines_layout)

        self.sample_selection_layout = QVBoxLayout()

        self.sample_spinboxes_layout = QHBoxLayout()
        self.sample_spinboxes_layout.addWidget(self.x_coord_label)
        self.sample_spinboxes_layout.addWidget(self.x_coord_combo)
        self.sample_spinboxes_layout.addWidget(self.y_coord_label)
        self.sample_spinboxes_layout.addWidget(self.y_coord_combo)
        self.sample_spinboxes_layout.addWidget(self.z_coord_label)
        self.sample_spinboxes_layout.addWidget(self.z_coord_combo)

        self.processed_spinboxes_layout = QHBoxLayout()
        self.processed_spinboxes_layout.addWidget(
            self.x_coord_edited_label)
        self.processed_spinboxes_layout.addWidget(
            self.x_coord_combo_edited)
        self.processed_spinboxes_layout.addWidget(
            self.y_coord_edited_label)
        self.processed_spinboxes_layout.addWidget(
            self.y_coord_combo_edited)
        self.processed_spinboxes_layout.addWidget(
            self.z_coord_edited_label)
        self.processed_spinboxes_layout.addWidget(
            self.z_coord_combo_edited)

        self.sample_selection_layout.addLayout(
            self.sample_spinboxes_layout)
        self.sample_selection_layout.addLayout(
            self.processed_spinboxes_layout)

        self.sample_selection_layout.addWidget(self.file_name_label)
        self.sample_selection_layout.addWidget(self.file_name_combo)

        self.update_data_layout = QHBoxLayout()
        self.update_data_layout.addStretch(1)
        self.update_data_layout.addWidget(self.update_data_button)

        self.spectra_plot_limits_layout = QHBoxLayout()
        self.spectra_plot_limits_layout.addLayout(self.plot_options_layout)
        self.spectra_plot_limits_layout.addStretch(1)
        self.spectra_plot_limits_layout.addLayout(
            self.sample_selection_layout)
        self.spectra_plot_limits_layout.addStretch(1)
        self.spectra_plot_limits_layout.addLayout(self.update_data_layout)

        self.grid_container.addLayout(self.spectra_plot_limits_layout,
                                      *(0, 0), 1, 1)

    def connect_event_handlers(self):
        self.show_baseline_combo.currentIndexChanged.connect(
            self.update_spectra_plots)
        self.lower_wn_lineedit.editingFinished.connect(
            self.update_spectra_plots)
        self.upper_wn_lineedit.editingFinished.connect(
            self.update_spectra_plots)
        self.draw_vertical_lines_lineedit.editingFinished.connect(
            self.update_spectra_plots)
        self.show_spectra_combo.currentIndexChanged.connect(
            self.switch_spectrum_mode)
        self.x_coord_combo.currentIndexChanged.connect(
            self.update_spectra_plot)
        self.y_coord_combo.currentIndexChanged.connect(
            self.update_spectra_plot)
        self.z_coord_combo.currentIndexChanged.connect(
            self.update_spectra_plot)
        self.x_coord_combo_edited.currentIndexChanged.connect(
            self.update_spectra_edited_plot)
        self.y_coord_combo_edited.currentIndexChanged.connect(
            self.update_spectra_edited_plot)
        self.z_coord_combo_edited.currentIndexChanged.connect(
            self.update_spectra_edited_plot)
        self.file_name_combo.currentIndexChanged.connect(
            self.update_spectra_plots)
        
        self.update_data_button.clicked.connect(self.update_data)

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_data(self):
        self.replace_data(self.raman_data)

    def replace_data(self, raman_data):
        # Disables the update_spectra_plots function.
        self.spectra_plots_active = False

        self.raman_data = raman_data
        self.wavenumbers = self.raman_data.spectral_data.columns.to_numpy()

        self.lower_wn_lineedit.setText(str(self.wavenumbers[0]))
        self.upper_wn_lineedit.setText(str(self.wavenumbers[-1]))

        self.show_baseline_combo.clear()
        self.show_baseline_combo.addItems(self.raman_data.baseline_data.keys())

        self.initialize_coord_combos()
        self.change_active_coord_combos()

        # Now the update_spectra_plots function will do something.
        self.spectra_plots_active = True

        self.update_spectra_plots()

    def update_spectra_plots(self):
        if self.spectra_plots_active is False:
            return

        self.update_spectra_plot()
        self.update_spectra_edited_plot()

    def update_spectra_plot(self):
        if self.spectra_plots_active is False:
            return

        baseline_option = self.show_baseline_combo.currentText()

        lower_wn = float(self.lower_wn_lineedit.text())
        upper_wn = float(self.upper_wn_lineedit.text())

        v_line_string = self.draw_vertical_lines_lineedit.text()
        v_line_coords = v_line_string.split(',') if v_line_string != '' else []

        self.spectra_plot_data = self.raman_data.spectral_data
        if baseline_option != '':
            baseline_plot_data = self.raman_data.baseline_data[baseline_option]
        else:
            baseline_plot_data = None

        # Get index values for wavenumber limits in plot
        closest_index_to_lower_wn = np.argmin(np.abs(
            self.spectra_plot_data.columns-lower_wn))
        closest_index_to_upper_wn = np.argmin(np.abs(
            self.spectra_plot_data.columns-upper_wn))
        if baseline_plot_data is not None:
            closest_index_to_lower_wn_baseline = np.argmin(np.abs(
                baseline_plot_data.columns-lower_wn))
            closest_index_to_upper_wn_baseline = np.argmin(np.abs(
                baseline_plot_data.columns-upper_wn))

        if self.show_spectra_combo.currentText() == 'Single spectrum':
            if type(self.raman_data) is raman_image:
                self.spectra_plot_data = self.spectra_plot_data.loc[
                    [(self.get_coord('coded', axis='x'),
                      self.get_coord('coded', axis='y'),
                      self.get_coord('coded', axis='z'))], :]
                if baseline_plot_data is not None:
                    baseline_plot_data = baseline_plot_data.loc[
                        [(self.get_coord('coded', axis='x'),
                          self.get_coord('coded', axis='y'),
                          self.get_coord('coded', axis='z'))], :]
            elif type(self.raman_data) is spectroscopy_data:
                self.spectra_plot_data = self.spectra_plot_data.loc[
                    [self.file_name_combo.currentText()], :]
                if baseline_plot_data is not None:
                    baseline_plot_data = baseline_plot_data.loc[
                        [self.file_name_combo.currentText()], :]

        min_intensity = self.spectra_plot_data.iloc[
            :, closest_index_to_lower_wn:closest_index_to_upper_wn+1
            ].values.min()
        max_intensity = self.spectra_plot_data.iloc[
            :, closest_index_to_lower_wn:closest_index_to_upper_wn+1
            ].values.max()
        if baseline_plot_data is not None:
            min_baseline = baseline_plot_data.iloc[
                :, closest_index_to_lower_wn_baseline:
                    closest_index_to_upper_wn_baseline+1].values.min()
            max_baseline = baseline_plot_data.iloc[
                :, closest_index_to_lower_wn_baseline:
                    closest_index_to_upper_wn_baseline+1].values.max()
        else:
            min_baseline = min_intensity
            max_baseline = max_intensity
        min_y = min(min_intensity, min_baseline)
        max_y = max(max_intensity, max_baseline)

        # curr_baseline_wn = baseline_plot_data.columns

        self.spectra_plot.axes.clear()
        self.spectra_plot.plot(self.raman_data.wavenumbers,
                               self.spectra_plot_data.values.T, pen='b')
        if baseline_plot_data is not None:
            self.spectra_plot.plot(baseline_plot_data.columns,
                                   baseline_plot_data.values.T, pen='b')
        for xc in v_line_coords:
            self.spectra_plot.axes.axvline(x=float(xc),color='k')
        self.spectra_plot.axes.set_xlim(left=lower_wn,right=upper_wn)
        self.spectra_plot.axes.set_ylim(bottom=min_y,top=max_y)
        self.spectra_plot.draw()

    def update_spectra_edited_plot(self):
        if self.spectra_plots_active is False:
            return

        lower_wn = float(self.lower_wn_lineedit.text())
        upper_wn = float(self.upper_wn_lineedit.text())

        v_line_string = self.draw_vertical_lines_lineedit.text()
        v_line_coords = v_line_string.split(',') if v_line_string != '' else []

        self.plot_data_edited = self.raman_data.spectral_data_processed

        closest_index_to_lower_wn = np.argmin(
            np.abs(self.plot_data_edited.columns-lower_wn))
        closest_index_to_upper_wn = np.argmin(
            np.abs(self.plot_data_edited.columns-upper_wn))

        if self.show_spectra_combo.currentText() == 'Single spectrum':
            if type(self.raman_data) is raman_image:
                self.plot_data_edited = self.plot_data_edited.loc[
                    [(self.get_coord('coded', axis='x', mode='processed'),
                      self.get_coord('coded', axis='y', mode='processed'),
                      self.get_coord('coded', axis='z', mode='processed'))]]
            elif type(self.raman_data) is spectroscopy_data:
                self.plot_data_edited = self.plot_data_edited.loc[
                    [self.file_name_combo.currentText()], :]

        min_intensity = self.plot_data_edited.iloc[
            :, closest_index_to_lower_wn:closest_index_to_upper_wn
            ].values.min()
        max_intensity = self.plot_data_edited.iloc[
            :, closest_index_to_lower_wn:closest_index_to_upper_wn
            ].values.max()
        curr_wavenumbers = self.plot_data_edited.columns

        self.spectra_plot_edited.axes.clear()
        self.spectra_plot_edited.plot(curr_wavenumbers,
                                      self.plot_data_edited.values.T, pen='b')
        for xc in v_line_coords:
            self.spectra_plot_edited.axes.axvline(x=float(xc), color='k')
        self.spectra_plot_edited.axes.set_xlim(left=lower_wn, right=upper_wn)
        self.spectra_plot_edited.axes.set_ylim(bottom=min_intensity, 
                                               top=max_intensity)
        self.spectra_plot_edited.draw()

    def initialize_coord_combos(self):
        # Disables the update_spectra_plots function.
        self.spectra_plots_active = False

        self.x_coord_combo.clear()
        self.y_coord_combo.clear()
        self.z_coord_combo.clear()
        self.x_coord_combo_edited.clear()
        self.y_coord_combo_edited.clear()
        self.z_coord_combo_edited.clear()
        self.file_name_combo.clear()

        if type(self.raman_data) is raman_image:
            self.x_coord_combo.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values('real', axis='x'), 1)))

            self.y_coord_combo.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values('real', axis='y'), 1)))

            self.z_coord_combo.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values('real', axis='z'), 1)))

            self.x_coord_combo_edited.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values(
                        'real', axis='x',
                        active_data=self.raman_data.spectral_data_processed),
                    1)))

            self.y_coord_combo_edited.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values(
                        'real', axis='y',
                        active_data=self.raman_data.spectral_data_processed),
                    1)))

            self.z_coord_combo_edited.addItems(
                np.char.mod('%s', np.around(
                    self.raman_data.get_coord_values(
                        'real', axis='z',
                        active_data=self.raman_data.spectral_data_processed),
                    1)))

        elif type(self.raman_data) is spectroscopy_data:
            self.file_name_combo.addItems(
                self.raman_data.spectral_data.index.values)

        # Now the update_spectra_plots function will do something.
        self.spectra_plots_active = True

    def change_active_coord_combos(self):
        if self.show_spectra_combo.currentText() == 'All spectra':
            self.x_coord_combo.setEnabled(False)
            self.y_coord_combo.setEnabled(False)
            self.z_coord_combo.setEnabled(False)
            self.x_coord_combo_edited.setEnabled(False)
            self.y_coord_combo_edited.setEnabled(False)
            self.z_coord_combo_edited.setEnabled(False)
            self.file_name_combo.setEnabled(False)
        else:
            if type(self.raman_data) is raman_image:
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'x')) > 1:
                    self.x_coord_combo.setEnabled(True)
                else:
                    self.x_coord_combo.setEnabled(False)
                    
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'y')) > 1:
                    self.y_coord_combo.setEnabled(True)
                else:
                    self.y_coord_combo.setEnabled(False)
                
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'z')) > 1:
                    self.z_coord_combo.setEnabled(True)
                else:
                    self.z_coord_combo.setEnabled(False)
                    
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'x',
                        active_data=self.raman_data.spectral_data_processed)
                        ) > 1:
                    self.x_coord_combo_edited.setEnabled(True)
                else:
                    self.x_coord_combo_edited.setEnabled(False)
                    
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'y',
                        active_data=self.raman_data.spectral_data_processed)
                        ) > 1:
                    self.y_coord_combo_edited.setEnabled(True)
                else:
                    self.y_coord_combo_edited.setEnabled(False)
                
                if len(self.raman_data.get_coord_values(
                        'coded', axis = 'z',
                        active_data=self.raman_data.spectral_data_processed)
                        ) > 1:
                    self.z_coord_combo_edited.setEnabled(True)
                else:
                    self.z_coord_combo_edited.setEnabled(False)
            elif type(self.raman_data) is spectroscopy_data:
                self.file_name_combo.setEnabled(True)

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

    def switch_spectrum_mode(self):
        self.change_active_coord_combos()
        self.update_spectra_plots()
