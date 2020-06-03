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
from hplc_calibration_window import hplc_calibration_window
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

    def init_window(self):

        self.setGeometry(500, 500, 1200, 900)  # xPos,yPos,width, heigth
        self.center()  # center function is defined below
        self.setWindowTitle('HPLC data analysis')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.import_data_label = QLabel('<b>Import data</b>')
        self.add_import_data_button = QPushButton('Add data files for import')
        self.clear_import_data_button = QPushButton('Clear')
        self.import_data_textedit = QTextEdit(readOnly=True)
        self.import_data_button = QPushButton('Import data')

        # Interesting preprocessing options: Baseline correction, normalize
        # with internal standard. Should both be integrated into hplc_data.
        self.open_preprocessing_button = QPushButton('Data preprocessing')
        self.open_calibration_button = QPushButton('Send data to \n calibration wizard')
        
        self.import_options_label = QLabel('<b>Import options</b>')

    def position_widgets(self):

        self.import_data_layout = QVBoxLayout()
        self.import_data_layout.addWidget(self.import_data_label)
        self.import_data_layout.addWidget(self.add_import_data_button)
        self.import_data_layout.addWidget(self.clear_import_data_button)
        self.import_data_layout.addWidget(self.import_data_textedit)
        self.import_data_layout.addWidget(self.import_data_button)

        self.grid_container.addLayout(self.import_data_layout, 0, 1, 1, 1)
        self.grid_container.addWidget(self.open_preprocessing_button, 0, 2, 1, 1)
        self.grid_container.addWidget(self.open_calibration_button, 0, 3, 1, 1)
        # self.grid_container.addWidget(self.import_options_label, *(1, 1), 1, 1)
        # self.spectra_plot_limits_layout.addStretch(1)

    def connect_event_handlers(self):
        self.add_import_data_button.clicked.connect(self.add_import_data_path)
        self.clear_import_data_button.clicked.connect(self.import_data_textedit.clear)
        self.import_data_button.clicked.connect(self.import_data)

        self.open_calibration_button.clicked.connect(self.open_calibration_window)
        # self.button_import_path.clicked.connect(self.get_import_path)
        # self.button_start_import.clicked.connect(self.start_import)

    def add_import_data_path(self):
        import_files = QFileDialog.getOpenFileNames(
            self, "Select data files for import")[0]
        for curr_file in import_files:
            self.import_data_textedit.append(curr_file)

    def import_data(self):
        self.imported_data = []
        
        import_files = self.import_data_textedit.toPlainText().split('\n')
        print(import_files)
        
        for curr_file in import_files:
            self.imported_data.append(
                hplc_data('import', full_path=curr_file))
            plt.plot(self.imported_data[-1].extract_elugram(220))

        self.preprocessed_data = self.imported_data.copy()

    def open_calibration_window(self):
        self.hplc_calibration_window = hplc_calibration_window(self)
        self.hplc_calibration_window.show()

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # def check_spectrum_plot_state(self):
    #     self.spectra_plots_active = self.spectrum_plot_activate_checkbox.isChecked()

    #     if self.spectra_plots_active:
    #         self.update_spectra_plots()
    #         self.update_stats_edited_plot()
    #         self.update_stats_plot()

    # def init_processing_parameters(self):
    #     self.SNIP_string = 'SNIP'
    #     self.ALSS_string = 'ALSS'
    #     self.iALSS_string = 'iALSS'
    #     self.drPLS_string = 'drPLS'
    #     self.sav_gol_string = 'sav_gol'
    #     self.median_filter_string = 'median_filter'
    #     self.pca_smoothing_string = 'pca_smoothing'
    #     self.SNV_string = 'SNV'
    #     self.clip_wn_string = 'clip_wn'
    #     self.clip_samples_string = 'clip_samples'
    #     self.mean_center_string = 'mean_center'
    #     self.int_at_point_string = 'int_at_point'
    #     self.sig_to_base_string = 'sig_to_base'
    #     self.pca_string = 'pca'
    #     self.reference_spectra_fit_string = 'ref_spec_fit'

    #     self.edit_args_dict = {self.SNIP_string: {
    #                                 'n_iter': 100, 'alg': 'SNIP'},
    #                            self.ALSS_string: {
    #                                'lam': 10000, 'p': 0.001, 'n_iter': 10,
    #                                'alg': 'ALSS'},
    #                            self.iALSS_string: {
    #                                'lam': 2000, 'lam_1': 0.01, 'p': 0.01,
    #                                'n_iter': 10, 'alg': 'iALSS'},
    #                            self.drPLS_string: {
    #                                'lam': 1000000, 'eta': 0.5, 'n_iter': 100,
    #                                'alg': 'drPLS'},
    #                            self.sav_gol_string: {
    #                                'deriv': 0, 'savgol_points': 9},
    #                            self.median_filter_string: {'window': 5},
    #                            self.pca_smoothing_string: {
    #                                'pca_components': 3},
    #                            self.SNV_string: {},
    #                            self.clip_wn_string: {
    #                                'wn_limits': [[1100, 1500]]},
    #                            self.clip_samples_string: {
    #                                'x_limits': None, 'y_limits': None,
    #                                'z_limits': None},
    #                            self.mean_center_string: {},
    #                            self.int_at_point_string: {'wavenumber': 1600},
    #                            self.sig_to_base_string: {
    #                                'lower_wn': 1600, 'upper_wn': 1800},
    #                            self.reference_spectra_fit_string: {
    #                                'reference_spectra': '',
    #                                'fit_mode': 'Levenberg-Marquardt'},
    #                            self.pca_string: {'pca_components': 7}}

    #     self.edit_args_dict_default = copy.deepcopy(self.edit_args_dict)

    # def update_spectra_plot(self):
    #     if self.spectra_plots_active == False:
    #         return
        
    #     if self.show_baseline_combo.currentText() == 'Show SNIP baseline':
    #         baseline_option = 'SNIP'
    #     elif self.show_baseline_combo.currentText() == 'Show ALSS baseline':
    #         baseline_option = 'ALSS'
    #     elif self.show_baseline_combo.currentText() == 'Show iALSS baseline':
    #         baseline_option = 'iALSS'
    #     elif self.show_baseline_combo.currentText() == 'Show drPLS baseline':
    #         baseline_option = 'drPLS'
    #     else:
    #         baseline_option = 'none'
            
    #     lower_wn = float(self.lower_wn_lineedit.text())
    #     upper_wn = float(self.upper_wn_lineedit.text())
        
    #     v_line_string = self.draw_vertical_lines_lineedit.text()
    #     v_line_coords = v_line_string.split(',') if v_line_string != '' else []
        
    #     self.spectra_plot_data = self.raman_data.hyperspectral_image
    #     baseline_plot_data = self.raman_data.baseline_data[baseline_option]
        
    #     #get index values for wavenumber limits in plot
    #     closest_index_to_lower_wn = np.argmin(np.abs(self.spectra_plot_data.columns-lower_wn))
    #     closest_index_to_upper_wn = np.argmin(np.abs(self.spectra_plot_data.columns-upper_wn))
        
    #     if self.show_spectra_combo.currentText() == 'Single spectrum':
    #         self.spectra_plot_data = self.spectra_plot_data.loc[[(self.get_coord('coded',axis = 'x'),self.get_coord('coded',axis = 'y'),self.get_coord('coded',axis = 'z'))],:]
    #         baseline_plot_data = baseline_plot_data.loc[[(self.get_coord('coded',axis = 'x'),self.get_coord('coded',axis = 'y'),self.get_coord('coded',axis = 'z'))]]

    #     min_intensity = self.spectra_plot_data.iloc[:,closest_index_to_upper_wn:closest_index_to_lower_wn+1].values.min()
    #     max_intensity = self.spectra_plot_data.iloc[:,closest_index_to_upper_wn:closest_index_to_lower_wn+1].values.max()
            
    #     curr_baseline_wn = baseline_plot_data.columns
           
    #     self.spectra_plot.axes.clear()
    #     self.spectra_plot.plot(self.raman_data.wavenumbers,self.spectra_plot_data.values.T,pen='b')
    #     if baseline_option != 'none':
    #         self.spectra_plot.plot(curr_baseline_wn,baseline_plot_data.values.T,pen='b')
    #     for xc in v_line_coords:
    #         self.spectra_plot.axes.axvline(x=float(xc),color='k')
    #     self.spectra_plot.axes.set_xlim(left=lower_wn,right=upper_wn)
    #     self.spectra_plot.axes.set_ylim(bottom=min_intensity,top=max_intensity)
    #     self.spectra_plot.draw()

    # def update_stats_plot(self):
    #     if self.spectra_plots_active == False:
    #         return
        
    #     self.spectra_plot_baseline_stats.axes.clear()
    #     if self.spectra_plot_select_stats_combo.currentText() == self.descriptive_statistics_options[0]:
    #         self.spectra_plot_baseline_stats.plot(self.raman_data.mean_spectrum(),self.raman_data.hyperspectral_image.values.T,pen='b')
    #         self.spectra_plot_baseline_stats.axes.set_xlabel('mean spectrum intensities')
    #     elif self.spectra_plot_select_stats_combo.currentText() == self.descriptive_statistics_options[1]:
    #         self.spectra_plot_baseline_stats.plot(self.raman_data.wavenumbers,self.raman_data.mean_spectrum(),mode='error_bar',error_data=self.raman_data.std())
    #         self.spectra_plot_baseline_stats.axes.set_xlabel('wavenumber [1/cm]')
    #     elif self.spectra_plot_select_stats_combo.currentText() == self.descriptive_statistics_options[2]:
    #         self.spectra_plot_baseline_stats.plot(self.raman_data.wavenumbers,self.raman_data.mean_spectrum(),mode='error_bar',error_data=self.raman_data.std())
    #         self.spectra_plot_baseline_stats.axes.set_xlabel('wavenumber [1/cm]')
    #         self.spectra_plot_baseline_stats.plot(self.raman_data.wavenumbers,self.raman_data.max_spectrum())
    #         self.spectra_plot_baseline_stats.plot(self.raman_data.wavenumbers,self.raman_data.min_spectrum())
            
    #     self.spectra_plot_baseline_stats.draw()
    
    # def process_spectra(self):
    #     edit_string = self.edit_string_textedit.toPlainText()
    #     edit_mods = edit_string.split(',') if edit_string != '' else []#list(filter(None,re.split(r',+',edit_string)))        
        
    #     self.processed_spectra_data = self.raman_data.hyperspectral_image.copy()
        
    #     for edit_arg in edit_mods:
    #         if edit_arg in [self.SNIP_string,self.ALSS_string,self.iALSS_string,self.drPLS_string]:
    #             self.processed_spectra_data = self.raman_data.baseline_correction(active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.sav_gol_string:
    #             self.processed_spectra_data = self.raman_data.smoothing('sav_gol', active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.median_filter_string:
    #             self.processed_spectra_data = self.raman_data.smoothing('rolling_median', active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.pca_smoothing_string:
    #             self.processed_spectra_data = self.raman_data.smoothing('pca', active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.SNV_string:
    #             self.processed_spectra_data = self.raman_data.standard_normal_variate(active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.clip_wn_string:
    #             self.processed_spectra_data = self.raman_data.clip_wavenumbers(active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
    #         elif edit_arg == self.clip_samples_string:
    #             self.processed_spectra_data = self.raman_data.clip_samples(active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
                
    #             self.x_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'x',active_image = self.processed_spectra_data)
    #             self.y_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'y',active_image = self.processed_spectra_data)
    #             self.z_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'z',active_image = self.processed_spectra_data)
    #             self.x_values_processed = self.raman_data.get_coord_values('real',axis = 'x',active_image = self.processed_spectra_data)
    #             self.y_values_processed = self.raman_data.get_coord_values('real',axis = 'y',active_image = self.processed_spectra_data)
    #             self.z_values_processed = self.raman_data.get_coord_values('real',axis = 'z',active_image = self.processed_spectra_data)
                
    #             self.initialize_coord_spinboxes()
    #             self.change_active_coord_combos()
                
    #         elif edit_arg == self.mean_center_string:
    #             self.processed_spectra_data = self.raman_data.mean_center(active_spectra=self.processed_spectra_data,**self.edit_args_dict[edit_arg])
                
    #     self.update_spectra_plot()
    #     self.update_spectra_edited_plot()
    #     self.update_stats_edited_plot()
    
    # def update_spectra_edited_plot(self):
    #     if self.spectra_plots_active == False:
    #         return
        
    #     lower_wn = float(self.lower_wn_lineedit.text())
    #     upper_wn = float(self.upper_wn_lineedit.text())
        
    #     v_line_string = self.draw_vertical_lines_lineedit.text()
    #     v_line_coords = v_line_string.split(',') if v_line_string != '' else []
        
    #     self.plot_data_edited = self.processed_spectra_data        

    #     closest_index_to_lower_wn = np.argmin(np.abs(self.plot_data_edited.columns-lower_wn))
    #     closest_index_to_upper_wn = np.argmin(np.abs(self.plot_data_edited.columns-upper_wn))
        
    #     if self.show_spectra_combo.currentText() == 'Single spectrum':
    #         self.plot_data_edited = self.plot_data_edited.loc[[(self.get_coord('coded',axis = 'x',mode = 'processed'),self.get_coord('coded',axis = 'y',mode = 'processed'),self.get_coord('coded',axis = 'z',mode = 'processed'))]]
        
    #     min_intensity = self.plot_data_edited.iloc[:,closest_index_to_upper_wn:closest_index_to_lower_wn].values.min()
    #     max_intensity = self.plot_data_edited.iloc[:,closest_index_to_upper_wn:closest_index_to_lower_wn].values.max()
    #     curr_wavenumbers = self.plot_data_edited.columns
        
    #     self.spectra_plot_edited.axes.clear()
    #     self.spectra_plot_edited.plot(curr_wavenumbers,self.plot_data_edited.values.T,pen='b')     
    #     for xc in v_line_coords:
    #         self.spectra_plot_edited.axes.axvline(x=float(xc),color='k')
    #     self.spectra_plot_edited.axes.set_xlim(left=lower_wn,right=upper_wn)
    #     self.spectra_plot_edited.axes.set_ylim(bottom=min_intensity,top=max_intensity)
    #     self.spectra_plot_edited.draw()
        
    # def update_stats_edited_plot(self):
    #     if self.spectra_plots_active == False:
    #         return
        
    #     self.spectra_plot_baseline_stats_edited.axes.clear()
    #     if self.spectra_plot_edited_select_stats_combo.currentText() == self.descriptive_statistics_options[0]:
    #         self.spectra_plot_baseline_stats_edited.plot(self.raman_data.mean_spectrum(active_spectra=self.processed_spectra_data),self.processed_spectra_data.values.T,pen='b')
    #         self.spectra_plot_baseline_stats_edited.axes.set_xlabel('mean spectrum intensities')
    #     elif self.spectra_plot_edited_select_stats_combo.currentText() == self.descriptive_statistics_options[1]:
    #         self.spectra_plot_baseline_stats_edited.plot(self.processed_spectra_data.columns,self.raman_data.mean_spectrum(active_spectra=self.processed_spectra_data),mode='error_bar',error_data=self.raman_data.std(active_spectra=self.processed_spectra_data))
    #         self.spectra_plot_baseline_stats_edited.axes.set_xlabel('wavenumber [1/cm]')
    #     elif self.spectra_plot_edited_select_stats_combo.currentText() == self.descriptive_statistics_options[2]:
    #         self.spectra_plot_baseline_stats_edited.plot(self.processed_spectra_data.columns,self.raman_data.mean_spectrum(active_spectra=self.processed_spectra_data),mode='error_bar',error_data=self.raman_data.std(active_spectra=self.processed_spectra_data))
    #         self.spectra_plot_baseline_stats_edited.axes.set_xlabel('wavenumber [1/cm]')
    #         self.spectra_plot_baseline_stats_edited.plot(self.processed_spectra_data.columns,self.raman_data.max_spectrum(active_spectra=self.processed_spectra_data))
    #         self.spectra_plot_baseline_stats_edited.plot(self.processed_spectra_data.columns,self.raman_data.min_spectrum(active_spectra=self.processed_spectra_data))
            
    #     self.spectra_plot_baseline_stats_edited.draw()
        
    # def update_spectra_plots(self):
    #     if self.spectra_plots_active == False:
    #         return
        
    #     self.update_spectra_plot()
    #     self.update_spectra_edited_plot()

app = QApplication(sys.argv)
   
window = main_window()

window.show()
#app.exec_()
sys.exit(app.exec_())
   


