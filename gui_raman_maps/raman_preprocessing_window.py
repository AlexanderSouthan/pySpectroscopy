# -*- coding: utf-8 -*-

# import numpy as np
import copy
from ast import literal_eval
from PyQt5.QtWidgets import (QMainWindow, QComboBox, QWidget, QGridLayout,
                             QDesktopWidget, QLabel, QVBoxLayout, QLineEdit,
                             QPushButton, QHBoxLayout, QTextEdit, QFileDialog)


class raman_preprocessing_window(QMainWindow):
    def __init__(self, raman_data):
        self.update_data(raman_data)

        super().__init__()
        self.init_window()
        self.define_widgets()
        self.init_baseline_parameters()
        self.set_option_values()
        self.position_widgets()
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500,500,600,900) #xPos,yPos,width, heigth
        self.center() #center function is defined below
        self.setWindowTitle('Spectrum preprocessing')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.processing_label = QLabel('<b>Spectrum processing string</b>')
        self.edit_string_textedit = QTextEdit()
        self.button_process_spectra = QPushButton('Preprocess spectra')

        self.preprocessing_label = QLabel('<b><h1>Baseline options</h1></b>')

        self.ALSS_label = QLabel('<b>ALSS baseline options</b>') 
        self.ALSS_lam_label = QLabel('lam',self.container0)
        self.ALSS_lam_lineedit = QLineEdit(self.container0)
        self.ALSS_p_label = QLabel('p',self.container0)
        self.ALSS_p_lineedit = QLineEdit(self.container0)
        self.ALSS_niter_label = QLabel('n_iter', self.container0)
        self.ALSS_niter_lineedit = QLineEdit(self.container0)

        self.iALSS_label = QLabel('<b>iALSS baseline options</b>') 
        self.iALSS_lam_label = QLabel('lam',self.container0)
        self.iALSS_lam_lineedit = QLineEdit(self.container0)
        self.iALSS_lam_1_label = QLabel('lam_1',self.container0)
        self.iALSS_lam_1_lineedit = QLineEdit(self.container0)
        self.iALSS_p_label = QLabel('p',self.container0)
        self.iALSS_p_lineedit = QLineEdit(self.container0)
        self.iALSS_niter_label = QLabel('n_iter', self.container0)
        self.iALSS_niter_lineedit = QLineEdit(self.container0)
        
        self.drPLS_label = QLabel('<b>drPLS baseline options</b>') 
        self.drPLS_lam_label = QLabel('lam',self.container0)
        self.drPLS_lam_lineedit = QLineEdit(self.container0)
        self.drPLS_eta_label = QLabel('eta',self.container0)
        self.drPLS_eta_lineedit = QLineEdit(self.container0)
        self.drPLS_niter_label = QLabel('n_iter', self.container0)
        self.drPLS_niter_lineedit = QLineEdit(self.container0)

        self.SNIP_label = QLabel('<b>SNIP baseline options</b>') 
        self.SNIP_niter_label = QLabel('n_iter',self.container0)
        self.SNIP_niter_lineedit = QLineEdit(self.container0)

        self.ModPoly_label = QLabel('<b>ModPoly and IModPoly baseline options</b>')
        self.ModPoly_niter_label = QLabel('n_iter',self.container0)
        self.ModPoly_niter_lineedit = QLineEdit(self.container0)
        self.ModPoly_polyorder_label = QLabel('poly_order', self.container0)
        self.ModPoly_polyorder_lineedit = QLineEdit(self.container0)

        self.PPF_label = QLabel('<b>Piecewise polynomial baseline options</b>')
        self.PPF_niter_label = QLabel('n_iter', self.container0)
        self.PPF_niter_lineedit = QLineEdit(self.container0)
        self.PPF_polyorders_label = QLabel('poly_orders', self.container0)
        self.PPF_polyorders_lineedit = QLineEdit(self.container0)
        self.PPF_segment_borders_label = QLabel('segment_borders', self.container0)
        self.PPF_segment_borders_lineedit = QLineEdit(self.container0)
        self.PPF_fit_method_label = QLabel('fit method', self.container0)
        self.PPF_fit_method_lineedit = QLineEdit(self.container0)

        self.sav_gol_label = QLabel('<b>Savitzky-Golay options</b>') 
        self.derivative_order_label = QLabel('Differentiation order', self.container0)
        self.derivative_order_combo = QComboBox(self.container0)
        self.derivative_order_combo.addItems(['0','1','2'])
        self.savgol_points_label = QLabel('Data points for Savitzky-Golay (one side)', self.container0)
        self.savgol_points_combo = QComboBox(self.container0)
        self.savgol_points_combo.addItems(['0','1','2','3','4','5','6','7','8','9'])

        self.median_filter_label = QLabel('<b>Median filter options</b>') 
        self.median_filter_points_label = QLabel('Window size',self.container0)
        self.median_filter_points_combo = QComboBox(self.container0)
        self.median_filter_points_combo.addItems(['3','5','7','9'])

        self.pca_smoothing_label = QLabel('<b>PCA smoothing options</b>') 
        self.pca_smoothing_components_label = QLabel('Number of PCs',self.container0)
        self.pca_smoothing_components_combo = QComboBox(self.container0)
        self.pca_smoothing_components_combo.addItems(['1','2','3','4','5','6','7','8','9'])

        self.clip_wn_label = QLabel('<b>Clip wavenumbers options</b>') 
        self.clip_wn_lower_limit_label = QLabel('Wavenumber limit string',self.container0)
        self.clip_wn_lower_limit_lineedit = QLineEdit(self.container0)

        self.reset_button = QPushButton('Reset defaults')

    def position_widgets(self):
        self.processing_layout = QHBoxLayout()
        self.processing_layout.addWidget(self.edit_string_textedit)
        self.processing_layout.addWidget(self.button_process_spectra)

        self.ALSS_selection_layout = QHBoxLayout()
        self.ALSS_selection_layout.addWidget(self.ALSS_lam_label)
        self.ALSS_selection_layout.addWidget(self.ALSS_lam_lineedit)
        self.ALSS_selection_layout.addWidget(self.ALSS_p_label)
        self.ALSS_selection_layout.addWidget(self.ALSS_p_lineedit)
        self.ALSS_selection_layout.addWidget(self.ALSS_niter_label)
        self.ALSS_selection_layout.addWidget(self.ALSS_niter_lineedit)

        self.iALSS_selection_layout = QHBoxLayout()
        self.iALSS_selection_layout.addWidget(self.iALSS_lam_label)
        self.iALSS_selection_layout.addWidget(self.iALSS_lam_lineedit)
        self.iALSS_selection_layout.addWidget(self.iALSS_lam_1_label)
        self.iALSS_selection_layout.addWidget(self.iALSS_lam_1_lineedit)
        self.iALSS_selection_layout.addWidget(self.iALSS_p_label)
        self.iALSS_selection_layout.addWidget(self.iALSS_p_lineedit)
        self.iALSS_selection_layout.addWidget(self.iALSS_niter_label)
        self.iALSS_selection_layout.addWidget(self.iALSS_niter_lineedit)

        self.drPLS_selection_layout = QHBoxLayout()
        self.drPLS_selection_layout.addWidget(self.drPLS_lam_label)
        self.drPLS_selection_layout.addWidget(self.drPLS_lam_lineedit)
        self.drPLS_selection_layout.addWidget(self.drPLS_eta_label)
        self.drPLS_selection_layout.addWidget(self.drPLS_eta_lineedit)
        self.drPLS_selection_layout.addWidget(self.drPLS_niter_label)
        self.drPLS_selection_layout.addWidget(self.drPLS_niter_lineedit)

        self.SNIP_selection_layout = QHBoxLayout()
        self.SNIP_selection_layout.addWidget(self.SNIP_niter_label)
        self.SNIP_selection_layout.addWidget(self.SNIP_niter_lineedit)

        self.ModPoly_selection_layout = QHBoxLayout()
        self.ModPoly_selection_layout.addWidget(self.ModPoly_niter_label)
        self.ModPoly_selection_layout.addWidget(self.ModPoly_niter_lineedit)
        self.ModPoly_selection_layout.addWidget(self.ModPoly_polyorder_label)
        self.ModPoly_selection_layout.addWidget(self.ModPoly_polyorder_lineedit)

        self.PPF_selection_layout = QHBoxLayout()
        self.PPF_selection_layout.addWidget(self.PPF_niter_label)
        self.PPF_selection_layout.addWidget(self.PPF_niter_lineedit)
        self.PPF_selection_layout.addWidget(self.PPF_polyorders_label)
        self.PPF_selection_layout.addWidget(self.PPF_polyorders_lineedit)
        self.PPF_selection_layout.addWidget(self.PPF_segment_borders_label)
        self.PPF_selection_layout.addWidget(self.PPF_segment_borders_lineedit)
        self.PPF_selection_layout.addWidget(self.PPF_fit_method_label)
        self.PPF_selection_layout.addWidget(self.PPF_fit_method_lineedit)

        self.sav_gol_selection_layout = QHBoxLayout()
        self.sav_gol_selection_layout.addWidget(self.savgol_points_label)
        self.sav_gol_selection_layout.addWidget(self.savgol_points_combo)
        self.sav_gol_selection_layout.addWidget(self.derivative_order_label)
        self.sav_gol_selection_layout.addWidget(self.derivative_order_combo)

        self.median_filter_layout = QHBoxLayout()
        self.median_filter_layout.addWidget(self.median_filter_points_label)
        self.median_filter_layout.addWidget(self.median_filter_points_combo)
        
        self.pca_smoothing_layout = QHBoxLayout()
        self.pca_smoothing_layout.addWidget(self.pca_smoothing_components_label)
        self.pca_smoothing_layout.addWidget(self.pca_smoothing_components_combo)
        
        self.clip_wn_lower_limit_selection_layout = QHBoxLayout()
        self.clip_wn_lower_limit_selection_layout.addWidget(self.clip_wn_lower_limit_label)
        self.clip_wn_lower_limit_selection_layout.addWidget(self.clip_wn_lower_limit_lineedit)

        self.preprocessing_layout = QVBoxLayout()
        self.preprocessing_layout.addWidget(self.processing_label)
        self.preprocessing_layout.addLayout(self.processing_layout)
        self.preprocessing_layout.addWidget(self.preprocessing_label)
        self.preprocessing_layout.addWidget(self.ALSS_label)
        self.preprocessing_layout.addLayout(self.ALSS_selection_layout)
        self.preprocessing_layout.addWidget(self.iALSS_label)
        self.preprocessing_layout.addLayout(self.iALSS_selection_layout)
        self.preprocessing_layout.addWidget(self.drPLS_label)
        self.preprocessing_layout.addLayout(self.drPLS_selection_layout)
        self.preprocessing_layout.addWidget(self.SNIP_label)
        self.preprocessing_layout.addLayout(self.SNIP_selection_layout)
        self.preprocessing_layout.addWidget(self.ModPoly_label)
        self.preprocessing_layout.addLayout(self.ModPoly_selection_layout)
        self.preprocessing_layout.addWidget(self.PPF_label)
        self.preprocessing_layout.addLayout(self.PPF_selection_layout)
        self.preprocessing_layout.addWidget(self.preprocessing_label)
        self.preprocessing_layout.addWidget(self.sav_gol_label)
        self.preprocessing_layout.addLayout(self.sav_gol_selection_layout)
        self.preprocessing_layout.addWidget(self.median_filter_label)
        self.preprocessing_layout.addLayout(self.median_filter_layout)
        self.preprocessing_layout.addWidget(self.pca_smoothing_label)
        self.preprocessing_layout.addLayout(self.pca_smoothing_layout)
        self.preprocessing_layout.addWidget(self.clip_wn_label)
        self.preprocessing_layout.addLayout(self.clip_wn_lower_limit_selection_layout)
        self.preprocessing_layout.addStretch(1)
        self.preprocessing_layout.addWidget(self.reset_button)

        self.grid_container.addLayout(self.preprocessing_layout, *(0,0),1,1)

    def connect_event_handlers(self):
        self.button_process_spectra.clicked.connect(self.process_spectra)

        self.ALSS_lam_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.ALSS_p_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.ALSS_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.iALSS_lam_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.iALSS_lam_1_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.iALSS_p_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.iALSS_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.drPLS_lam_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.drPLS_eta_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.drPLS_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.SNIP_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.ModPoly_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.ModPoly_polyorder_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.PPF_niter_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.PPF_polyorders_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.PPF_segment_borders_lineedit.editingFinished.connect(
            self.update_processing_parameters)
        self.PPF_fit_method_lineedit.editingFinished.connect(
            self.update_processing_parameters)

        self.savgol_points_combo.currentIndexChanged.connect(
            self.update_processing_parameters)
        self.median_filter_points_combo.currentIndexChanged.connect(
            self.update_processing_parameters)
        self.derivative_order_combo.currentIndexChanged.connect(
            self.update_processing_parameters)
        self.pca_smoothing_components_combo.currentIndexChanged.connect(
            self.update_processing_parameters)
        self.clip_wn_lower_limit_lineedit.editingFinished.connect(
            self.update_processing_parameters)

        self.reset_button.clicked.connect(self.reset_defaults)

    def update_data(self, raman_data):
        self.raman_data = raman_data

    def set_option_values(self, mode='initial'):
        if mode == 'default':
            current_options_dict = self.edit_args_dict_default
        else:
            current_options_dict = self.edit_args_dict

        self.ALSS_lam_lineedit.setText(
            str(current_options_dict['ALSS']['lam']))
        self.ALSS_p_lineedit.setText(str(current_options_dict['ALSS']['p']))
        self.ALSS_niter_lineedit.setText(
            str(current_options_dict['ALSS']['n_iter']))

        self.iALSS_lam_lineedit.setText(
            str(current_options_dict['iALSS']['lam']))
        self.iALSS_lam_1_lineedit.setText(
            str(current_options_dict['iALSS']['lam_1']))
        self.iALSS_p_lineedit.setText(str(current_options_dict['iALSS']['p']))
        self.iALSS_niter_lineedit.setText(
            str(current_options_dict['iALSS']['n_iter']))

        self.drPLS_lam_lineedit.setText(
            str(current_options_dict['drPLS']['lam']))
        self.drPLS_eta_lineedit.setText(
            str(current_options_dict['drPLS']['eta']))
        self.drPLS_niter_lineedit.setText(
            str(current_options_dict['drPLS']['n_iter']))

        self.SNIP_niter_lineedit.setText(
            str(current_options_dict['SNIP']['n_iter']))

        self.ModPoly_niter_lineedit.setText(
            str(current_options_dict['ModPoly']['n_iter']))
        self.ModPoly_polyorder_lineedit.setText(
            str(current_options_dict['IModPoly']['poly_order']))

        self.PPF_niter_lineedit.setText(
            str(current_options_dict['PPF']['n_iter']))
        self.PPF_polyorders_lineedit.setText(
            str(current_options_dict['PPF']['poly_orders']))
        self.PPF_segment_borders_lineedit.setText(
            str(current_options_dict['PPF']['segment_borders']))
        self.PPF_fit_method_lineedit.setText(
            str(current_options_dict['PPF']['fit_method']))

        self.derivative_order_combo.setCurrentIndex(
            self.derivative_order_combo.findText(
                str(current_options_dict['sav_gol']['deriv'])))
        self.savgol_points_combo.setCurrentIndex(
            self.savgol_points_combo.findText(
                str(current_options_dict['sav_gol']['savgol_points'])))

        self.median_filter_points_combo.setCurrentIndex(
            self.median_filter_points_combo.findText(
                str(current_options_dict['median_filter']['window'])))

        self.pca_smoothing_components_combo.setCurrentIndex(
            self.pca_smoothing_components_combo.findText(
                str(current_options_dict['pca_smoothing']['pca_components'])))

        self.clip_wn_lower_limit_lineedit.setText(
            str(current_options_dict['clip_wn']['wn_limits']))

    # def save_options_to_file(self):
    #     save_file_name,__ = QFileDialog.getSaveFileName(self,'Save option settings to file','','*.npy')
    #     np.save(save_file_name,self.edit_args_dict)
    
    # def read_options_from_file(self):
    #     open_file_name,__ = QFileDialog.getOpenFileName(self,'Read option settings from file','','*.npy')
    #     self.edit_args_dict = np.load(open_file_name).item()
    #     self.set_option_values()

    def reset_defaults(self):
        self.set_option_values(mode='default')

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_baseline_parameters(self):
        self.edit_args_dict = {
            'SNIP': {
                'n_iter': 100},
            'ALSS': {
                'lam': 10000, 'p': 0.001, 'n_iter': 10},
            'iALSS': {
                'lam': 2000, 'lam_1': 0.01, 'p': 0.01, 'n_iter': 10},
            'drPLS': {
                'lam': 1000000, 'eta': 0.5, 'n_iter': 100},
            'ModPoly': {
                'n_iter': 100, 'poly_order': 5},
            'IModPoly': {
                'n_iter': 100, 'poly_order': 5},
            'PPF': {
                'n_iter': 100, 'poly_orders': [3, 3],
                'segment_borders': [1000], 'fit_method': 'ModPoly'},
            'convex_hull': {},
            'sav_gol': {
                'deriv': 0, 'savgol_points': 9},
            'median_filter': {
                'window': 5},
            'pca_smoothing': {
                'pca_components': 3},
            'SNV': {},
            'clip_wn': {
                'wn_limits': [[1100, 1500]]},
            'mean_center': {},
            'total_intensity': {}}

        self.edit_args_dict_default = copy.deepcopy(self.edit_args_dict)

    def update_processing_parameters(self):
        self.edit_args_dict = {
            'SNIP': {
                'n_iter': int(self.SNIP_niter_lineedit.text())},
            'ALSS': {
                'lam': float(self.ALSS_lam_lineedit.text()),
                'p': float(self.ALSS_p_lineedit.text()),
                'n_iter': int(self.ALSS_niter_lineedit.text())},
            'iALSS': {
                'lam': float(self.iALSS_lam_lineedit.text()),
                'lam_1': float(self.iALSS_lam_1_lineedit.text()),
                'p': float(self.iALSS_p_lineedit.text()),
                'n_iter': int(self.iALSS_niter_lineedit.text())},
            'drPLS': {
                'lam': float(self.drPLS_lam_lineedit.text()),
                'eta': float(self.drPLS_eta_lineedit.text()),
                'n_iter': int(self.drPLS_niter_lineedit.text())},
            'ModPoly': {
                'n_iter': float(self.ModPoly_niter_lineedit.text()),
                'poly_order': int(self.ModPoly_polyorder_lineedit.text())},
            'IModPoly': {
                'n_iter': float(self.ModPoly_niter_lineedit.text()),
                'poly_order': int(self.ModPoly_polyorder_lineedit.text())},
            'PPF': {
                'n_iter': int(self.PPF_niter_lineedit.text()),
                'poly_orders': literal_eval(self.PPF_polyorders_lineedit.text()),
                'segment_borders': literal_eval(self.PPF_segment_borders_lineedit.text()),
                'fit_method': self.PPF_fit_method_lineedit.text()},
            'convex_hull': {},
            'sav_gol':{
                'deriv':int(self.derivative_order_combo.currentText()),
                'savgol_points':int(self.savgol_points_combo.currentText())},
            'median_filter':{
                'window':int(self.median_filter_points_combo.currentText())},
            'pca_smoothing':{
                'pca_components':int(
                    self.pca_smoothing_components_combo.currentText())},
            'SNV':{},
            'clip_wn':{
                'wn_limits':literal_eval(
                    self.clip_wn_lower_limit_lineedit.text())},
            'mean_center':{},
            'total_intensity':{}}

    def process_spectra(self):
        edit_string = self.edit_string_textedit.toPlainText()
        edit_mods = edit_string.split(',') if edit_string != '' else []

        self.raman_data.reset_processed_data()

        for edit_arg in edit_mods:
            if edit_arg in ['SNIP', 'ALSS', 'iALSS', 'drPLS', 'ModPoly',
                            'IModPoly', 'PPF', 'convex_hull']:
                self.raman_data.baseline_correction(
                    mode=edit_arg, **self.edit_args_dict[edit_arg])
            elif edit_arg == 'sav_gol':
                self.raman_data.smoothing(
                    'sav_gol', **self.edit_args_dict[edit_arg])
            elif edit_arg == 'median_filter':
                self.raman_data.smoothing(
                    'rolling_median', **self.edit_args_dict[edit_arg])
            elif edit_arg == 'pca_smoothing':
                self.raman_data.smoothing(
                    'pca', **self.edit_args_dict[edit_arg])
            elif edit_arg == 'SNV':
                self.raman_data.standard_normal_variate(
                    **self.edit_args_dict[edit_arg])
            elif edit_arg == 'clip_wn':
                self.raman_data.clip_wavenumbers(
                    **self.edit_args_dict[edit_arg])
            elif edit_arg == 'mean_center':
                self.raman_data.mean_center(**self.edit_args_dict[edit_arg])
            elif edit_arg == 'total_intensity':
                self.raman_data.normalize('total_intensity',
                                          **self.edit_args_dict[edit_arg])
