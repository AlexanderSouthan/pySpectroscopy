# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:33:44 2020

@author: aso
"""
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget,QComboBox,QCheckBox,QSpinBox,QApplication,QMainWindow,QSizePolicy,QLineEdit,QGridLayout,QHBoxLayout,QLabel,QDesktopWidget

# import preprocessing.baseline_correction
# import data_objects.hplc_data
from gui_objects.plot_canvas import plot_canvas


class hplc_visualization_window(QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.init_window()
        self.define_widgets()

        self.set_active_dataset()

        self.position_widgets()
        self.connect_event_handlers()
        self.update_elugram_plot()

    def init_window(self):
        self.setGeometry(500, 500, 1200, 900)  # xPos, yPos, width, heigth
        self.center()  # c enter function is defined below
        self.setWindowTitle('Elugram viewer')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.select_sample_label = QLabel('<b>Sample number</b>')
        self.select_sample_spinbox = QSpinBox()
        # self.select_sample_spinbox.setRange(
        #     1, len(self.parent.hplc_datasets[self.active_dataset]))

        self.sample_name_label = QLabel('<b>Sample name</b>')
        self.sample_name_lineedit = QLineEdit()
        self.sample_name_lineedit.setReadOnly(True)

        self.show_all_label = QLabel('Show all elugrams')
        self.show_all_checkbox = QCheckBox()
        self.show_all_checkbox.setChecked(False)

        self.elugram_plot = plot_canvas(
            plot_title='elugram plot', x_axis_title = 'elution time [min]')

        self.data_selection_combo = QComboBox()
        self.data_selection_combo.addItems(
            ['Raw elugram', 'Corrected elugram', 'Raw elugram + baseline',
             'Baseline'])

    def position_widgets(self):
        self.header_layout = QHBoxLayout()
        self.header_layout.addWidget(self.select_sample_label)
        self.header_layout.addWidget(self.select_sample_spinbox)
        self.header_layout.addWidget(self.sample_name_label)
        self.header_layout.addWidget(self.sample_name_lineedit)
        self.header_layout.addWidget(self.show_all_label)
        self.header_layout.addWidget(self.show_all_checkbox)
        self.header_layout.addWidget(self.data_selection_combo)

        self.grid_container.addLayout(self.header_layout, *(0, 0), 1, 1)
        self.grid_container.addWidget(self.elugram_plot, *(1, 0), 1, 1)

    def connect_event_handlers(self):
        self.select_sample_spinbox.valueChanged.connect(self.change_elugram)
        self.show_all_checkbox.stateChanged.connect(self.toggle_all_elugrams)
        self.data_selection_combo.currentIndexChanged.connect(
            self.toggle_all_elugrams)

    def toggle_all_elugrams(self):
        if self.show_all_checkbox.checkState() == 2:
            self.update_elugram_plot(mode='all')
            self.select_sample_spinbox.setEnabled(False)
        else:
            self.update_elugram_plot(mode='single')
            self.select_sample_spinbox.setEnabled(True)

    def change_elugram(self):
        self.update_elugram_plot(mode='single')

    def update_elugram_plot(self, mode='single'):
        #if self.data_selection_combo.currentText() in ['Raw elugram','Raw elugram + baseline']:
        #    plot_data = elugrams
        #elif self.data_selection_combo.currentText() == 'Corrected elugram':
        #    plot_data = corrected_elugrams
        #elif self.data_selection_combo.currentText() == 'Baseline':
        #    plot_data = baselines
        #else:
        #    plot_data = elugrams

        current_sample_number = self.select_sample_spinbox.value() - 1
        self.sample_name_lineedit.setText(
            self.parent.hplc_file_names[self.active_dataset][current_sample_number])

        self.elugram_plot.axes.clear()
        if mode == 'single':
            self.elugram_plot.plot(
                self.parent.hplc_datasets[self.active_dataset][current_sample_number].extract_elugram(220).index,
                self.parent.hplc_datasets[self.active_dataset][current_sample_number].extract_elugram(220), pen='k')
            #if self.data_selection_combo.currentText() in ['Raw elugram + baseline']:
            #    self.elugram_plot.plot(baselines[hplc_data_files[current_sample_number]].index,baselines[hplc_data_files[current_sample_number]],pen='k')
            #if self.data_selection_combo.currentText() in ['Corrected elugram']:
            #    self.elugram_plot.plot(elugrams[hplc_data_files[current_sample_number]].iloc[np.invert(clipping_masks[hplc_data_files[current_sample_number]])].index,elugrams[hplc_data_files[current_sample_number]].iloc[np.invert(clipping_masks[hplc_data_files[current_sample_number]])],pen='k')
            #    self.elugram_plot.axes.axvline(x=LOWER_CLIPPING_LIMIT,color='k')
        elif mode == 'all':
            for ii, file_name in enumerate(self.parent.hplc_file_names[self.active_dataset]):
                self.elugram_plot.plot(
                    self.parent.hplc_datasets[self.active_dataset][ii].extract_elugram(220).index,
                    self.parent.hplc_datasets[self.active_dataset][ii].extract_elugram(220),
                    pen='k')
                #if self.data_selection_combo.currentText() in ['Raw elugram + baseline']:
                #    self.elugram_plot.plot(baselines[file_name].index,baselines[file_name],pen='k')
                #if self.data_selection_combo.currentText() in ['Corrected elugram']:
                #    self.elugram_plot.plot(elugrams[file_name].iloc[np.invert(clipping_masks[file_name])].index,elugrams[file_name].iloc[np.invert(clipping_masks[file_name])],pen='k')
                #    self.elugram_plot.axes.axvline(x=LOWER_CLIPPING_LIMIT,color='k')
        #self.elugram_plot.axes.axvline(x=INTEGRATION_LIMIT_LEFT,color='k',dashes=(5,5))
        #self.elugram_plot.axes.axvline(x=INTEGRATION_LIMIT_RIGHT,color='k',dashes=(5,5))

        self.elugram_plot.draw()

    def center(self):  # centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def set_active_dataset(self):
        self.active_dataset = self.parent.dataset_selection_combo.currentText()

        self.select_sample_spinbox.setRange(
            1, len(self.parent.hplc_datasets[self.active_dataset]))
        self.sample_name_lineedit.setText(
            self.parent.hplc_file_names[self.active_dataset][0])
