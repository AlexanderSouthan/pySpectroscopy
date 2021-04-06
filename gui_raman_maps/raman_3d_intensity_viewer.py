# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:44:12 2019

@author: AlMaMi
"""

from gui_objects.plot_canvas import plot_canvas
from gui_objects.image_canvas import image_canvas
from PyQt5.QtWidgets import (QMainWindow,QComboBox,QWidget,QGridLayout,
                             QDesktopWidget)

class raman_3d_intensity_viewer(QMainWindow):
    def __init__(self, raman_data):
        super().__init__()
        self.init_window()
        self.define_widgets()
        self.position_widgets()

        self.plots_active = True
        self.replace_data(raman_data)
        self.connect_event_handlers()

    def init_window(self):
        self.setGeometry(500, 500, 1200, 900)  # xPos,yPos,width, heigth
        self.center()
        self.setWindowTitle('Intensitiy projections')

        self.container0 = QWidget(self)
        self.setCentralWidget(self.container0)

        self.grid_container = QGridLayout()
        self.container0.setLayout(self.grid_container)

    def define_widgets(self):
        self.select_data_combo = QComboBox()
        self.select_column_combo = QComboBox()

        self.zScan_projections = plot_canvas(plot_title='zScan projections',x_axis_title = 'z coordinate')
        self.MaxIP_xz = image_canvas(self,plot_title='MaxIP xz plane')
        self.MinIP_xz = image_canvas(self,plot_title='MinIP xz plane')
        self.AvgIP_xz = image_canvas(self,plot_title='AvgIP xz plane')
        self.intensities_xz = image_canvas(self,plot_title='use scroll wheel to navigate xz slices',scroll_type='y')

        self.yScan_projections = plot_canvas(plot_title='yScan projections',x_axis_title = 'y coordinate')
        self.MaxIP_yz = image_canvas(self,plot_title='MaxIP yz plane')
        self.MinIP_yz = image_canvas(self,plot_title='MinIP yz plane')
        self.AvgIP_yz = image_canvas(self,plot_title='AvgIP yz plane')
        self.intensities_yz = image_canvas(self,plot_title='use scroll wheel to navigate yz slices\ncurrent slice:',scroll_type='x')

        self.xScan_projections = plot_canvas(plot_title='xScan projections',x_axis_title = 'x coordinate')
        self.MaxIP_xy = image_canvas(self,plot_title='MaxIP xy plane')
        self.MinIP_xy = image_canvas(self,plot_title='MinIP xy plane')
        self.AvgIP_xy = image_canvas(self,plot_title='AvgIP xy plane')
        self.intensities_xy = image_canvas(self,plot_title='use scroll wheel to navigate xy slices',scroll_type='z')

    def position_widgets(self):
        self.grid_container.addWidget(self.select_data_combo, *(0,0),1,1)
        self.grid_container.addWidget(self.select_column_combo, *(0,1),1,1)

        self.grid_container.addWidget(self.zScan_projections, *(1,0),1,1)
        self.grid_container.addWidget(self.MaxIP_xz, *(1,1),1,1)
        self.grid_container.addWidget(self.MinIP_xz, *(1,2),1,1)
        self.grid_container.addWidget(self.AvgIP_xz, *(1,3),1,1)   
        self.grid_container.addWidget(self.intensities_xz, *(1,4),1,1)

        self.grid_container.addWidget(self.yScan_projections, *(2,0),1,1)
        self.grid_container.addWidget(self.MaxIP_yz, *(2,1),1,1)
        self.grid_container.addWidget(self.MinIP_yz, *(2,2),1,1)
        self.grid_container.addWidget(self.AvgIP_yz, *(2,3),1,1)
        self.grid_container.addWidget(self.intensities_yz, *(2,4),1,1)

        self.grid_container.addWidget(self.xScan_projections, *(3,0),1,1)
        self.grid_container.addWidget(self.MaxIP_xy, *(3,1),1,1)
        self.grid_container.addWidget(self.MinIP_xy, *(3,2),1,1)
        self.grid_container.addWidget(self.AvgIP_xy, *(3,3),1,1)  
        self.grid_container.addWidget(self.intensities_xy, *(3,4),1,1)

    def connect_event_handlers(self):
        self.intensities_xz.mpl_connect('scroll_event', self.intensities_xz.onscroll)
        self.intensities_yz.mpl_connect('scroll_event', self.intensities_yz.onscroll)
        self.intensities_xy.mpl_connect('scroll_event', self.intensities_xy.onscroll)
        self.select_data_combo.currentIndexChanged.connect(self.update_column_combo)
        #also defined in self.init_combos

    def replace_data(self, raman_data):
        # Disables the update_spectra_plots function.
        self.plots_active = False

        self.raman_data = raman_data
        self.x_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'x')
        self.y_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'y')
        self.z_values_coded_processed = self.raman_data.get_coord_values('coded',axis = 'z')
        self.x_values_processed = self.raman_data.get_coord_values('real',axis = 'x')
        self.y_values_processed = self.raman_data.get_coord_values('real',axis = 'y')
        self.z_values_processed = self.raman_data.get_coord_values('real',axis = 'z')

        self.init_combos()

        # Now the update_spectra_plots function will do something.
        self.plots_active = True

        self.plot_images()

    def init_combos(self):
        self.select_data_combo.clear()
        self.select_data_combo.addItems(self.raman_data.monochrome_data.keys())
        self.select_data_combo.setCurrentIndex(0)

        self.update_column_combo()

        self.select_column_combo.currentIndexChanged.connect(self.plot_images)

    def update_column_combo(self):
        curr_data = self.select_data_combo.currentText()
        self.select_column_combo.clear()
        self.select_column_combo.addItems(
            self.raman_data.monochrome_data[curr_data].columns.astype(str).to_list())
        self.select_column_combo.setCurrentIndex(0)

    def plot_images(self):
        if self.plots_active is False:
            return

        selected_data = self.select_data_combo.currentText()
        selected_column = self.select_column_combo.currentText()

        self.raman_data.generate_intensity_projections(
            selected_data, selected_column)

        self.zScan_projections.axes.clear()
        self.yScan_projections.axes.clear()
        self.xScan_projections.axes.clear()

        self.zScan_projections.plot(self.z_values_processed,self.raman_data.zScanProjections,pen='b')
        self.yScan_projections.plot(self.y_values_processed,self.raman_data.yScanProjections,pen='r')
        self.xScan_projections.plot(self.x_values_processed,self.raman_data.xScanProjections,pen='g')

        self.MaxIP_xz.set_image_data(self.raman_data.MaxIP_xzPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.z_values_processed)
        self.MinIP_xz.set_image_data(self.raman_data.MinIP_xzPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.z_values_processed)
        self.AvgIP_xz.set_image_data(self.raman_data.AvgIP_xzPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.z_values_processed)
        self.intensities_xz.set_image_data(self.raman_data.xz_slice(self.y_values_coded_processed[0], selected_data, selected_column),vertical_axis = self.x_values_processed,horizontal_axis = self.z_values_processed)

        self.MaxIP_yz.set_image_data(self.raman_data.MaxIP_yzPlane_monochrome,vertical_axis = self.y_values_processed,horizontal_axis = self.z_values_processed)
        self.MinIP_yz.set_image_data(self.raman_data.MinIP_yzPlane_monochrome,vertical_axis = self.y_values_processed,horizontal_axis = self.z_values_processed)
        self.AvgIP_yz.set_image_data(self.raman_data.AvgIP_yzPlane_monochrome,vertical_axis = self.y_values_processed,horizontal_axis = self.z_values_processed)
        self.intensities_yz.set_image_data(self.raman_data.yz_slice(self.x_values_coded_processed[0], selected_data, selected_column),vertical_axis = self.y_values_processed,horizontal_axis = self.z_values_processed)

        self.MaxIP_xy.set_image_data(self.raman_data.MaxIP_xyPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.y_values_processed)
        self.MinIP_xy.set_image_data(self.raman_data.MinIP_xyPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.y_values_processed)
        self.AvgIP_xy.set_image_data(self.raman_data.AvgIP_xyPlane_monochrome,vertical_axis = self.x_values_processed,horizontal_axis = self.y_values_processed)
        self.intensities_xy.set_image_data(self.raman_data.xy_slice(self.z_values_coded_processed[0], selected_data, selected_column),vertical_axis = self.x_values_processed,horizontal_axis = self.y_values_processed)

    def center(self):#centers object on screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
