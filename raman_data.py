# -*- coding: utf-8 -*-
"""
For inspection of confocal LSM and Raman datasets. 
"""

import numpy as np
import pandas as pd
import glob
import imageio
import spc
from tqdm import tqdm

# import own modules#############################
from .spectroscopy_data import spectroscopy_data
from .confocal_data import confocal_data as confocal_data
#################################################

class raman_image(spectroscopy_data, confocal_data):
    def __init__(self, measurement_type=None, file_extension='txt',
                 data_source='import', directory=None, spectral_data=None,
                 decimals_coordinates=1, file_names=None, **kwargs):
        self.data_source = data_source
        self.kwargs = kwargs

        self.directory = directory
        self.measurement_type = measurement_type
        self.file_extension = file_extension
        self.decimals_coordinates = decimals_coordinates
        self.coord_conversion_factor = int(10**self.decimals_coordinates)
        self.file_names = file_names

        if data_source == 'import':
            # imports images into self.spectral_data
            self.__import_data()
        elif data_source == 'DataFrame':
            self.spectral_data = spectral_data
            index_frame = self.spectral_data.index.to_frame()
            index_frame.columns = ['x_coded', 'y_coded', 'z_coded']
            new_index = pd.MultiIndex.from_frame(
                (index_frame*self.coord_conversion_factor).astype(np.int64))
            self.spectral_data.index = new_index

        self.reset_processed_data()
        self.wavenumbers = self.spectral_data.columns.to_numpy()
        self.baseline_data = {}
        self.monochrome_data = {}

    ###############################
#####        basic methods        #############################################
    ###############################

    def __import_data(self):
        if self.file_names is None:
            self.file_names = glob.glob(
                self.directory + '*.' + self.file_extension)

        self.file_list = pd.DataFrame(self.file_names, columns=['file_name'],
                                      index=np.arange(len(self.file_names)))
        self.file_list['x_coded'] = np.zeros(len(self.file_names), dtype=int)
        self.file_list['y_coded'] = np.zeros(len(self.file_names), dtype=int)
        self.file_list['z_coded'] = np.zeros(len(self.file_names), dtype=int)

        if self.measurement_type in ['Raman_volume', 'Raman_x_scan',
                                     'Raman_y_scan', 'Raman_z_scan',
                                     'Raman_single_spectrum']:
            if self.measurement_type in ['Raman_volume', 'Raman_x_scan']:
                self.file_list.iloc[:, 1] = (
                    pd.to_numeric(
                        self.file_list.iloc[:, 0].str.extract(
                            r'__X_([-*\d*.*\d*]*)\__Y_', expand=False)) *
                    self.coord_conversion_factor).astype(int)
            if self.measurement_type in ['Raman_volume', 'Raman_y_scan']:
                self.file_list.iloc[:, 2] = (
                    pd.to_numeric(self.file_list.iloc[:, 0].str.extract(
                        r'__Y_([-*\d*.*\d*]*)\__Z_', expand=False)) *
                    self.coord_conversion_factor).astype(int)
            if self.measurement_type in ['Raman_volume', 'Raman_z_scan']:
                self.file_list.iloc[:, 3] = (
                    pd.to_numeric(self.file_list.iloc[:, 0].str.extract(
                        r'__Z_([-*\d*.*\d*]*)\__', expand=False)) *
                    self.coord_conversion_factor).astype(int)

            self.file_list = self.file_list.sort_values(
                by=['z_coded', 'y_coded', 'x_coded'])
            self.file_list.index = pd.RangeIndex(len(self.file_list.index))

            wavenumbers = np.fromfile(
                self.file_list['file_name'][0], sep=' ')[::2]
            intensities = np.zeros((len(self.file_list.index),
                                    wavenumbers.size))

            for index, curr_index in enumerate(tqdm(self.file_list.index)):
                intensities[index] = np.fromfile(
                    self.file_list.iloc[index, 0], sep=' ')[1::2]

        # Inline_IR and LSM still have to get their own classes
        elif self.measurement_type == 'Inline_IR':
            spectrum_data = spc.File(self.file_list.iloc[0, 0])
            number_of_spectra = len(spectrum_data.sub)
            wavenumbers = spectrum_data.x
            intensities = np.zeros((number_of_spectra, len(spectrum_data.x)))
            time_data = np.zeros(number_of_spectra)

            for index, curr_spec in enumerate(tqdm(spectrum_data.sub)):
                intensities[index, :] = curr_spec.y
                time_data[index] = curr_spec.subtime

            self.file_list = self.file_list.loc[
                self.file_list.index.repeat(number_of_spectra)].reset_index(
                    drop=True)
            self.file_list.iloc[:, 1] = (pd.Series(time_data) *
                                         self.coord_conversion_factor).astype(
                                             int)

        # Is still experimental, especially correct coordinates are missing and
        # possibly not working for not square images
        elif self.measurement_type == 'LSM':
            # read first image to get image dimensions
            first_image = imageio.imread(self.file_list.iloc[0, 0])
            pixels_per_image = np.shape(first_image)[0]*np.shape(
                first_image)[1]
            number_of_images = len(self.file_list.index)

            intensities = np.zeros((number_of_images*pixels_per_image,
                                    np.shape(first_image)[2]), dtype='uint8')
            z_coords = np.repeat(np.arange(number_of_images), pixels_per_image)
            x_coords = np.tile(np.repeat(np.arange(np.shape(first_image)[0]),
                                         np.shape(first_image)[1]),
                               number_of_images)
            y_coords = np.tile(np.tile(np.arange(np.shape(first_image)[1]),
                                       np.shape(first_image)[0]),
                               number_of_images)

            wavenumbers = np.arange(np.shape(first_image)[2])

            for index, curr_file in enumerate(tqdm(self.file_list.iloc[:, 0])):
                intensities[index*pixels_per_image:(index+1)*pixels_per_image, :] = np.reshape(imageio.imread(curr_file), (-1, 3))

            self.file_list = pd.DataFrame(np.stack(
                [np.repeat(self.file_list.iloc[:, 0], pixels_per_image),
                 x_coords, y_coords, z_coords]).T,
                columns=self.file_list.columns)

        hyperspectral_image_index = pd.MultiIndex.from_frame(
            self.file_list.iloc[:, 1:4])
        self.spectral_data = pd.DataFrame(
            intensities, index=hyperspectral_image_index,
            columns=np.around(wavenumbers, 2))
