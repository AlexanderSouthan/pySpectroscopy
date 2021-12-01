# -*- coding: utf-8 -*-

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

class confocal_data:
    def __init__(self, confocal_image, decimals_coordinates=1):

        self.decimals_coordinates = decimals_coordinates
        self.coord_conversion_factor = int(10**self.decimals_coordinates)

        self.spectral_data = confocal_image

    def get_coord_values(self, value_sort, axis='x', active_data=None):
        active_data = self.check_active_data(active_data)

        if axis == 'x':
            return_value = active_data.index.get_level_values(
                'x_coded').drop_duplicates().to_numpy().astype(int)
        elif axis == 'y':
            return_value = active_data.index.get_level_values(
                'y_coded').drop_duplicates().to_numpy().astype(int)
        elif axis == 'z':
            return_value = active_data.index.get_level_values(
                'z_coded').drop_duplicates().to_numpy().astype(int)

        if value_sort == 'real':
            return_value = return_value / self.coord_conversion_factor

        return return_value

    def check_active_data(self, active_data):
        if active_data is None:
            active_data = self.spectral_data_processed
        return active_data

    def generate_intensity_projections(self, dataset, col_index):
        """Generates maximum intensity projection from numpy array with 8-bit
        images.  Export is optional if export path and file name are given. """

        monochrome_image = self.univariate_data(dataset, col_index)

        self.MinIP_xyPlane_monochrome = monochrome_image.groupby(level=(0, 1)).min().unstack()
        self.MinIP_xzPlane_monochrome = monochrome_image.groupby(level=(0, 2)).min().unstack()
        self.MinIP_yzPlane_monochrome = monochrome_image.groupby(level=(1, 2)).min().unstack()
        self.MaxIP_xyPlane_monochrome = monochrome_image.groupby(level=(0, 1)).max().unstack()
        self.MaxIP_xzPlane_monochrome = monochrome_image.groupby(level=(0, 2)).max().unstack()
        self.MaxIP_yzPlane_monochrome = monochrome_image.groupby(level=(1, 2)).max().unstack()
        self.AvgIP_xyPlane_monochrome = monochrome_image.groupby(level=(0, 1)).mean().unstack()
        self.AvgIP_xzPlane_monochrome = monochrome_image.groupby(level=(0, 2)).mean().unstack()
        self.AvgIP_yzPlane_monochrome = monochrome_image.groupby(level=(1, 2)).mean().unstack()
        # self.SumIP_xyPlane_monochrome = self.monochrome_image[dataset][
        #     :,col_index].groupby(level=(0,1)).sum().unstack()
        # self.SumIP_xzPlane_monochrome = self.monochrome_image[dataset][
        #     :,col_index].groupby(level=(0,2)).sum().unstack()
        # self.SumIP_yzPlane_monochrome = self.monochrome_image[dataset][
        #     :,col_index].groupby(level=(1,2)).sum().unstack()

        self.zScanProjections = pd.concat(
            [monochrome_image.groupby(level=2).min(),
             monochrome_image.groupby(level=2).max(),
             monochrome_image.groupby(level=2).mean()],
            axis=1)
        self.zScanProjections.columns = ['zScanMin', 'zScanMax',
                                         'zScanAverage']
        self.yScanProjections = pd.concat(
            [monochrome_image.groupby(level=1).min(),
             monochrome_image.groupby(level=1).max(),
             monochrome_image.groupby(level=1).mean()],
            axis=1)
        self.yScanProjections.columns = ['yScanMin', 'yScanMax',
                                         'yScanAverage']
        self.xScanProjections = pd.concat(
            [monochrome_image.groupby(level=0).min(),
             monochrome_image.groupby(level=0).max(),
             monochrome_image.groupby(level=0).mean()],
            axis=1)
        self.xScanProjections.columns = ['xScanMin', 'xScanMax',
                                         'xScanAverage']

##############
# plot options
##############

    def generate_3D_plot(self, col_index, min_value, max_value, opacity=1.0,
                         surface_count=100):
        # This function is based on plotly which is currently not imported, so
        # this function is not functional.
        active_data = self.__decode_image_index(
                active_data=self.monochrome_image)

        coords = active_data.index.to_frame()

        fig = go.Figure()
        fig.add_trace(go.Volume(x=coords['x_values'],
                                y=coords['y_values'],
                                z=coords['z_values'],
                                value=active_data.loc[:, col_index],
                                isomin=min_value,
                                isomax=max_value,
                                opacity=opacity,
                                surface_count=surface_count))
        plot(fig)

    ###########################
#####      export methods     #################################################
    ###########################
    
    def export_intensity_projections(self, dataset, col_index, export_path):
        self.generate_intensity_projections(dataset, col_index)

        self.MinIP_xyPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MinIP_xyPlane.txt', sep='\t')
        self.MinIP_xzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MinIP_xzPlane.txt', sep='\t')
        self.MinIP_yzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MinIP_yzPlane.txt', sep='\t')
        self.MaxIP_xyPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MaxIP_xyPlane.txt', sep='\t')
        self.MaxIP_xzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MaxIP_xzPlane.txt', sep='\t')
        self.MaxIP_yzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_MaxIP_yzPlane.txt', sep='\t')
        self.AvgIP_xyPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_AvgIP_xyPlane.txt', sep='\t')
        self.AvgIP_xzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_AvgIP_xzPlane.txt', sep='\t')
        self.AvgIP_yzPlane_monochrome.to_csv(
            export_path + dataset + '_' + col_index + '_AvgIP_yzPlane.txt', sep='\t')

        self.zScanProjections.to_csv(
            export_path + dataset + '_' + col_index + '_zScanProjections.txt', sep='\t')
        self.yScanProjections.to_csv(
            export_path + dataset + '_' + col_index + '_yScanProjections.txt', sep='\t')
        self.xScanProjections.to_csv(
            export_path + dataset + '_' + col_index + '_xScanProjections.txt', sep='\t')

    def export_stack(self, dataset, col_index, export_path, axis='z'):
        monochrome_image = self.univariate_data(dataset, col_index)

        if axis == 'z':
            curr_axis_values = self.get_coord_values(
                'coded', axis='z', active_data=monochrome_image)
            curr_axis_real = self.get_coord_values(
                'real', axis='z', active_data=monochrome_image)
        elif axis == 'y':
            curr_axis_values = self.get_coord_values(
                'coded', axis='y', active_data=monochrome_image)
            curr_axis_real = self.get_coord_values(
                'real', axis='y', active_data=monochrome_image)
        elif axis == 'x':
            curr_axis_values = self.get_coord_values(
                'coded', axis='x', active_data=monochrome_image)
            curr_axis_real = self.get_coord_values(
                'real', axis='x', active_data=monochrome_image)

        for curr_coord, curr_real in zip(curr_axis_values, curr_axis_real):
            if axis == 'z':
                curr_dataset = self.xy_slice(curr_coord, dataset, col_index)
            elif axis == 'y':
                curr_dataset = self.xz_slice(curr_coord, dataset, col_index)
            elif axis == 'x':
                curr_dataset = self.yz_slice(curr_coord, dataset, col_index)

            curr_dataset.to_csv(
                export_path + dataset + '_' + col_index + '_' + axis + '_slice_' +
                str(curr_real) + '.txt', sep='\t')

    def export_monochrome_image(self, dataset, col_index, export_path,
                                export_name):
        monochrome_image = self.univariate_data(dataset, col_index)

        monochrome_image = self.__decode_image_index(monochrome_image)
        monochrome_image.to_csv(export_path + export_name + '.txt', sep='\t',
                                header=True)

    def __decode_image_index(self, active_data):
        active_data_copy = active_data.copy()
        active_data_index_frame = active_data_copy.index.to_frame()
        active_data_index_frame.columns = ['x_values', 'y_values', 'z_values']
        active_data_new_index = pd.MultiIndex.from_frame(
            active_data_index_frame/self.coord_conversion_factor)

        active_data_copy.index = active_data_new_index
        return active_data_copy

    ###########################
#####     extract methods     #################################################
    ###########################

    def univariate_data(self, dataset, col_index=None):
        try:
            col_index = float(col_index)
        except:
            col_index = col_index
        if col_index is None:
            univariate_data = self.monochrome_data[dataset]
        else:
            univariate_data = self.monochrome_data[dataset].loc[:, [col_index]]

        return univariate_data

    def xScan(self, yPos, zPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        x_scans = monochrome_image.xs((yPos, zPos), level=[1, 2])
        return x_scans

    def yScan(self, xPos, zPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        y_scans = monochrome_image.xs((xPos, zPos), level=[0, 2])
        return y_scans

    def zScan(self, xPos, yPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        z_scans = monochrome_image.xs((xPos, yPos), level=[0, 1])
        return z_scans

    def xz_slice(self, yPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        xz_slices = monochrome_image.xs(yPos, level=1)
        return xz_slices.unstack()

    def yz_slice(self, xPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        yz_slices = monochrome_image.xs(xPos, level=0)
        return yz_slices.unstack()

    def xy_slice(self, zPos, dataset, col_index):
        monochrome_image = self.univariate_data(dataset, col_index)

        xy_slices = monochrome_image.xs(zPos, level=2)
        return xy_slices.unstack()
