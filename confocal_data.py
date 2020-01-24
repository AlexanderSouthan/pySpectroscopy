# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:24:34 2020

@author: AlMaMi
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz,trapz

class confocal_data:
    def __init__(self,confocal_image,decimals_coordinates = 1):
        
        self.decimals_coordinates = decimals_coordinates
        self.coord_conversion_factor = int(10**self.decimals_coordinates)
        
        self.hyperspectral_image = confocal_image
    
    def get_coord_values(self,value_sort,axis='x',active_image = None):
        active_image = self.check_active_image(active_image)

        if axis == 'x':
            return_value = active_image.index.get_level_values('x_coded').drop_duplicates().to_numpy().astype(int)
        elif axis == 'y':
            return_value = active_image.index.get_level_values('y_coded').drop_duplicates().to_numpy().astype(int)
        elif axis == 'z':
            return_value = active_image.index.get_level_values('z_coded').drop_duplicates().to_numpy().astype(int)

        if value_sort == 'real':
            return_value = return_value / self.coord_conversion_factor

        return return_value

    def check_active_image(self,active_image):
        if active_image is None:
            active_image = self.hyperspectral_image
        return active_image

    ###########################
#####     imaging methods     #################################################
    ###########################
    
    def multi2monochrome(self,mode='int_at_point',wavenumber1=521,wavenumber2=None,active_image=None,
                         pca_components = 7,
                         fit_mode = 'Levenberg-Marquardt',ref_spec = None,max_iter = 1000,initial_guess = None,
                         lower_bounds = None,upper_bounds = None): #convert multichrome images to monochrome

        active_image = self.check_active_image(active_image)

        if mode == 'int_at_point':
            self.monochrome_image = active_image.loc[:,[active_image.columns[np.argmin(np.abs(active_image.columns - wavenumber1))]]]
            self.monochrome_image.columns = [mode + '_' + str(wavenumber1)]
        elif mode in ['sig_to_base','sig_to_axis']:
            self.hyperspectral_image_integrated = self.integrate_image(active_spectra=active_image)
            closest_index_lower = np.argmin(np.abs(self.hyperspectral_image_integrated.columns-wavenumber1))
            closest_index_upper = np.argmin(np.abs(self.hyperspectral_image_integrated.columns-wavenumber2))

            baseline_x_array = np.array([self.hyperspectral_image_integrated.columns[closest_index_lower],self.hyperspectral_image_integrated.columns[closest_index_upper]])
            baseline_y_array = active_image.loc[:,baseline_x_array]
            self.area_under_baseline = trapz(baseline_y_array,x=baseline_x_array) if mode == 'sig_to_base' else 0

            self.monochrome_image = self.hyperspectral_image_integrated.iloc[:,closest_index_lower].values-self.hyperspectral_image_integrated.iloc[:,closest_index_upper].values - self.area_under_baseline
            self.monochrome_image = pd.DataFrame(self.monochrome_image,index=active_image.index,columns=[mode + '_' + str(wavenumber1) + '_' + str(wavenumber2)])
        elif mode == 'pca':
            self.pca_results = self.principal_component_analysis(pca_components,active_spectra=active_image)
            self.monochrome_image = self.pca_results['scores']

            #reconstruction_pca_components = 3
            #self.reconstructed_pca_image = pd.DataFrame(np.dot(self.pca_results['scores'].iloc[:,0:reconstruction_pca_components],
            #                                self.pca_results['loadings'].iloc[0:reconstruction_pca_components,:])
            #                                + self.mean_spectrum(active_image = active_image).values,
            #                                index = active_image.index,columns = active_image.columns)
        elif mode == 'ref_spec_fit':
            self.monochrome_image = self.reference_spectra_fit(ref_spec,fit_mode,max_iter,initial_guess,lower_bounds,upper_bounds,active_spectra = active_image)
            self.fitted_spectra = pd.DataFrame(np.dot(self.monochrome_image.values,ref_spec.values),index = self.monochrome_image.index,columns = ref_spec.columns)
        else:
            self.monochrome_image = pd.DataFrame(np.zeros((len(active_image.index),1)),index=active_image.index,columns=['empty image'])

        return self.monochrome_image

    def generate_intensity_projections(self,col_index):
        """Generates maximum intensity projection from numpy array with 8-bit images. 
        Export is optional if export path and file name are given. """
        
        assert hasattr(self,'monochrome_image'),'No monochrome image exists, call multi2monochrome first.'
        
        self.MinIP_xyPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,1)).min().unstack()
        self.MinIP_xzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,2)).min().unstack()
        self.MinIP_yzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(1,2)).min().unstack()
        self.MaxIP_xyPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,1)).max().unstack()
        self.MaxIP_xzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,2)).max().unstack()
        self.MaxIP_yzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(1,2)).max().unstack()
        self.AvgIP_xyPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,1)).mean().unstack()
        self.AvgIP_xzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,2)).mean().unstack()
        self.AvgIP_yzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(1,2)).mean().unstack()
        #self.SumIP_xyPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,1)).sum().unstack()
        #self.SumIP_xzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(0,2)).sum().unstack()
        #self.SumIP_yzPlane_monochrome = self.monochrome_image.loc[:,col_index].groupby(level=(1,2)).sum().unstack()
        
        self.zScanProjections = pd.concat([self.monochrome_image.loc[:,col_index].groupby(level=2).min(),self.monochrome_image.loc[:,col_index].groupby(level=2).max(),self.monochrome_image.loc[:,col_index].groupby(level=2).mean()],axis=1)
        self.zScanProjections.columns = ['zScanMin','zScanMax','zScanAverage']
        self.yScanProjections = pd.concat([self.monochrome_image.loc[:,col_index].groupby(level=1).min(),self.monochrome_image.loc[:,col_index].groupby(level=1).max(),self.monochrome_image.loc[:,col_index].groupby(level=1).mean()],axis=1)
        self.yScanProjections.columns = ['yScanMin','yScanMax','yScanAverage']
        self.xScanProjections = pd.concat([self.monochrome_image.loc[:,col_index].groupby(level=0).min(),self.monochrome_image.loc[:,col_index].groupby(level=0).max(),self.monochrome_image.loc[:,col_index].groupby(level=0).mean()],axis=1)
        self.xScanProjections.columns = ['xScanMin','xScanMax','xScanAverage']
    
    ###########################
#####      export methods     #################################################
    ###########################
    
    def export_intensity_projections(self,export_path):
        assert hasattr(self,'monochrome_image'),'No monochrome image exists, call multi2monochrome first.'
        
        for curr_column in self.monochrome_image.columns:
            self.generate_intensity_projections(col_index=curr_column)
            
            self.MinIP_xyPlane_monochrome.to_csv(export_path + curr_column + '_MinIP_xyPlane.txt',sep='\t')
            self.MinIP_xzPlane_monochrome.to_csv(export_path + curr_column + '_MinIP_xzPlane.txt',sep='\t')
            self.MinIP_yzPlane_monochrome.to_csv(export_path + curr_column + '_MinIP_yzPlane.txt',sep='\t')
            self.MaxIP_xyPlane_monochrome.to_csv(export_path + curr_column + '_MaxIP_xyPlane.txt',sep='\t')
            self.MaxIP_xzPlane_monochrome.to_csv(export_path + curr_column + '_MaxIP_xzPlane.txt',sep='\t')
            self.MaxIP_yzPlane_monochrome.to_csv(export_path + curr_column + '_MaxIP_yzPlane.txt',sep='\t')
            self.AvgIP_xyPlane_monochrome.to_csv(export_path + curr_column + '_AvgIP_xyPlane.txt',sep='\t')
            self.AvgIP_xzPlane_monochrome.to_csv(export_path + curr_column + '_AvgIP_xzPlane.txt',sep='\t')
            self.AvgIP_yzPlane_monochrome.to_csv(export_path + curr_column + '_AvgIP_yzPlane.txt',sep='\t')     
            
            self.zScanProjections.to_csv(export_path + curr_column + '_zScanProjections.txt',sep='\t')
            self.yScanProjections.to_csv(export_path + curr_column + '_yScanProjections.txt',sep='\t')
            self.xScanProjections.to_csv(export_path + curr_column + '_xScanProjections.txt',sep='\t')
        
    def export_stack(self,export_path,axis = 'z'):
        assert hasattr(self,'monochrome_image'),'No monochrome image exists, call multi2monochrome first.'
        
        if axis == 'z':
            curr_axis_values = self.get_coord_values('coded',axis='z',active_image = self.monochrome_image)
        elif axis == 'y':
            curr_axis_values = self.get_coord_values('coded',axis='y',active_image = self.monochrome_image)
        elif axis == 'x':
            curr_axis_values = self.get_coord_values('coded',axis='x',active_image = self.monochrome_image)
        
        for curr_coord in curr_axis_values:
            for curr_column in self.monochrome_image.columns:
                if axis == 'z':
                    curr_dataset = self.xy_slice(curr_coord,curr_column)
                elif axis == 'y':
                    curr_dataset = self.xz_slice(curr_coord,curr_column)
                elif axis == 'x':
                    curr_dataset = self.yz_slice(curr_coord,curr_column)
                
                curr_dataset.to_csv(export_path + curr_column + '_' + axis + '_slice_' + str(curr_coord) + '.txt',sep='\t')
        
    def __decode_image_index(self,active_image):
        active_image_copy = active_image.copy()
        active_image_index_frame = active_image_copy.index.to_frame()
        active_image_index_frame.columns = ['x_values','y_values','z_values']
        active_image_new_index = pd.MultiIndex.from_frame(active_image_index_frame/self.coord_conversion_factor)
        
        active_image_copy.index = active_image_new_index
        return active_image_copy
    
    ###########################
#####     extract methods     #################################################
    ###########################
    
    def xScan(self,yPos,zPos,col_index):
        x_scans = self.monochrome_image.xs((yPos,zPos),level=[1,2])
        return x_scans.loc[:,col_index]
    
    def yScan(self,xPos,zPos,col_index):
        y_scans = self.monochrome_image.xs((xPos,zPos),level=[0,2])
        return y_scans.loc[:,col_index]
    
    def zScan(self,xPos,yPos,col_index):
        z_scans = self.monochrome_image.xs((xPos,yPos),level=[0,1])
        return z_scans.loc[:,col_index]
    
    def xz_slice(self,yPos,col_index):
        xz_slices = self.monochrome_image.xs(yPos,level=1)
        return xz_slices.loc[:,[col_index]].unstack()
        
    def yz_slice(self,xPos,col_index):
        yz_slices = self.monochrome_image.xs(xPos,level=0)
        return yz_slices.loc[:,[col_index]].unstack()
    
    def xy_slice(self,zPos,col_index):
        xy_slices = self.monochrome_image.xs(zPos,level=2)
        return xy_slices.loc[:,[col_index]].unstack()