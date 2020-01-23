# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:20:27 2020

@author: AlMaMi
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from scipy.integrate import cumtrapz
from scipy.signal import savgol_filter  # ,find_peaks
from sklearn.decomposition import PCA

######### import own packagaes ##############
import preprocessing.baseline_correction as baseline_correction
import regression.nonlinear_regression as nl_reg
#############################################


class spectroscopy_data:
    def __init__(self, spectral_data):
        self.hyperspectral_image = spectral_data

        self.wavenumbers = self.hyperspectral_image.columns.to_numpy()

        self.baseline_data = {'SNIP': pd.DataFrame(np.zeros_like(self.hyperspectral_image.values), index=self.hyperspectral_image.index, columns=self.hyperspectral_image.columns),
                              'ALSS': pd.DataFrame(np.zeros_like(self.hyperspectral_image.values), index=self.hyperspectral_image.index, columns=self.hyperspectral_image.columns),
                              'iALSS': pd.DataFrame(np.zeros_like(self.hyperspectral_image.values), index=self.hyperspectral_image.index, columns=self.hyperspectral_image.columns),
                              'drPLS': pd.DataFrame(np.zeros_like(self.hyperspectral_image.values), index=self.hyperspectral_image.index, columns=self.hyperspectral_image.columns),
                              'none': pd.DataFrame(np.zeros_like(self.hyperspectral_image.values), index=self.hyperspectral_image.index, columns=self.hyperspectral_image.columns)}

    def check_active_image(self, active_image):
        if active_image is None:
            active_image = self.hyperspectral_image
        return active_image

    ###############################
#####    preprocessing methods    #############################################
    ###############################

    def mean_center(self, active_image=None):
        active_image = self.check_active_image(active_image)

        mean_centered_data = pd.DataFrame(preprocessing.scale(active_image, axis=0, with_std=False), index=active_image.index, columns=active_image.columns)
        #mean_centered_data = pd.DataFrame(active_image.subtract(active_image.mean(axis=0),axis=1),index=active_image.index,columns=active_image.columns)
        return mean_centered_data.round(decimals=6)

    def standard_normal_variate(self,active_image=None):
        active_image = self.check_active_image(active_image)

        #processed_data = pd.DataFrame(preprocessing.scale(active_image,axis=1,with_std=True),index=active_image.index,columns=active_image.columns)
        SNV_scaled_data = pd.DataFrame(active_image.subtract(active_image.mean(axis=1),axis=0).divide(active_image.std(axis=1),axis=0),index=active_image.index,columns=active_image.columns)
        return SNV_scaled_data.round(decimals=6)

    def clip_wavenumbers(self,wn_limits,active_image=None):
        active_image = self.check_active_image(active_image)

        wn_limits = np.array(wn_limits)

        lower_wn = wn_limits[:,0]
        upper_wn = wn_limits[:,1]

        lower_wn = lower_wn[np.argsort(-lower_wn)]
        upper_wn = upper_wn[np.argsort(-upper_wn)]

        closest_index_to_lower_wn = np.argmin(np.abs(active_image.columns.values[:,np.newaxis]-lower_wn),axis=0)
        closest_index_to_upper_wn = np.argmin(np.abs(active_image.columns.values[:,np.newaxis]-upper_wn),axis=0)

        clipping_index = np.concatenate(np.array([np.r_[closest_index_to_upper_wn[ii]:closest_index_to_lower_wn[ii]+1] for ii in np.arange(len(closest_index_to_lower_wn))]))

        clipped_data = active_image.iloc[:,clipping_index]
        return clipped_data

    def clip_samples(self,x_limits = None,y_limits = None,z_limits = None,active_image = None):
        active_image = self.check_active_image(active_image)

        x_clipping_mask = self.generate_sample_clipping_mask(active_image,x_limits,0)
        y_clipping_mask = self.generate_sample_clipping_mask(active_image,y_limits,1)
        z_clipping_mask = self.generate_sample_clipping_mask(active_image,z_limits,2)

        clipped_data = active_image.loc[(x_clipping_mask,y_clipping_mask,z_clipping_mask)]
        return clipped_data

    def savitzky_golay(self,deriv=0,savgol_points=9,poly_order=2,active_image=None):
        active_image = self.check_active_image(active_image)

        sav_gol_filtered_data = pd.DataFrame(savgol_filter(active_image,1+2*savgol_points,poly_order,deriv=deriv,axis=1),index=active_image.index,columns=active_image.columns).round(decimals=6)
        return sav_gol_filtered_data

    def median_filter(self,window=5,active_image=None):
        active_image = self.check_active_image(active_image)

        edge_value_count = int((window-1)/2)
        median_filtered_data = active_image.rolling(window,axis=1,center=True).median().iloc[:,edge_value_count:-edge_value_count]

        return median_filtered_data.round(decimals=6)

    def pca_smoothing(self,pca_components=3,active_image = None):
        active_image = self.check_active_image(active_image)

        pca_results = self.principal_component_analysis(pca_components,active_image = active_image)
        #pca_result is also calculated in multi2monochrome, possibly it can be bundled in one place

        reconstructed_pca_image = pd.DataFrame(np.dot(pca_results['scores'],
                                    pca_results['loadings'])
                                    + self.mean_spectrum(active_image = active_image).values,
                                    index = active_image.index,columns = active_image.columns)

        return reconstructed_pca_image

    def integrate_image(self,active_image=None):
        active_image = self.check_active_image(active_image)

        hyperspectral_image_integrated = pd.DataFrame(-cumtrapz(active_image,x=active_image.columns+0,initial=0),index=active_image.index,columns=active_image.columns).round(decimals=6)
        return hyperspectral_image_integrated

    def baseline_correction(self,lam=100,lam_1=100,p=0.01,eta=0.5,n_iter=100,conv_crit=0.001,active_image=None,alg='SNIP'):
        active_image = self.check_active_image(active_image)

        baseline_types = ['ALSS','iALSS','drPLS','SNIP']
        assert alg in baseline_types,'baseline type unknown'

        if alg == baseline_types[0]:
            self.baseline_data[alg] = pd.DataFrame(baseline_correction.generate_baseline(active_image.values, baseline_types[0], lam=lam,p=p,n_iter=n_iter,conv_crit=conv_crit),index = active_image.index,columns = active_image.columns)
        elif alg == baseline_types[1]:
            self.baseline_data[alg] = pd.DataFrame(baseline_correction.generate_baseline(active_image.values, baseline_types[1], wavenumbers=active_image.columns.to_numpy(),lam=lam,lam_1=lam_1,p=p,n_iter=n_iter,conv_crit=conv_crit),index = active_image.index,columns = active_image.columns)
        elif alg == baseline_types[2]:
            self.baseline_data[alg] = pd.DataFrame(baseline_correction.generate_baseline(active_image.values, baseline_types[2], lam=lam, eta=eta, n_iter=n_iter, conv_crit=conv_crit),index = active_image.index,columns = active_image.columns)
        elif alg == baseline_types[3]:
            self.baseline_data[alg] = pd.DataFrame(baseline_correction.generate_baseline(active_image.values, baseline_types[3], n_iter=n_iter),index = active_image.index,columns = active_image.columns)

        corrected_data = active_image - self.baseline_data[alg]
        return corrected_data.round(decimals=6)

    def generate_sample_clipping_mask(self,active_image,limits,dimension):
        if limits is not None:
            limits = np.array(limits)
            lower_index = limits[:,0]
            lower_index = lower_index[np.argsort(lower_index)]
            upper_index = limits[:,1]
            upper_index = upper_index[np.argsort(upper_index)]
            lower_clipping = np.empty((len(lower_index),len(active_image.index)),dtype=bool)
            upper_clipping = np.empty((len(upper_index),len(active_image.index)),dtype=bool)

            for ii,(curr_lower_index,curr_upper_index) in enumerate(zip(lower_index,upper_index)):
                lower_clipping[ii] = active_image.index.get_level_values(dimension) >= curr_lower_index
                upper_clipping[ii] = active_image.index.get_level_values(dimension) <= curr_upper_index

            clipping_mask = np.sum(lower_clipping*upper_clipping,axis = 0,dtype = bool)
        else:
            clipping_mask = np.full(len(active_image.index),True)
        return clipping_mask

    ####################################
##### spectrum characteristics methods ########################################
    ####################################

    def mean_spectrum(self,active_image=None):
        active_image = self.check_active_image(active_image)

        mean_spectrum = active_image.mean()
        return mean_spectrum.round(decimals=6)

    def max_spectrum(self,active_image=None):
        active_image = self.check_active_image(active_image)

        max_spectrum = active_image.max()
        return max_spectrum

    def min_spectrum(self,active_image=None):
        active_image = self.check_active_image(active_image)

        min_spectrum = active_image.min()
        return min_spectrum

    def std(self,active_image=None):
        active_image = self.check_active_image(active_image)

        std = active_image.std()
        return std.round(decimals=6)

    ####################################
##### spectrum analysis methods ########################################
    ####################################

    # PCA reconstruction is implemented in PCA_viewer and should be transferred to here

    def principal_component_analysis(self,pca_components,active_image = None):
        active_image = self.check_active_image(active_image)

        pca = PCA(n_components=pca_components)
        scores = pca.fit_transform(active_image)
        loadings = pca.components_.T# * np.sqrt(pca.explained_variance_)

        list_pc_components = ['pc' + str(ii) for ii in np.arange(pca_components)]

        return {'scores':pd.DataFrame(scores,index=active_image.index,columns=list_pc_components),
                'loadings':pd.DataFrame(loadings.T,index=list_pc_components,columns=active_image.columns),
                'explained_variance':pca.explained_variance_ratio_}

    def reference_spectra_fit(self,reference_data,mode,maxiter,initial_guess,lower_bounds,upper_bounds,active_image=None):#experimental
        active_image = self.check_active_image(active_image)

        reference_components = len(reference_data.index)
        if lower_bounds is None:
            lower_bounds = np.zeros(reference_components)
        if upper_bounds is None:
            upper_bounds = np.full(reference_components,1000)
        list_reference_components = ['comp' + str(ii) for ii in np.arange(reference_components)]

        fit_coeffs = pd.DataFrame(np.empty((len(active_image.index),len(reference_data.index))),index=active_image.index,columns=list_reference_components)
        #self.fit_results = np.empty(len(active_image.index),dtype=object)

        for ii in tqdm(np.arange(len(active_image.index))):
            if mode == 'Evolutionary':
                boundaries = list(zip(lower_bounds,upper_bounds))
                current_fit_result = nl_reg.dataset_regression(active_image.iloc[ii,:],reference_data,boundaries=boundaries,max_iter=maxiter,alg = 'evo')
            elif mode == 'Levenberg-Marquardt':
                if initial_guess is None:
                    initial_guess = np.ones(reference_components)
                current_fit_result = nl_reg.dataset_regression(active_image.iloc[ii,:],reference_data,initial_guess = initial_guess,alg ='lm')#,bounds=(lower_bounds,upper_bounds)
            fit_coeffs.iloc[ii,:] = current_fit_result.x
            #self.fit_results[ii] = current_fit_result
        return fit_coeffs
    
#     def find_peaks(self,active_image=None):#is still experimental
#         active_image = self.check_active_image(active_image)
#         
#         processed_data = [find_peaks(row,height=200,prominence=100) for row in active_image.values]
#         
#         return processed_data