# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyRegression.nonlinear_regression import calc_function

from pyAnalytics.hplc_data import hplc_data
from pyAnalytics.hplc_calibration import hplc_calibration


class hplc_prediction():
    """
    Contains prediction methods for 3D HPLC data containing time resolved
    absorption spectra, e.g. measured with a DAD detector. The underlying
    calibration can be univariate or multivariate.
    """
    def __init__(self, samples, calibrations):
        """
        Only collects the data needed for the different prediction methods.

        Apart from that does nothing.

        Parameters
        ----------
        samples : list of hplc_data instances
            List of complete datasets from the HPLC measurements, collected in
            instances of hplc_data.
        calibrations : list of hplc_calibration instances
            Each element of the list must be a list itself. If more than one
            calibration is contained in an element, they must share the same
            time and spectral constraints, and in the advanced classical least
            squares algorithm, the respective calibration datasets are all used
            for spectral fitting.

        Returns
        -------
        None.

        """
        self.samples = samples
        self.calibrations = calibrations

    def simple_prediction(self):
        """
        Predicts concentrations of n samples by the classical method.

        Based on first integrating elugrams at all wavelengths and subsequent
        classical least squares fitting of the resulting spectrum with the m
        calibration data K present in self.calibrations. Thus, the separation
        of the spectral information is lost in the elugram regions present in
        the calibrations. Data are analyzed based on time limits and wavelength
        limits given in the calibrations. This procedure works well for
        baseline separated peaks and it is possible to use wavelength ranges
        (multivariate) or a single wavelength (univariate, as routinely done in
        HPLC analysis).

        Returns
        -------
        predictions : ndarray
            Contains the predicted concentrations of the n samples for m
            components determined by the m calibrations. Shape is (n, m).

        """
        cals_squeezed = [
            item for sublist in self.calibrations for item in sublist]
        predictions = np.zeros((len(self.samples), len(cals_squeezed)))
        for ii, sample in enumerate(self.samples):
            for jj, calibration in enumerate(cals_squeezed):
                prediction = np.dot(
                        np.linalg.inv(np.dot(calibration.K.T, calibration.K)),
                        np.dot(
                            calibration.K.T,
                            sample.integrate_all_data(
                                time_limits=calibration.time_limits,
                                wavelength_limits=calibration.wavelength_limits
                                ).values[np.newaxis].T))
                predictions[ii, jj] = prediction.item()

        return predictions

    def advanced_prediction(self):
        """
        Predicts concentrations of n samples including chemometric methods.

        Based on first fitting all spectra from the different time points
        with the calibration data given with a classical least squares method.
        By giving more than one calibration for a certain time interval,
        overlapping peaks might be resolved based on differences in their
        spectra. Thus, the separation of the spectral information is used for
        data analysis. Data are analyzed based on time limits and wavelength
        limits given in the calibrations. This procedure works best if some
        separation of the different components in retention time and spectrum
        exists, more separation is better. Baseline separated peaks are however
        not necessary. It is only possible to use wavelength ranges
        (multivariate), using only a single wavelength (univariate) is not
        possible because the classical least squares fit of the spectra will
        fail.

        Raises
        ------
        ValueError
            In case calibration data is based on a single wavelength only.

        Returns
        -------
        predictions : ndarray
            Contains the predicted concentrations of the n samples for m
            components determined by the m calibrations. Shape is (n, m).

        """
        cal_set_sizes = [len(cal_set) for cal_set in self.calibrations]
        number_of_cals = sum(cal_set_sizes)
        predictions = np.zeros((len(self.samples), number_of_cals))
        
        index_counter = 0
        for curr_set_index, curr_cals in enumerate(self.calibrations):
            # calibrations in one set must have equal time and wavelength range
            Ks = []
            for calibration in curr_cals:
                Ks.append(np.squeeze(calibration.K))
            Ks = np.array(Ks).T
            if len(Ks.shape) == 1:
                raise ValueError(
                    'Singular wavelength used for multivariate calibration.')
            time_limits = curr_cals[0].time_limits
            wavelength_limits = curr_cals[0].wavelength_limits

            for ii, sample in enumerate(self.samples):
                sample_cropped = sample.crop_data(
                    time_limits=time_limits, wavelength_limits=wavelength_limits)

                curr_pred = np.dot(
                    np.linalg.inv(np.dot(Ks.T, Ks)),
                    np.dot(Ks.T, sample_cropped.T)
                    )
                curr_pred = np.trapz(np.squeeze(curr_pred), x=sample_cropped.index)

                predictions[
                    ii, index_counter:index_counter+cal_set_sizes[
                        curr_set_index]] = curr_pred
            index_counter = index_counter+cal_set_sizes[curr_set_index]

        return predictions


if __name__ == "__main__":

    def simulate_hplc_data(concentrations, retention_times,
                           spectrum_wavelengths, spectrum_amplitudes,
                           spectrum_widths,
                           wavelengths=np.linspace(200, 400, 201),
                           times=np.linspace(0, 10, 1001),
                           noise_level=0.05):
        number_of_components = len(concentrations)
        # first step: pure component absorption spectra are calculated and
        # stored in uv_spectra
        uv_spectra = np.zeros((number_of_components, len(wavelengths)))
        for ii, (curr_amp, curr_wl, curr_width) in enumerate(
                zip(spectrum_amplitudes, spectrum_wavelengths,
                    spectrum_widths)):
            curr_y_offset = len(curr_amp)*[0]
            curr_params = np.ravel(np.array(
                [curr_amp, curr_wl, curr_y_offset, curr_width]).T)

            uv_spectra[ii] = calc_function(
                wavelengths, curr_params, 'Gauss')

        # second step: basic chromatogram shapes separately for each component
        # are calculated and stored in chromatograms
        chromatograms = np.zeros((number_of_components, len(times)))
        chrom_params = np.repeat([[1, 0, 0.2]], number_of_components, axis=0)
        chrom_params = np.insert(chrom_params, 1, retention_times, axis=1)
        for jj, curr_params in enumerate(chrom_params):
            chromatograms[jj] = calc_function(times, curr_params, 'Gauss')

        # third step: spectra and chromatrograms are combined to 3D dataset
        # and noise is added
        weighted_spectra = np.array(concentrations)[:, np.newaxis]*uv_spectra
        data_3D = np.dot(chromatograms.T, weighted_spectra)
        noise = np.random.standard_normal(data_3D.shape)*noise_level
        data_3D = pd.DataFrame(
            data_3D + noise, index=times, columns=wavelengths)

        return data_3D

    # calculate simulated HPLC/DAD output for calibration
    concentrations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    calibration_1_dfs = []
    calibration_2_dfs = []
    calibration_3_dfs = []
    calibration_4_dfs = []
    for curr_c in concentrations:
        calibration_1_dfs.append(
            simulate_hplc_data([curr_c], [4.3], [[200, 250]], [[4, 0.8]],
                               [[15, 15]])
            )
        calibration_2_dfs.append(
            simulate_hplc_data([curr_c], [5], [[200, 275]], [[4, 1.3]],
                               [[15, 12]])
            )
        calibration_3_dfs.append(
            simulate_hplc_data([curr_c], [7.3], [[200, 275]], [[4, 1.3]],
                               [[15, 12]])
            )
        calibration_4_dfs.append(
            simulate_hplc_data([curr_c], [7.3], [[200, 300]], [[4, 1.8]],
                               [[15, 12]])
            )

    # generate hplc_calibration instance with simulated calibration data
    calibration_1 = hplc_calibration('DataFrame',
                                     calibration_data=calibration_1_dfs,
                                     concentrations=concentrations,
                                     time_limits=[3, 6],
                                     wavelength_limits=[225, 300])

    calibration_1_uni = hplc_calibration('DataFrame',
                                         calibration_data=calibration_1_dfs,
                                         concentrations=concentrations,
                                         time_limits=[3, 6],
                                         wavelength_limits=[250, 250])

    calibration_2 = hplc_calibration('DataFrame',
                                     calibration_data=calibration_2_dfs,
                                     concentrations=concentrations,
                                     time_limits=[3, 6],
                                     wavelength_limits=[225, 300])

    calibration_3 = hplc_calibration('DataFrame',
                                     calibration_data=calibration_3_dfs,
                                     concentrations=concentrations,
                                     time_limits=[6.1, 9],
                                     wavelength_limits=[225, 300])

    calibration_4 = hplc_calibration('DataFrame',
                                     calibration_data=calibration_4_dfs,
                                     concentrations=concentrations,
                                     time_limits=[6.1, 9],
                                     wavelength_limits=[250, 350])

    # calculate samples with unknown concentration
    unknown_sample_1 = simulate_hplc_data(
        [0.35, 0.2], [4.3, 5], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]])
    unknown_sample_1 = hplc_data('DataFrame', data=unknown_sample_1)

    unknown_sample_2 = simulate_hplc_data(
        [0.27, 0.55], [4.3, 5], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]])
    unknown_sample_2 = hplc_data('DataFrame', data=unknown_sample_2)
    
    unknown_sample_3 = simulate_hplc_data(
        [3, 2], [4.3, 7.3], [[200, 250], [200, 275]],
        [[4, 0.8], [4, 1.3]], [[15, 15], [15, 12]])
    unknown_sample_3 = hplc_data('DataFrame', data=unknown_sample_3)
    
    unknown_sample_4 = simulate_hplc_data(
        [0.9, 1.4, 0.7], [4.3, 5, 7.3], [[200, 250], [200, 275], [200,300]],
        [[4, 0.8], [4, 1.3], [4, 1.8]], [[15, 15], [15, 12], [15, 12]])
    unknown_sample_4 = hplc_data('DataFrame', data=unknown_sample_4)

    unknown_sample_5 = simulate_hplc_data(
        [0.49], [4.3], [[200, 250]],
        [[4, 0.8]], [[15, 15]])
    unknown_sample_5 = hplc_data('DataFrame', data=unknown_sample_5)

    # predict unknown concentrations with multivariate calibrations
    predicted_concentrations = hplc_prediction(
        [unknown_sample_1, unknown_sample_2], [[calibration_1, calibration_2]])
    unknown_concentrations_simple = predicted_concentrations.simple_prediction()
    unknown_concentrations_advanced = predicted_concentrations.advanced_prediction()

    predicted_concentrations_2 = hplc_prediction(
        [unknown_sample_3], [[calibration_1], [calibration_3]])
    unknown_concentrations_simple_2 = predicted_concentrations_2.simple_prediction()
    unknown_concentrations_advanced_2 = predicted_concentrations_2.advanced_prediction()
    
    predicted_concentrations_3 = hplc_prediction(
        [unknown_sample_4, unknown_sample_5], [[calibration_1, calibration_2], [calibration_4]])
    unknown_concentrations_simple_3 = predicted_concentrations_3.simple_prediction()
    unknown_concentrations_advanced_3 = predicted_concentrations_3.advanced_prediction()

    print('Correct_concentrations:\n0.35, 0.2\n0.27, 0.55')
    print('Predicted concentrations (multivariate, simple):\n',
          unknown_concentrations_simple)
    print('Predicted concentrations (multivariate, advanced):\n',
          unknown_concentrations_advanced)

    print('Simple sample, simple (3, 2):\n', unknown_concentrations_simple_2)
    print('Simple sample, advanced (3, 2):\n', unknown_concentrations_advanced_2)
    print('Complex sample, simple (0.9, 1.4, 0.7):\n', unknown_concentrations_simple_3)
    print('Complex sample, advanced (0.9, 1.4, 0.7):\n', unknown_concentrations_advanced_3)

    # plot some data
    plt.figure(0)
    plt.plot(calibration_1.calibration_data[4].raw_data.columns, calibration_1.calibration_data[4].raw_data.loc[4.3, :],
             calibration_2.calibration_data[4].raw_data.columns, calibration_2.calibration_data[4].raw_data.loc[5, :])

    plt.figure(1)
    plt.plot(unknown_sample_1.raw_data.index, unknown_sample_1.raw_data.loc[:,200])

    plt.figure()
    plt.plot(calibration_1.K)
    plt.plot(calibration_2.K)