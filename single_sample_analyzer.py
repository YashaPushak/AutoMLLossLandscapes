import copy

import numpy as np
import streamlit as st

from benchmark_analyzer import Benchmark

class SingleSampleBenchmark(Benchmark):
    """SingleSampleBenchmark

    This class overloads the test for parameter independence so that it can be
    applied to confidence intervals for the objective function instead of on
    samples of the objective function. It is intended for use on scenarios 
    where there is only a single sample available for the objective function
    value and confidence intervals therefore need to be approximated using 
    some other method. 

    This class therefore assumes that self.y_bounds is initialized to something
    with meaningful values for the confidence intervals. 
    """

    def _calculate_partials(self, *args, **kwargs):
        """_calculate_partials

        Prior to calling the parent implementation of _calculate_partials, this
        method encodes the upper and lower bounds into self.y_samples, such 
        that each adjacent parameter alternatives between having an upper bound
        and a lower bound. 
        """
        #for hp in self.config_space.get_hyperparameters():
        #    if isinstance(hp, CategoricalHyperparameter):
        #        raise ValueError("Categorical hyperparameters are not"
        #                         "currently supported for SingleSampleBenchmark"
        #                         "._calculate_partials()")

        # Save for later
        y_samples = self.y_samples
        try:
            # Replace the samples with the bounds
            self.y_samples = copy.deepcopy(self.y_bounds)
            # Now call the parent implementation
            partials = super()._calculate_partials(*args, **kwargs)
        except:
            # Restore the state of y_samples
            self.y_samples = y_samples
            raise
        # Restore the state of y_samples
        self.y_samples = y_samples
        # partials is now a dict of mapping tuples of hyper-parameters to
        # a grid of upper and lower bounds on the partial derivatives, but
        # the upper and lower bounds alternate being sorted along the last
        # dimension, so they need to be sorted.
        for hps in partials:
            partials[hps] = np.sort(partials[hps], -1)
        return partials 

    def _embed_y(self, X, y_samples, *args, **kwargs):
        """_embed_y

        Encodes the y_samples using the parent, and then swaps every second
        value of the lower bound and the upper bound in the grid.
        Note: y_samples should be y_bounds due to the swap in
        _calculate_partials (see above). 
        """

        embedded_y, value_lists, dimensions = super()._embed_y(X, y_samples)

        def flip_along_axis(data, axis):
            """flip_along_axis

            flip the objective function sample values for every second value
            along the specified axis.
            """
            # Move the axis along which we want to flip the samples to the
            # front 
            data = np.swapaxes(data, 0, axis)
            # Flip every second set of samples along that axis
            data[::2,...] = np.flip(data[::2,...], axis=-1)
            # And return the axes to their original locations
            data = np.swapaxes(data, 0, axis)
            return data

        for axis in range(len(embedded_y.shape)-1):
            embedded_y = flip_along_axis(embedded_y, axis)

        return embedded_y, value_lists, dimensions

    def test_reject_parameter_independence(self, *args, **kwargs):
        """test_reject_parameter_independence

        A modified version of the test for significance that operates on the
        confidence intervals for the objective function samples instead of on
        the raw sample values themselves.
        """
        interactions = self._calculate_partials(*args, **kwargs)

        def get_percent_dependent(interaction):
            samples = interaction.reshape((-1, 2))
            # We replaced large objective function values with nans in order to
            # exclude them from the analysis. Now we can drop all nans that 
            # propogated through to this point.
            samples = samples[np.where(~np.isnan(samples))[0]]
            contains_zero = np.logical_and(samples[:,0] <= 0,
                                           0 <= samples[:,1])
            return (1 - np.mean(contains_zero))*100, len(contains_zero)

        percent_dependent = {}
        partial_intervals = {}
        total_percent_dependent = 0
        total_n_samples = 0
        for hps in interactions:
            parsed_hps = tuple([self.hyperparameters[hp] for hp in hps])
            dependent, n_samples = get_percent_dependent(interactions[hps])
            percent_dependent[parsed_hps] = dependent
            partial_intervals[parsed_hps] = interactions[hps].reshape((-1, interactions[hps].shape[-1]))
            total_percent_dependent += dependent
            total_n_samples += 1
            
        return percent_dependent, total_percent_dependent/total_n_samples, \
               partial_intervals












