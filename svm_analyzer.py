import numpy as np
import pandas as pd
from scipy.stats import norm

import ConfigSpace

from single_sample_analyzer import SingleSampleBenchmark
import svm_landscape

class SVMBenchmark(SingleSampleBenchmark):
    def __init__(self, 
                 confidence_interval='wald', 
                 confidence_level=0.95,
                 dataset='lssvm_motif',
                 **kwargs):
        self.confidence_level = confidence_level
        self._validate_confidence_interval(confidence_interval)
        self.confidence_interval = confidence_interval

        X, y, y_bounds, samples, runtimes = self.get_data_dump()
        config_space, hyperparameters = get_config_space()
        self.runtimes = runtimes

        samples = np.reshape(samples, (len(samples), 1))

        super().__init__(X, samples, hyperparameters, config_space,
                         confidence_level=confidence_level,
                         y_bounds=y_bounds,
                         scenario_name=dataset,
                         **kwargs)

    def _validate_confidence_interval(self, confidence_interval):
        options =  ['wald']
        if not (isinstance(confidence_interval, str) and
                confidence_interval in options):
            raise ValueError('confidence_interval must be in {}.'
                             'Provided {}.'.format(options,
                                                   confidence_interval))

    def calculate_confidence_interval(self, samples, confidence_level):

        if self.confidence_interval == 'wald':
            cis = self.confidence_interval_wald(samples, confidence_level)

        return cis

    def confidence_interval_wald(self, samples, confidence_level):
        # We have a maximum likelihood estimate for the binary classification error
        # from ~20000 test instances. The estimate is actually the mean of the test
        # error from 5 independently trained and evaluated models. However, this
        # mean is all that we have to work with so we are forced to ignore the
        # uncertainty due to training and can only capture the uncertainty from the
        # test instances themselves.

        # Estimated number of errors
        phat = samples
        # Approximate number of test instances
        n = 20000
        
        # Quantile of the standard normal distribution
        alpha = 1-confidence_level
        q = 1 - 0.5*alpha
        z = norm.ppf(q)
        
        lower = phat - z*np.sqrt(phat*(1-phat)/n)         
        upper = phat + z*np.sqrt(phat*(1-phat)/n)

        return np.array([lower, upper]).transpose()
        
    def get_data_dump(self):
        svm_on_grid = np.array(svm_landscape.get())

        X = svm_on_grid[:,:3]
        y = svm_on_grid[:,3]
        runtimes = svm_on_grid[:,4]

        y_bounds = self.calculate_confidence_interval(y, self.confidence_level)

        return X, y, y_bounds, y, runtimes

def get_config_space():
    c_values = [0.1,
               1.0,
               5.0,
               10.0,
               25.0,
               50.0,
               75.0,
               100.0,
               200.0,
               300.0,
               400.0,
               500.0,
               600.0,
               700.0,
               800.0,
               900.0,
               1000.0,
               2000.0,
               3000.0,
               4000.0,
               5000.0,
               6000.0,
               7000.0,
               10000.0,
               1000000.0]
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 1, 1.5, 2, 3, 5]
    epsilon_values = [0.1, 0.01, 0.001, 0.0001]
    config_space = {'c': c_values,
                    'alpha': alpha_values,
                    'epsilon': epsilon_values}

    hyperparameters = ['c', 'alpha', 'epsilon']

    cs = ConfigSpace.ConfigurationSpace()
    for hp in config_space:
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter(hp,
                                                                config_space[hp]))

    return cs, hyperparameters




