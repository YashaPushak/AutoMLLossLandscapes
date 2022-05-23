import numpy as np
import pandas as pd

import ConfigSpace

from single_sample_analyzer import SingleSampleBenchmark
import lda_landscape

class LDABenchmark(SingleSampleBenchmark):
    def __init__(self, 
                 confidence_interval='1.35465%',
                 confidence_level=0.95, 
                 dataset='olda_wiki',
                 **kwargs):
        self._validate_confidence_interval(confidence_interval)
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level

        X, y, y_bounds, samples, runtimes = self.get_data_dump()
        config_space, hyperparameters = get_config_space()
        samples = samples.reshape((len(samples), 1))

        super().__init__(X, samples, hyperparameters, config_space,
                         confidence_level=confidence_level,
                         y_bounds=y_bounds,
                         scenario_name=dataset,
                         **kwargs)

    def _validate_confidence_interval(self, confidence_interval):
        options =  ['none', '*%']
        if not (isinstance(confidence_interval, str) and
                confidence_interval in options):
            if not '%' in confidence_interval:
                raise ValueError('confidence_interval must be in {}.'
                                 'Provided {}.'.format(options,
                                                       confidence_interval))

    def calculate_confidence_interval(self, samples, confidence_level):

        if self.confidence_interval == 'none':
            cis = self.confidence_interval_none(samples, confidence_level)
        if '%' in self.confidence_interval:
            cis = self.confidence_interval_percent(samples, confidence_level)

        return cis

    def confidence_interval_none(self, samples, confidence_level):
        # Here we calculate no confidence interval. We simply return the sample values
        # as the upper and lower bounds.

        s = np.reshape(samples, (len(samples)))
        return np.array([s, s]).transpose()

    def confidence_interval_percent(self, samples, confidence_level):
        # Here we are just going to use an absolute percentage of the error as the
        # confidence interval
        s = np.reshape(samples, (len(samples)))
        p = float(self.confidence_interval[:-1])/100
        lower = s*(1-p)
        upper = s*(1+p)
        return np.array([lower, upper]).transpose()
        

    def get_data_dump(self):
        lda_on_grid = np.array(lda_landscape.get())
        
        X = lda_on_grid[:,:3]
        y = lda_on_grid[:,3]
        runtimes = lda_on_grid[:,4]

        y_bounds = self.calculate_confidence_interval(y, self.confidence_level)

        return X, y, y_bounds, y, runtimes

def get_config_space():
    kappa_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tau_values = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]
    s_values = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0]
    config_space = {'kappa': kappa_values,
                    'tau': tau_values,
                    's': s_values}

    hyperparameters = ['kappa', 'tau', 's']

    cs = ConfigSpace.ConfigurationSpace()
    for hp in config_space:
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter(hp,
                                                                config_space[hp]))

    return cs, hyperparameters

