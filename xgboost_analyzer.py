import copy 

import numpy as np
import pandas as pd
from scipy import stats

import ConfigSpace

from benchmark_analyzer import Benchmark
import helper


class XGBoostBenchmark(Benchmark):

    def __init__(self,
                 filename='xgboost/results-5-runs-7*3^10.csv',
                 confidence_level=0.95,
                 load_precomputed_intervals=True,
                 n_runs=5,
                 replace_subsample_data=False,
                 *args,
                 **kwargs):
        self.confidence_level = confidence_level

        config_space, hyperparameters = get_config_space()

        data = pd.read_csv(filename, index_col=0)
        if replace_subsample_data:
            # Manually replace the one configuration's data for subsample which we observed
            # got unlucky in 4 out of 5 of its runs and timed out the first time we collected
            # the data, but in 0 out of 5 of them when we investigated by re-running that 1D
            # slice with more hyper-parameter values.
            filename='xgboost/results-5-runs-subsample.csv'
            data_new = pd.read_csv(filename, index_col=0)
            data_new = data_new[data_new['subsample'] == 0.505]
            cols = copy.deepcopy(hyperparameters)
            cols.append('instance')
            # Find all the columns where the new data and the old data match
            match = True
            for col in cols:
                match = np.logical_and(data[col].isin(data_new[col]), match)
            # Splice together the two sets of data
            data = pd.concat([data[~match], data_new])

        data = data.sort_values(hyperparameters)
        X = np.array(data.drop_duplicates(hyperparameters)[hyperparameters])
        y_samples = np.array(data['error']).reshape((len(X), n_runs))
        y_bounds = helper.confidence_interval_student(y_samples, confidence_level)

 

        super().__init__(X, y_samples, hyperparameters, config_space, 
                         confidence_level=confidence_level,
                         y_bounds=y_bounds,
                         scenario_name='xgboost_covertype', 
                         *args, **kwargs)

    def _student_interval(self, sample):
        ci = stats.t.interval(self.confidence_level,
                              len(sample) - 1,
                              loc=np.mean(sample),
                              scale=stats.sem(sample))
        # If all the samples have identical performance measurements,
        # we get nan. We want to just return the point estimate as the
        # confidence interval in this case.
        if np.isnan(ci[0]):
            ci = (list(sample)[0], list(sample)[0])
        return ci

    def _low(self, sample):
        return self._student_interval(sample)[0]

    def _up(self, sample):
        return self._student_interval(sample)[1]
    
 
class XGBoostSubsampleBenchmark(XGBoostBenchmark):
     def __init__(self,
                 filename='xgboost/results-5-runs-subsample.csv',
                 confidence_level=0.95,
                 load_precomputed_intervals=False,
                 n_runs=5,
                 *args,
                 **kwargs):
        self.confidence_level = confidence_level

        config_space, hyperparameters = get_config_space_subsample(filename)

        data = pd.read_csv(filename, index_col=0)
        data = data.sort_values(hyperparameters)
        X = np.array(data.drop_duplicates(hyperparameters)[hyperparameters])
        y_samples = np.array(data['error']).reshape((len(X), n_runs))
        y_bounds = helper.confidence_interval_student(y_samples, confidence_level)

        super(XGBoostBenchmark, self).__init__(X, y_samples, hyperparameters, config_space, 
                                               confidence_level=confidence_level,
                                               y_bounds=y_bounds,
                                               scenario_name='xgboost_covertype_subsample', 
                                               *args, **kwargs)
    

def get_config_space():
    """get_config_space

    Returns
    -------
    config_space : ConfigSpace.configuration_space.ConfigurationSpace
        The configuration space
    hyperparameters : list of str
        The names of the hyperparameters as strings.
    """
    config_space = {'eta': [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
                    # [0,1] [0.3]  -- most important (0.63)
                    'gamma': [0, 5, 10],
                    # [0, 10] [0]
                    'max_depth': [2, 6, 10],
                    # [1, 10] [6] -- second most important (0.032)
                    'min_child_weight': [1, 10.5, 20],
                    # [0, 20] [1]
                    'max_delta_step': [0, 5, 10],
                    # [0, 10] [0]
                    'subsample': [0.01, 0.505, 1],
                    # [0.01, 1] [1] -- third most important (0.028)
                    'colsample_bytree': [0.5, 0.75, 1.0],
                    # [0.5, 1] [1]
                    'colsample_bylevel': [0.5, 0.75, 1.0],
                    # [0.5, 1] [1]
                    'lambda': [1, 5.5, 10],
                    # [1, 10] [1]
                    'alpha': [0, 5, 10],
                    # [0, 10] [0]
                    'num_round': [50, 150, 250]}
                    # [50, 500] [100]i

    cs = ConfigSpace.ConfigurationSpace()    
    for hp in config_space:
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter(hp,
                                                                config_space[hp]))

    return cs, list(config_space.keys())


def get_config_space_subsample(filename='xgboost/results-5-runs-subsample.csv'):
    """get_config_space_subsample

    Returns
    -------
    config_space : ConfigSpace.configuration_space.ConfigurationSpace
        The configuration space
    hyperparameters : list of str
        The names of the hyperparameters as strings.
    """
    df = pd.read_csv(filename)
    config_space = {'subsample': sorted(df['subsample'].unique())}

    cs = ConfigSpace.ConfigurationSpace()    
    for hp in config_space:
        cs.add_hyperparameter(ConfigSpace.OrdinalHyperparameter(hp,
                                                                config_space[hp]))

    return cs, list(config_space.keys())

