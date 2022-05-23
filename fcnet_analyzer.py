import copy 
import itertools

import numpy as np
import scipy.stats as stats
import pandas as pd

from tabular_benchmarks import FCNetSliceLocalizationBenchmark
from tabular_benchmarks import FCNetProteinStructureBenchmark
from tabular_benchmarks import FCNetNavalPropulsionBenchmark
from tabular_benchmarks import FCNetParkinsonsTelemonitoringBenchmark
from tabular_benchmarks import nas_cifar10
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import OrdinalHyperparameter

import helper
from helper import Progress
from benchmark_analyzer import Benchmark

class FCNetBenchmark(Benchmark):

    def __init__(self, 
                 data_dir='./fcnet_tabular_benchmarks/', 
                 dataset='slice_localization', 
                 confidence_interval='student-t',
                 confidence_level=0.95,
                 dataset_type=['validation', 'test'],
                 **kwargs):
        self._validate_confidence_level(confidence_level)
        self._validate_confidence_interval(confidence_interval)
        self._validate_dataset_type(dataset_type)

        datasets = ['slice_localization',
                    'protein_structure',
                    'naval_propulsion',
                    'parkinsons_telemonitoring',
                    'cifar10_A', 'cifar10',
                    'cifar10_B',
                    'cifar10_C']
        if dataset == 'slice_localization':
            b = FCNetSliceLocalizationBenchmark(data_dir=data_dir)
        elif dataset == 'protein_structure':
            b = FCNetProteinStructureBenchmark(data_dir=data_dir)
        elif dataset == 'naval_propulsion':
            b = FCNetNavalPropulsionBenchmark(data_dir=data_dir)
        elif dataset == 'parkinsons_telemonitoring':
            b = FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir)
        #elif dataset in ['cifar10_A', 'cifar10']:
        #    b = nas_cifar10.NASCifar10A(data_dir, False)
        #elif dataset == 'cifar10_B':
        #    b = nas_cifar10.NASCifar10B(data_dir, False)
        #elif dataset == 'cifar10_C':
        #    b = nas_cifar10.NASCifar10C(data_dir, False)
        else:
            raise ValueError('dataset must be one of {}. '
                             'provided {}.'.format(datasets, dataset))

        self.benchmark = b
        config_space = b.get_configuration_space()
        self.confidence_interval = confidence_interval
        self.confidence_level = confidence_level
        if isinstance(dataset_type, list):
            self.dataset_type = dataset_type
        else:
            self.dataset_type = [dataset_type]
        self._num_trials = 4

        X, y, y_bounds, samples, runtimes, hyperparameters \
            = self.get_data_dump(config_space, dataset)

        # Temporarily use only the validation data instead of the test data
        #samples = samples[:, 4:8]
        #print("Warning, we're dropping the validation data and only using "
        #      "the test data")

        self.runtimes = runtimes
        super().__init__(X, samples, hyperparameters, config_space,
                         confidence_level=confidence_level,
                         y_bounds=y_bounds,
                         scenario_name=dataset,
                         **kwargs) 
        
    def _validate_confidence_level(self, confidence_level):
        if not (isinstance(confidence_level, float) and 
               (0 < confidence_level and confidence_level < 1)):
            raise ValueError('confidence_level must be a float in (0,1). '
                             'Provided {}.'.format(confidence_level))

    def _validate_confidence_interval(self, confidence_interval):
        options =  ['student-t', 
                    'bootstrap',
                    'nested student-t bootstrap',
                    'nested bootstrap']
        if not (isinstance(confidence_interval, str) and 
                confidence_interval in options):
            raise ValueError('confidence_interval must be in {}.'
                             'Provided {}.'.format(options,
                                                   confidence_interval))

    def _validate_dataset_type(self, dataset_type):
         if not isinstance(dataset_type, list):
             original = copy.deepcopy(dataset_type)
             dataset_type = [dataset_type]
         for d in dataset_type:
             if d not in ['train', 'validation', 'test']:
                raise ValueError('dataset_type must be a subset of '
                                 '["train", "validation", "test"]. '
                                 'Provided {}.'.format(original))

    def get_data_dump(self, config_space, dataset):
        """
        Gets the raw data, calculates confidence intervals and returns all of it.
        """
        value_lists = []
        name_list = []
        for hp in config_space.get_hyperparameters():
            name_list.append(hp.name)
            if isinstance(hp, OrdinalHyperparameter):
                values = hp.sequence
            elif isinstance(hp, CategoricalHyperparameter):
                values = hp.choices
            else:
                raise NotImplementedError('Only Categorical and Ordinal Hyperparameters are '
                                          'supported at the moment. '
                                          'Provided {}. '.format(type(hp)))
            value_lists.append(list(values))
        value_grid = list(itertools.product(*value_lists))
 
        config_ids = []
        configurations = []
        progress = Progress(len(value_grid),
                            '{}% done enumerating all configurations')
        for values in value_grid:
            progress.update()
            values = dict(zip(name_list, values))
            config = Configuration(config_space, values=values)
            config_id = self.get_unique_configuration_identifier(config)
            config_ids.append(config_id)
            configurations.append(config)
        configuration_table = pd.DataFrame({'Configuration ID': config_ids,
                                            'Configuration': configurations})
        configuration_table = configuration_table.drop_duplicates('Configuration ID')
        progress = Progress(len(value_grid),
                            '{}% done extracting the sample values')
        
        
        # Check to see if we can load the data samples directly
        try:
            samples = helper.loadObj('./samples/', dataset)
            runtime_samples = helper.loadObj('./samples/', 'runtimes-' + dataset)
        except:
            samples = []
            runtime_samples = []
            for config in configuration_table['Configuration']:
                progress.update()
                sample = []
                runtime_sample = []
                for dataset_type in self.dataset_type:
                    sample.append([])
                    runtime_sample.append([])
                    for i in range(int(round(self._num_trials))):
                        result = self.benchmark.objective_function_deterministic(config,
                                                                                 index=i,
                                                                                 dataset=dataset_type) 
                        sample[-1].append(result[0])
                        runtime_sample[-1].append(result[1])
                samples.append(sample)
                runtime_samples.append(runtime_sample)
            samples = np.asarray(samples)
            runtime_samples = np.asarray(runtime_samples)
            helper.saveObj('./samples/', samples, dataset)
            helper.saveObj('./samples/', runtime_samples, 'runtimes-' + dataset)
            helper.saveObj('./samples/', np.asarray(configuration_table['Configuration']),
                           'configurations-' + dataset)

        confidence_level = self.confidence_level

        if self.confidence_interval == 'nested student-t bootstrap':
            cis = self.confidence_interval_nested_student_bootstrap(samples, confidence_level)
        elif self.confidence_interval == 'nested bootstrap':
            cis = self.confidence_interval_nested_bootstrap(samples, confidence_level)
        elif self.confidence_interval == 'student-t':
            cis = self.confidence_interval_student(samples, confidence_level)
        #elif self.confidence_interval == 'bootstrap':
        #    cis = self.confidence_interval_bootstrap(samples, confidence_level)


        X = np.array(pd.DataFrame(value_grid).infer_objects())
        y_bounds = cis
        samples = samples.reshape((len(samples), -1))
        y = np.mean(samples, axis=1)
        runtimes = runtime_samples.reshape((len(samples), -1))
        hyperparameters = name_list       
 
        return X, y, y_bounds, samples, runtimes, hyperparameters
        
    def confidence_interval_nested_bootstrap(self,
                                             samples,
                                             confidence_level):
        n_data = len(samples)
        num_trials = self._num_trials
        num_datasets = 2
        alpha = 1 - confidence_level
        n_bootstrap = int(round(10/(alpha/2) + 1)) 
        # we only have two data points, one for each data set. So rather than using
        # random bootstrap samples we enumerate all of them
        idx = [0,1]
        inner_b_idx = list(itertools.product(idx, idx))
        # Take these bootstrap samples
        b_inner_samples = samples[:,inner_b_idx,:]
        # Calculate the mean of each one and reshape so that we end up with an arary
        # of length n_data, with each entry containing the list of all possible sub
        # statistics
        b_inner_means = np.mean(b_inner_samples, axis=2).reshape((n_data, -1))

        # Calculate the outer bootstrap samples
        outer_b_idx = np.random.randint(0, num_datasets*num_trials, (n_bootstrap, num_trials))
        # Get the outer bootstrap samples
        b_outer_samples = b_inner_means[:, outer_b_idx]
        # And calculate the statistics for them
        b_outer_means = np.mean(b_outer_samples, axis=2)

        # Calculate the confidence interval
        lower, upper = np.quantile(b_outer_means, 
                                   [alpha/2, 1-alpha/2],
                                   axis=1,
                                   interpolation='nearest')
        return np.array([lower, upper]).transpose()

    def confidence_interval_nested_student_bootstrap(self,
                                                       samples,
                                                       confidence_level):
        n_data = len(samples)
        num_trials = self._num_trials
        num_datasets = 2
        alpha = 1 - confidence_level
        n_bootstrap = int(round(10/(alpha/2) + 1)) 
        # we only have two data points, one for each data set. So rather than using
        # random bootstrap samples we enumerate all of them
        idx = [0,1]
        inner_b_idx = list(itertools.product(idx, idx))
        # Take these bootstrap samples
        b_inner_samples = samples[:,inner_b_idx,:]
        # Calculate the mean of each one and reshape so that we end up with an arary
        # of length n_data, with each entry containing the list of all possible sub
        # statistics
        b_inner_means = np.mean(b_inner_samples, axis=2).reshape((n_data, -1))

        # Calculate the outer bootstrap samples
        outer_b_idx = np.random.randint(0, num_datasets*num_trials, (n_bootstrap, num_trials))
        # Get the outer bootstrap samples
        b_outer_samples = b_inner_means[:, outer_b_idx]
        # And calculate the statistics for them
        b_outer_means = np.mean(b_outer_samples, axis=2)
        # Calculate the standard deviation of the sample statistics
        b_outer_std = np.std(b_outer_means, axis=1)

        # Calculate the standard errors for each bootstrap sample
        b_outer_std_errs = np.std(b_outer_samples, axis=2) / np.sqrt(num_trials)

        # Calculate the actual sample means
        sample_means = np.mean(np.mean(samples, axis=1), axis=1)
        # Broadcast them across the bootstrap samples
        sample_means_broadcasted = np.array([sample_means 
                                             for _ in range(n_bootstrap)]).transpose()
        # Calculate the studentized test statistics (q values)
        t_statistics = (b_outer_means - sample_means_broadcasted) / b_outer_std_errs

        # Get the quantiles of the t_statistics
        # When we calculate the lower bound we treat nans as negative inf
        # This is a conservative thing to do, since a nan came from a 0/0 and
        # depending on whether or not you approach this limit from the
        # positive or negative direction you will either get positive or
        # negative infinity. Ultimately, this means that we will end up with
        # slightly larger confidence intervals, so overconfidence will not
        # arise due to poor handling of nans.
        # Interpolation='nearest' is necessary for cases with weird confidence
        # levels such that the quantiles don't correspond exactly to a
        # single data point. If any other interpolation method is used nans
        # end up being returned any time interpolation between infinity and
        # another number occurs. Since we're automatically calculating the
        # number of bootstrap samples to be such that a particular data point
        # while be close to the desired quantile, it doesn't matter too much
        # that we are interpolating here.
        lower = np.quantile(np.nan_to_num(t_statistics,
                                          nan=-np.inf,
                                          posinf=np.inf,
                                          neginf=-np.inf),
                            alpha/2,
                            axis=1,
                            interpolation='nearest')
        # Treat nans as positive infinity. See above
        upper = np.quantile(np.nan_to_num(t_statistics,
                                          nan=np.inf,
                                          posinf=np.inf,
                                          neginf=-np.inf),
                            1-alpha/2,
                            axis=1,
                            interpolation='nearest')

        # Now compute the final confidence interval bounds
        lower_bound = sample_means - b_outer_std*upper
        upper_bound = sample_means - b_outer_std*lower

        return np.array([lower_bound, upper_bound]).transpose()

        
    def confidence_interval_student(self, 
                                    samples,
                                    confidence_level):
        # Treat validation and test samples identically
        samples = samples.reshape((len(samples), -1))
        ci = stats.t.interval(confidence_level,
                              samples.shape[1] - 1,
                              loc=np.mean(samples, axis=1),
                              scale=stats.sem(samples, axis=1))
        ci = np.array(ci).transpose()
        # nans occur everywhere that the standard deviation is 0
        # We want to fill these with confidence intervals equal to the point estimate
        ci = pd.DataFrame(ci)
        ci[0] = ci[0].fillna(pd.Series(samples[:,0]))
        ci[1] = ci[1].fillna(pd.Series(samples[:,0]))
        return np.array(ci)

    def confidence_interval_bootstrap(self,
                                      samples,
                                      confidence_level):
        # Treat validation and test samples identically
        samples = samples.reshape((len(samples), -1))
        alpha = 1 - confidence_level
        n_bootstrap = int(round(10/(alpha/2) + 1))
        bidx = np.random.randint(0, self._num_trials, (n_bootstrap, self._num_trials))
        b_samples = samples[:,bidx]
        b_means = np.mean(b_samples, axis=2)
        ci = np.quantile(b_means, [alpha/2, 1-alpha/2], axis=1).transpose()
        return ci
         
