import time
import itertools
import copy 
from multiprocessing import Pool
from numbers import Number

import tqdm
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scipy.stats import spearmanr
import altair as alt
import streamlit as st

import ConfigSpace
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from ConfigSpace.hyperparameters import OrdinalHyperparameter
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from fanova import fANOVA

from heap import Heap
import helper
from helper import Progress


class Benchmark:

    def __init__(self, 
                 X,
                 y_samples,
                 hyperparameters,
                 config_space,
                 confidence_level=0.95,
                 y_bounds=None,
                 scenario_name='benchmark',
                 permute_objective=False,
                 seed=None,
                 y=None):
        """__init__

        Parameters
        ----------
        X: numpy.ndarray
            A numpy array containing the grid of evaluated configurations. Each
            row is a configuration and each column is a hyper-parameter. The
            order of the hyperparameter columns must match the list of names of
            the hyperparameters in hyperparameters.
        y_samples: numpy.ndarray
            The set of performance measurements (samples) for each 
            configuration contained in X. The number of rows must match
            X, and the entries must be in the same order. Samples within
            each row should be IID, and each samples within each column
            should block on the same features (e.g., training/validiation
            sets).
        hyperparameters: list of str
            The list of hyperparameter names as they appear in the grid X.
            Must be of length equal to X.shape[1].
        config_space: ConfigSpace.configuration_space.ConfigurationSpace
            The configuration space of the algorithm being analyzed. Currently
            only supports categorical and ordinal parameters. Does not 
            support conditional parameters. Integer- or real-valued parameters
            should be converted to ordinal parameters whose values match the
            entries in the configurations grid X.
        y_bounds: numpy.ndarray
            The confidence intervals for the performance of the configurations.
            Must contain the same number of rows and X, and must contain 
            exacty two columns, in the order: [lower bound, upper bound].
            Row entries must be in the same order as the entries in X.
            If None, the bounds will be determined for you automatically by
            calculating student-t-based confidence intervals with the 
            specified significance level. 
        scenario_name : str (optional)
            The name of the scenario being analyzed.
        permute_objective : bool
            If True, the association between the rows of X and y is broken
            by randomly shuffling the rows of y. This allows you to verify that
            the confidence intervals are not too large and/or the configuration
            space is not too connected to be able to reject uni-modality of the
            landscape for a randomly permuted objective function.
        seed : int | None
            The random seed used by the benchmark analyzer.
        y : numpy.ndarray | None
            The best point-estimate for the loss of the configurations. If
            None, it will be calculated for you as the mean of y_samples.
        """
        # Assume one until we can prove otherwise
        self._num_modes = 1
        self._permute_objective = permute_objective
        self._validate_config_space(config_space)
        self.config_space = config_space
        if y is None:
            y = np.mean(y_samples, axis=1)
        if y_bounds is None:
            y_bounds = helper.confidence_interval_student(y_samples,
                                                          confidence_level)
        # @TODO: Validate X, y_samples, y_bounds and hyperparameters.
        self._random = np.random.RandomState(seed)
        self._global_minima = None
        self._dataset = scenario_name
        self.X = X
        self.y = y
        self.y_bounds = y_bounds
        self.y_samples = y_samples
        self.hyperparameters = hyperparameters
        self.confidence_level = confidence_level
        self.fANOVA = None

        n_unique = len(pd.DataFrame(X).drop_duplicates())
        n_original = len(X)
        if n_unique != n_original:
            raise ValueError('The configuration dataframe, X, contains {} '
                             'duplicatated configurations.'
                             ''.format(n_original - n_unique))

        self._initialize_configuration_table(X, 
                                             y, 
                                             y_samples, 
                                             y_bounds,
                                             hyperparameters)
 
    def _validate_config_space(self, config_space):
        if not (isinstance(config_space, ConfigurationSpace) and
                len(config_space.get_conditions()) == 0):
            raise NotImplementedError('Configuration Spaces with conditional hyperparameters '
                                      'are not currently supported.')
        for hp in config_space.get_hyperparameters():
            if not (isinstance(hp, OrdinalHyperparameter) 
                    or isinstance(hp, CategoricalHyperparameter)):
                raise NotImplementedError('Only Categorical and Ordinal Hyperparameters are '
                                          'supported at the moment. '
                                          'Provided {}. '.format(type(hp)))
 
    def _initialize_configuration_table(self, 
                                        X, 
                                        y, 
                                        y_samples, 
                                        y_bounds,  
                                        hyperparameters):
        """
        Creates a pandas dataframe with an entry for every valid configuration
        in self.config_space.
        """
 
        config_ids = []
        configurations = []
        progress = Progress(len(X),
                            '{}% done making the configuration table')
        for values in X:
            progress.update()
            values = dict(zip(hyperparameters, values))
            config = Configuration(self.config_space, values=values)
            config_id = self.get_unique_configuration_identifier(config)
            config_ids.append(config_id)
            configurations.append(config)
        # Randomly permute the objective function confidence intervals if necessary
        if self._permute_objective:
            state = self._random.get_state()
            self._random.shuffle(y)
            self._random.set_state(state)
            self._random.shuffle(y_bounds)
            self._random.set_state(state)
            self._random.shuffle(y_samples)

        # Note that Quality in this table is the minimum solution quality necessary
        # to reach the configuration from the global optimum. Not the estimated quality.
        self._configuration_table = pd.DataFrame({'Configuration ID': config_ids,
                                                  'Configuration': configurations,
                                                  'Visited': False,
                                                  'Parent': None,
                                                  'Quality': float('inf'),
                                                  'Estimated Objective': y,
                                                  'Lower Bound': y_bounds[:,0],
                                                  'Upper Bound': y_bounds[:,1]}) 
        progress = Progress(len(X),
                            '{}% done extracting the sample values')

        self._confidence_interval \
            = dict(zip(np.array(self._configuration_table['Configuration ID']),
                       y_bounds))
        self._objective \
            = dict(zip(np.array(self._configuration_table['Configuration ID']),
                       y))
        self._samples \
            = dict(zip(np.array(self._configuration_table['Configuration ID']),
                       y_samples))

        self._mark_all_unreachable()
        self._stale_table = False
        
    def get_confidence_interval(self, 
                                config):
        config_id = self.get_unique_configuration_identifier(config)
        return self._confidence_interval[config_id]

        
    def get_neighbours(self,
                       config):
        """get_neighbours
 
        Given a configuration, returns its list of neighbours. The
        neighbourhood of a configuration is defined by mutating individual
        parameters of the configuration. Ordinal parameter values can be
        increased or decreased by one unit, and categorical parameter values
        can be changed to any other value.

        Parameters
        ----------
        config : ConfigSpace.configuration_space.Configuration
            The configuration whose neighbours you are getting.

        Returns 
        -------
        neighbours : list of ConfigSpace.configuration_space.Configuration
            The neighbours of config.
        """
        neighbours = []
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, OrdinalHyperparameter):
                # Ordinal parameters may move up or down by one index
                values = []
                idx = list(hp.sequence).index(config[hp.name])
                if idx-1 in range(len(hp.sequence)):
                    values.append(hp.sequence[idx-1])
                if idx+1 in range(len(hp.sequence)):
                    values.append(hp.sequence[idx+1])
            elif isinstance(hp, CategoricalHyperparameter):
                # Any single categorical value may be swapped with another
                values = list(hp.choices)
                values.remove(config[hp.name])
            else:
                raise NotImplementedError('Only Categorical and Ordinal Hyperparameters are '
                                          'supported by get_neighbours() at the moment. '
                                          'Provided {}. '.format(type(hp)))
            for value in values:
                neighbour = copy.deepcopy(config)
                neighbour[hp.name] = value
                neighbours.append(neighbour)

        return neighbours

    def get_unique_configuration_identifier(self,
                                            config):
        """get_unique_configuration_identifier
        Outputs a string representation of the dictionary that contains this configuration.
        However, the dictionary keys are sorted to ensure that the representation is unique. 

        Parameters
        ----------
        config : Configuration
            The configuration to convert to a unique dictionary
        
        Returns
        -------
        s : str
            The unique string representation of config
        """
        if not (isinstance(config, Configuration) or config is None):
            raise ValueError('Expected config to be a Configuration or None. '
                             'Provided {}.'.format(config))
        if config is None:
            return 'None'
        d = config.get_dictionary()
        s = []
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, str):
                value = "'{}'".format(value)
            elif isinstance(value, Number):
                if float(value).is_integer():
                    value = int(value)
                else:
                    value = float(value)
            s .append("'{}': {}".format(key, 
                                        value))
        return '{' + ', '.join(s) + '}'
         
    def test_reject_unimodality(self, stop_early=False, verbose=True):
        """
        Checks to see if the current benchmark is unimodal by applying 
        Dijkstra's algorithm on the configuration space graph defined
        implicitly by get_neighbours(). If, starting from the configuration
        with the global minima lower bound, it is possible to construct a tree
        with monotically increasing possible objective function values
        (according to the confidence intervals) then it is not possible to 
        reject the possibility that the benchmark is unimodal.

        We construct this tree using Dijkstra's algorithm, where the distance
        is the solution quality and an edge exists between a source 
        configuration and a destination configuration if minimum solution 
        quality needed to reach the source is not larger than the upper bound
        for the confidence interval of the destination. The distance for 
        reaching the destination from the source is the maximum of the source's
        distance and the lower bound of the confidence interval for the 
        destination. The search process starts at the global minima for the 
        lower bounds of the confidence intervals. If the resulting tree does not
        span the entire (discretized) configuration state space, then we can 
        conclude that it is not uni-modal. 

        Parameters
        ----------
        stop_early : bool
            If stop_early is True, then the search with terminate as soon as it
            can be concluded that all configurations are reachable from the 
            global optimum using some path. Otherwise, the search process 
            continues until the shortest path to each configuration has been 
            found.
        verbose : bool
            If True, progress about the current state of the search will be
            printed to the console.

        Returns
        -------
        unimodal : bool
            True if the benchmark is significantly non-unimodal, otherwise 
            False.
        """
        heap = Heap()
        self._mark_all_unvisited() 
        config = self.get_smallest_unvisited_minima()
        config_id = self.get_unique_configuration_identifier(config)
        ci = self.get_confidence_interval(config)
        self._mark_reachable(config)
        heap.push(ci[0], config_id, (config, None))

        iteration = 0
        while len(heap) > 0:
            quality, config_id, (config, parent) = heap.pop()
            self._mark_visited(config, quality, parent)
            iteration += 1
            neighbours = self.get_neighbours(config)
            if verbose:
                if iteration % 1000 == 0 or len(heap) == 0:
                   print('Iteration: {0}; Heap size: {1}; Still Unreached: {4}; Neighbours: {2:02d}; Quality: {3:.5f}'
                         ''.format(iteration, len(heap), len(neighbours), 
                                   quality, self._still_unreachable), end='\n', flush=True)
                elif iteration % 100 == 0 and False:
                    print('Iteration: {0}; Heap size: {1}; Still Unreached: {4} Neighbours: {2:02d}; Quality: {3:.5f}'
                         ''.format(iteration, len(heap), len(neighbours), 
                                   quality, self._still_unreachable), end='\r', flush=True)
               
            for neighbour in neighbours:
                if self._was_visited(neighbour):
                    continue
                ci = self.get_confidence_interval(neighbour)
                if quality <= ci[1]:
                    self._mark_reachable(neighbour)
                    heap.push_if_better(max(quality, ci[0]),
                                        self.get_unique_configuration_identifier(neighbour), 
                                        (neighbour, config))
            if stop_early:
                # Check to see if everything is reachable, if it is, we don't actually have to 
                # continue the search to conclude that there are no more modes. However, continuing
                # the search does have the advantage that we can determine which configurations are
                # uniquely reachable from each mode.
                if self._still_unreachable == 0:
                    break
        
        all_visited = self._all_visited()    
        if verbose:
            print('{0:.2f}% of the state space was visited during the search procedure.'
                  ''.format(self._percent_visited))
            print('{} of the configurations were determined to remain unreachable.'
                  ''.format(self._still_unreachable))
            if self._still_unreachable > 0:
                reject = 'reject'
            else:
                reject = 'cannot reject'
            # @TODO: confidence_interval is never initialized in this class. Right
            # now we're relying on the existence of a definition for the 
            # confidence level that comes from the child class
            print('Therefore, we {} the possibility that this benchmark has {} mode{} or less '
                  'with {}% confidence.'.format(reject, 
                                                self._num_modes,
                                                '' if self._num_modes == 1 else 's',
                                                self.confidence_level*100), flush=True)

        #self._configuration_table.to_csv('debug.csv')
        return self._still_unreachable > 0
           
    def count_modes(self, verbose=True):
        """count_modes
        Count the number of modes in the landscape by iteratively applying
        test_reject_unimodality until all configurations have been reached at
        least once.

        Parameters
        ----------
        verbose : bool
            If True, information about the current progress of the search will 
            be printed to the console.

        Returns
        -------
        num_modes : int
            The minimum number of nodes that can be proved must exist given the
            confidence intervals provided.
        """
        self._num_modes = 1
        self._mark_all_unvisited()
        self._mark_all_unreachable()
        not_all_visited = True
        while not_all_visited:
            # The following function actually runs Dijkstra's algorithm from the smallest unvisited
            # point, so calling it multiple times with increasing values to self._num_modes will
            # effectively count the number of modes 
            not_all_visited = self.test_reject_unimodality()
            if not_all_visited:
                self._num_modes += 1

        return self._num_modes 
 
    def _mark_all_unvisited(self):
        self._stale_table = True
        config_ids = np.array(self._configuration_table['Configuration ID'])
        self._visited = dict(zip(config_ids,
                                 np.full(len(config_ids), False)))
        self._quality = dict(zip(config_ids,
                                 np.full(len(config_ids), float('inf'))))
        self._parent = dict(zip(config_ids,
                                np.full(len(config_ids), None)))

    def _mark_all_unreachable(self):
        config_ids = np.array(self._configuration_table['Configuration ID'])
        self._reachable = dict(zip(config_ids,
                                   np.full(len(config_ids), False)))
        self._still_unreachable = len(config_ids)
                                     
    def _mark_visited(self,
                      config,
                      quality,
                      parent):
        self._stale_table = True
        config_id = self.get_unique_configuration_identifier(config)
        self._visited[config_id] = True
        self._quality[config_id] = quality
        self._parent[config_id] = self.get_unique_configuration_identifier(parent)

    def _mark_reachable(self,
                        config):
        # A less strict version of visited. Technically, as soon as every
        # configuration has been reached, we know that the space is uni-modal.
        # Therefore, we don't need to go until we've visited every single configuration.
        self._stale_table = True
        config_id = self.get_unique_configuration_identifier(config)
        if not self._reachable[config_id]:
            # Only decrement the counter if we haven't been able to reach this configuration before
            self._still_unreachable -= 1
            self._reachable[config_id] = True

    def _was_visited(self,
                     config):
        config_id = self.get_unique_configuration_identifier(config)
        return self._visited[config_id]

    def _get_all_reachable(self):
        return self._still_unreachable == 0

    def _get_visited_at_least_once(self):
        if self._stale_table:
            self._update_table()
        visited = self._configuration_table[['Visited from mode {}'
                                                 ''.format(i+1) for i in range(self._num_modes)]]
        visited = np.any(np.array(visited), axis=1)
        return visited

    def _all_visited(self):
        if self._stale_table:
            self._update_table()
        visited = self._get_visited_at_least_once()
        num_visited = np.sum(visited)
        num_total = len(self._configuration_table)
        self._percent_visited = num_visited/num_total*100
        return num_visited == num_total

    def get_unique_reachability_sizes(self):
        # We define the unique reachability set of a mode as the set of configuration that can
        # only be reached from within that mode while moving upwards.
        visit_columns = ['Visited from mode {}'.format(i+1) for i in range(self._num_modes)]
        visited = np.array(self._configuration_table[visit_columns])
        # Get the index of all configurations which can only be reach from a single mode
        idx = np.sum(visited, axis=1) == 1
        # And then count them for each mode.
        return np.array(self._configuration_table[idx][visit_columns].sum())

    def _update_table(self):
        self._stale_table = False
        visited = []
        quality = []
        parent = []
        reachable = []
        for config_id in self._configuration_table['Configuration ID']:
            visited.append(self._visited[config_id])
            quality.append(self._quality[config_id])
            parent.append(self._parent[config_id])
            reachable.append(self._reachable[config_id])
        self._configuration_table['Visited from mode {}'.format(self._num_modes)] = visited
        self._configuration_table['Quality from mode {}'.format(self._num_modes)] = quality
        self._configuration_table['Parent from mode {}'.format(self._num_modes)] = parent
        self._configuration_table['Reachable'] = reachable

    def get_smallest_unvisited_minima(self):
        # It can be proved, that if the entire configuration space is not reachable from the 
        # configuration with the smallest upper bound, then there does not exist any point 
        # with a reachability set that contains the entire configuration space. 
        # Hint: Prove by contradiction.
        visited = self._get_visited_at_least_once()
        df = self._configuration_table[~visited]
        indexer = df['Upper Bound'] == df['Upper Bound'].min()
        smallest = df[indexer].iloc[0]['Configuration']
        if self._num_modes == 1:
            self._global_minima = smallest
        return smallest
               
    def get_score(self, configuration, drop_probability=0):
        if self._random.rand() < drop_probability:
            return float('inf')
        else: 
            config_id = self.get_unique_configuration_identifier(configuration)
            return self._objective[config_id]

    def optimize_hyperparameters(self, config, drop_probability=0, iterations=10, update='Multi-Incumbent'):
        """optimize_hyperparameters
        
        A simplistic approach to optimizing the hyper-parameters of an
        algorithm. It merely performs a 1-dimensional optimization of each 
        parameter independently and then simultaneously updates all of them. 
        This process is embarassingly parallel and can be repeated until
        convergence.

        Parameters
        ----------
        config : Configuration
            The initial starting configuration for the search
        drop_probability : float
            A float in [0, 1). The probability that configurations are
            randomly dropped in each iteration of the method. If this were 
            applied with the evaluation of each configuration in parallel, it 
            may be optimal to end each iteration after a sufficiently large
            percentage of the configurations have been evaluated and then 
            to treat anything left un-evaluated as censored, rather than
            possibly blocking and waiting for a long time for those final few
            configurations to finish. By specifing the probability that a
            configuration will be dropped in each iteration we can simulate
            this effect. Note: We assume that the quality of each configuration
            is independent of its probability of being slowest and therefore
            dropped, which is probably an over-simplification.
        iterations : int
            A positive integer. The number of iterations of the method to
            perform.

        Returns
        -------
        anytime_performance : list of float
            The performance of the incumbent configuration at each iteration.
        anytime_incumbent : list of Configuration
            The list of anytime incumbents for each iteration.
        centroid_performance : list of float
            The performance of the centroid configuration at each iteration.
        anytime_centroid : list of Configuration
            The list of centroids used at each iteration.
       
        """
        centroid = config
        incumbent = copy.deepcopy(config)
        incumbent_score = self.get_score(config)

        anytime_performance = []
        anytime_incumbent = []
        centroid_performance = []
        anytime_centroid = []
        anytime_incumbent.append(incumbent)
        anytime_performance.append(incumbent_score)

        for i in range(iterations):
            score = self.get_score(centroid)
            anytime_centroid.append(centroid)
            centroid_performance.append(score)

            next_centroid = copy.deepcopy(centroid)
            # In practice this entire loop and the one within it would be parallelized
            # completely. 
            for hp in self.config_space.get_hyperparameters():
                if isinstance(hp, OrdinalHyperparameter):
                    values = list(hp.sequence)
                elif isinstance(hp, CategoricalHyperparameter):
                    # Any single categorical value may be swapped with another
                    values = list(hp.choices)
                else:
                    raise NotImplementedError('Only Categorical and Ordinal '
                                              'Hyperparameters are supported '
                                              'at the moment. '
                                              'Provided {}. '.format(type(hp)))
                best_score = float('inf')
                best = None
                # Break ties uniformly at random.
                self._random.shuffle(values)
                for value in values:
                    challenger = copy.deepcopy(centroid)
                    challenger[hp.name] = value
                    score = self.get_score(challenger, drop_probability)
                    # Check if this is better than the current best for this slice
                    if score <= best_score:
                        best_score = score
                        best = copy.deepcopy(challenger)
                        # Check if this is better than all confiugrations so far
                        if score <= incumbent_score:
                            incumbent_score = score
                            incumbent = challenger
                # Update the centroid
                next_centroid[hp.name] = best[hp.name]
            if update == 'Single-Incumbent':
                centroid = incumbent
            else:
                centroid = next_centroid                
            anytime_incumbent.append(incumbent)
            anytime_performance.append(incumbent_score)

        return  anytime_performance, anytime_incumbent, \
                centroid_performance, anytime_centroid

    def optimize_hyperparameters_randomly(self, config, drop_probability=0, iterations=10):
        """optimize_hyperparameters_randomly

        Performs the same amount of work as it's non-random counterpart, but
        instead of doing anything intelligent within each iteration it just 
        randomly samples configurations.
       
        Parameters
        ----------
        config : Configuration
            The initial starting configuration for the search
        drop_probability : float
            A float in [0, 1). The probability that configurations are
            randomly dropped in each iteration of the method. If this were 
            applied with the evaluation of each configuration in parallel, it 
            may be optimal to end each iteration after a sufficiently large
            percentage of the configurations have been evaluated and then 
            to treat anything left un-evaluated as censored, rather than
            possibly blocking and waiting for a long time for those final few
            configurations to finish. By specifing the probability that a
            configuration will be dropped in each iteration we can simulate
            this effect. Note: We assume that the quality of each configuration
            is independent of its probability of being slowest and therefore
            dropped, which is probably an over-simplification.
        iterations : int
            A positive integer. The number of iterations of the method to
            perform.

        Returns
        -------
        anytime_performance : list of float
            The performance of the incumbent configuration at each iteration.
        anytime_incumbent : list of Configuration
            The list of anytime incumbents for each iteration.       
        """
        incumbent = copy.deepcopy(config)
        incumbent_score = self.get_score(config)

        anytime_performance = []
        anytime_incumbent = []
        anytime_incumbent.append(incumbent)
        anytime_performance.append(incumbent_score)

        for i in range(iterations):
            # In practice this entire loop and the one within it would be parallelized
            # completely. 
            for hp in self.config_space.get_hyperparameters():
                if isinstance(hp, OrdinalHyperparameter):
                    values = list(hp.sequence)
                elif isinstance(hp, CategoricalHyperparameter):
                    # Any single categorical value may be swapped with another
                    values = list(hp.choices)
                else:
                    raise NotImplementedError('Only Categorical and Ordinal '
                                              'Hyperparameters are supported '
                                              'at the moment. '
                                              'Provided {}. '.format(type(hp)))
 
                for value in values:
                    challenger = self.config_space.sample_configuration()
                    score = self.get_score(challenger, drop_probability)
                    # Check if this is better than all confiugrations so far
                    if score <= incumbent_score:
                        incumbent_score = score
                        incumbent = challenger
            anytime_incumbent.append(incumbent)
            anytime_performance.append(incumbent_score)

        return  anytime_performance, anytime_incumbent

       
    def _embed_y(self, X, y_samples):
        """_embed_y

        Embeds the configurations into a matrix with dimensions equal to the 
        number of hyperparmaeters.

        Parameters
        ----------
        X : np.ndarray
            A 2D array containing all of the configurations, one configuration
            per row.
        y_samples : np.ndarray
            A 1 or 2D array containing samples of objective function 
            performance scores. Rows correspond to the configurations of X.
        
        Returns
        -------
        embedded_y : np.ndarray
            The configuration performance estimates embedded in the dimensions
            of the hyperparameter configurations. All numerical parameters are
            encoded such that adjacent values are adjacent in the embedding 
            space. The last dimensions of embedded_y contains the samples of y.
        value_lists : list of lists
            Each of the first n-1 dimensions of embedded_y correspond to a 
            particular hyperparameter. Each list in value_lists corresponds
            to the matching parameter, and the order of the parameter values
            as they appear in the embedded space is provided by the 
            corresponding list.
        dimensions : tuple of ints
            The shape of embedded_y, excluding the last dimension.
        """    
        X = np.array(copy.deepcopy(X))
        y = np.array(copy.deepcopy(y_samples))

        if len(y.shape) == 1:
            y = np.reshape(y, (len(y), 1))
         
        def add_columns(data, col_name):
            cols = []
            for i in range(data.shape[1]):
                df[col_name.format(i)] = data[:,i]  
                cols.append(col_name.format(i))
            return cols
        
        df = pd.DataFrame()
        x_cols = add_columns(X, 'X_{}')
        y_cols = add_columns(y, 'y_{}')
        
        df = df.sort_values(by=x_cols)
        X = np.array(df[x_cols])
        y = np.array(df[y_cols])
        dimensions = tuple(df[x_cols].nunique())
        # We now want to reshape y so that the index of each of the dimensions
        # (except the last one) provide the indices for the respective values
        # of the parameters. That is, the relative position of each of the 
        # y_samples is the same as if the y_values were in an n-dimesional
        # grid with unit length between each of their parameter values. 
        embedded_y = y.reshape(dimensions + (-1,))

        # The index of a set of samples in embedded_y indexes the parameter
        # values as found in value_lists
        value_lists = []
        for x_col in x_cols:
            values = sorted(list(set(list(df[x_col]))))
            value_lists.append(values)
        
        return embedded_y, value_lists, dimensions

    def _encode_categoricals_as_ints(self, y, value_lists, dimensions):
        """encode_categoricals_as_ints
 
        Since we're calculating partial derivatives with finite differences,
        we can generalize this to be applied to the categorical variables
        for all four-cycles of configurations. The only piece that is really
        missing is the natural notion of distance between two configurations
        when categorical variables are changing. We heuristically assign all
        such distances a value of 1. In practice, this doesn't actually affect
        our tests for significance anyways, since these distances end up just
        being constant factors that don't affect tests for significance of
        a difference from zero. However, it does change the absoluate 
        magnitude of the partial derivative values. 

        To encode categoricals as integers, we need to create multiple copies
        of each value of the categorical so that every combination of values
        appears adjacent to each other. For simplicity, we create all such
        pairs and then just concatenate together duplicate slices of the data
        for each set of pairs. This is a little bit less efficient than 
        strictly necessary, since we only actually care about the partial 
        values that are computed for every second pair (the rest will all be
        duplicates), but it allows for simple code design and hence reduces
        the risk of implementation errors.

        Parameters
        ----------
        y : numpy.ndarray
            The pre-embedded array of y value samples (see _embed_y)
        value_lists : list of lists
            The values that correspond to the dimensions of y
        dimensions : tuple
            The size of the dimensions of y (excluding the last dimension,
            which contains all of the samples of y).        

        Returns
        -------
        encoded_y : numpy.ndarray
            The objective function samples with the categorical axes
            encoded as integers
        value_lists : list of lists
            The updated list of values for each dimension, using the
            new encoding space for the categorical parameters.
        dimensions : tuple
            The new dimension sizes of the encoding space
        encodings : dict of list of tuples
            The encodings that were used for the categorical parameters
            The dict keys correspond to the axis of the categorical
            hyperparameters in the encoding space. The list of tuples
            is a list of pairs, where each pair corresponds to one of the
            pairs of categorical values with which we are interested in
            computing the partial derivative. The pairs contain the 
            original values of the categorical parameter. The position
            of the values in the list of tuples when flattened corresponds
            to the position of those values in the encoding space. 
        """
        # So that we can modify it
        dimensions = list(dimensions)        
        # The map that let's use get back the original encoding space
        encodings = {}

        for axis in range(len(self.hyperparameters)):
            hp_name = self.hyperparameters[axis]
            hp = self.config_space.get_hyperparameter(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                choices = hp.choices
                positions = list(range(len(choices)))
                value_pairs = list(itertools.combinations(choices, 2))
                position_pairs = list(itertools.combinations(positions, 2))
                # Flatten
                position_pairs = [p for pair in position_pairs for p in pair]
                # Create the duplicates along the axis
                y = y.take(position_pairs, axis=axis)
                # To get a distance of 1 between each categorical encoding we
                # replace the "values" with a list of integers
                value_lists[axis] = list(range(len(position_pairs)))
                # Update the size of this dimension
                dimensions[axis] = len(position_pairs)
                encodings[axis] = value_pairs
        dimensions = tuple(dimensions)

        return y, value_lists, dimensions, encodings

    def _decode_categoricals(self, interactions, encodings):
        """_decode_categoricals

        Removes the duplicates of the categorical partial derivatives that
        were by-products of encoding the categorical parameters as if they
        are integers.
  
        Parameters
        ----------
        interactions : dict
            A dict mapping tuples of hyperparameters indicies to grids of 
            partial derivatives with respect to the hyperparameter indicies.
        encodings : dict of list of tuples
            The dict keys correspond to the axis of the categorical
            hyperparameters in the encoding space. The list of tuples
            is a list of pairs, where each pair corresponds to one of the
            pairs of categorical values with which we are interested in
            computing the partial derivative. The pairs contain the 
            original values of the categorical parameter. The position
            of the values in the list of tuples when flattened corresponds
            to the position of those values in the encoding space. 

        Returns
        -------
        interactions : dict
            A dict mapping tuples of hyperparameter indices to grids of
            partial derivatives with respect to the hyperparameter indicies.
        """
        def decode_categorical(interaction, p):
            n = interaction.shape[p]
            interaction = np.take(interaction, range(0, n, 2), axis=p)
            return interaction

        for hps in interactions:
            interaction = interactions[hps]
            for hp in hps:
                if hp in encodings:
                    interaction = decode_categorical(interaction, hp)
            interactions[hps] = interaction

        return interactions
            
    def test_reject_parameter_independence(self, order=2, distance_type='graphical', exclude=0):
        """test_reject_parameter_independence

        Calculates the percentage of the landscape for which tuples of 
        hyperparameters of length order are independent from each other.
    
        Technically, this is achieved by approximating the partial derivatives
        with respect to the tuples of hyperparameters using the method of 
        finite differences. This is done for each sample value of the objective
        function of the landscape. A two-sided t-test is performed to test for
        a significant difference from zero for each such set of partial 
        derivative estimates. 

        Note that this can still be used for order=1, in which case the 
        'independence' is bet thought of as the fraction of the landscape
        for which the value of the objective function is independent from 
        the individual hyperparameter values.

        Parameters
        ----------
        order : int
            A positive integer that encodes the degree of the interactions
            which will be returned.
        distance_type : str
            Choose what kind of distance space is used to calculate distances.
            Options are 'cartesian', which normalizes the numerical parameters
            to the range [0,1] and then uses these values as the distances; and
            'graphical', which counts the minimal number of steps that would 
            need to be taken in a neighbourhood graph of the landscape.
        exlude : float
            The percentage of the landscape which you are excluding from the
            analysis. Must be a number in [0,100). The worst exclude% of the
            configuration space will be omitted from the analysis. If any one
            configuration that is needed to calculate a partial derivative is
            excluded, we will exclude that partial derivative.

        Returns
        -------
        percent_dependent : dict of tuples to floats
            A dict containing tuples of hyperparamater names and the
            corresponding percentage of the landscape for which the two
            parameters are dependent.
        total_percent_dependent : float
            The total percentage of the partial derivatives that are dependent
            in the landscape.
        partial_intervals : dict of tuples to np.ndarrays
            A dict containing tuples of hyperparameter names and arrays 
            containing the upper and lower bounds on the partial derivatives.
        """
        # Stuff I removed:
        """
        absolute_partials : dict of tuples to floats
            The mean absolute value of the partial derivatives for each tuple
            of hyperparameters.
        mean_absolute_partial : float
            The mean absolute value of all of the partial derivatives
        """

        #def get_mean_abs_partial(interaction):
        #    samples = interaction.reshape((-1))
        #    return np.mean(np.abs(samples))

        def get_interaction_intervals(interaction):
            samples = interaction.reshape((-1, interaction.shape[-1]))
            intervals = helper.confidence_interval_student(samples, self.confidence_level)
            return intervals

        interactions = self._calculate_partials(order=order, distance_type=distance_type, exclude=exclude)
        percent_dependent = {}
        #absolute_partials = {}
        partial_intervals = {}
        total_percent_dependent = 0
        total_n_samples = 0
        #mean_absolute_partial = 0
        for hps in interactions:
            parsed_hps = tuple([self.hyperparameters[hp] for hp in hps])
            dependent, n_samples = self._get_percent_dependent(interactions[hps])
            #current_absolute_partial = get_mean_abs_partial(interactions[hps])
            intervals = get_interaction_intervals(interactions[hps])
            percent_dependent[parsed_hps] = dependent
            #absolute_partials[parsed_hps] = current_absolute_partial
            partial_intervals[parsed_hps] = intervals
            total_percent_dependent += dependent
            #mean_absolute_partial += current_absolute_partial
            total_n_samples += 1

        return percent_dependent, total_percent_dependent/total_n_samples, \
               partial_intervals
               #absolute_partials, mean_absolute_partial/total_n_samples, \

    def _get_percent_dependent(self, interaction):
        samples = interaction.reshape((-1, interaction.shape[-1]))
        # nans may appear because we excluded part of the landscape by 
        # replacing their values with nans, now we just want to remove them
        samples = samples[np.where(np.all(~np.isnan(samples), axis=1))[0]]
        pvalues = scipy.stats.ttest_1samp(samples, 0, axis=1)[1]
        # Note that scipy.stats.ttest_1samp returns nan values if the
        # a sample contains nothing but zeros. In this case, all evidence
        # clearly points to the fact that the parameters are independent,
        # so that's what we want our output to be. 
        pvalues = np.nan_to_num(pvalues, nan=1)
        dependent = pvalues <= 1 - self.confidence_level
        return np.mean(dependent)*100, len(dependent)

    def check_partial_objective_correlation(self, order=2, distance_type='graphical', exclude=0):
        """test_partial_objective_correlation

        Calculates the rank correlation between the absolute value of the 
        partial derivative values and objective function values.
    
        Technically, this is achieved by approximating the partial derivatives
        with respect to the tuples of hyperparameters using the method of 
        finite differences. This is done for each sample value of the objective
        function of the landscape, and then we take the mean of these values.

        We also approximate the objective function values at the locations of
        the partial derivatives using linear interpolation.
 
        Parameters
        ----------
        order : int
            A positive integer that encodes the degree of the interactions
            which will be returned.
        distance_type : str
            Choose what kind of distance space is used to calculate distances.
            Options are 'cartesian', which normalizes the numerical parameters
            to the range [0,1] and then uses these values as the distances; and
            'graphical', which counts the minimal number of steps that would 
            need to be taken in a neighbourhood graph of the landscape.
        exlude : float
            The percentage of the landscape which you are excluding from the
            analysis. Must be a number in [0,100). The worst exclude% of the
            configuration space will be omitted from the analysis. If any one
            configuration that is needed to calculate a partial derivative is
            excluded, we will exclude that partial derivative.

        Returns
        -------
        correlations : dict of tuples to spearmanr results
            The mean result from applying spearman's test for rank correlation
        raw_correlations : dict of tuples to np.ndarray
            The array of results from applying spearman's test for rank 
            correlation to each slice of the tuple of hyper-parameters.
        mean_abs_partials : dict of tuples to list
            A dict containing tuples of hyperparameter names and arrays
            containing the mean estimate for the partial derivatives.
        mean_objectives : dict of tuples to list
            A dict contianing tuples of hyperparameter names and arrays 
            containing the estimate for the objective function values at
            the location of the partial derivatives.
        """
        def reshape(data, axes):
            """reshape

            This function takes the axes of data, keeps them,
            and unravels everything else. Finally, it unravels
            the important axes into the second dimension. This 
            creates a 2D matrix, such that a row corresponds to
            a single instantiation of the values of all of the
            axes not in axes, and the values in the columns 
            of a particular row correspond to the values in
            the axes.
            """
            # Move all of the important axes to the end
            axes = sorted(list(axes), reverse=True)
            newshape = []
            for i in range(0, len(axes)):
                newshape.append(data.shape[axes[i]])
                data = data.swapaxes(axes[i],-1-i)
            newshape.append(-1)
            newshape = tuple(newshape[::-1])
            # Unravel all instantiations of all other axes
            data = data.reshape(newshape)
            data = data.reshape((data.shape[0], -1))
            return data
        def get_mean_abs_partials(interaction, axes):
            samples = np.mean(interaction, axis=-1)
            samples = np.abs(samples)
            samples = reshape(samples, axes)
            return samples
        def reshape_objectives(objective, axes):
            #st.write(objective.shape)
            samples = np.mean(objective, axis=-1)
            #st.write(samples.shape)
            samples = reshape(samples, axes)
            #st.write(samples.shape)
            return samples
        def spearmanr_n(data1, data2):
            """spearmanr_n

            apply spearmanr n times, once for each row of
            data1 and data2
            """
            data = np.append(data1, data2, axis=1)
            if np.isnan(data).any():
                st.write(data)
            def spearman_helper(data):
                n = len(data)
                data1 = data[:int(n/2)]
                data2 = data[int(n/2):]
                #if np.isnan(data).all():
                #    st.write(data1)
                #    st.write(data2)
                r = spearmanr(data1, data2, nan_policy='omit')[0]
                #if np.isnan(r):
                #    st.write(data1)
                #    st.write(data2)
                return r
            return np.apply_along_axis(spearman_helper,
                                       1,
                                       data)
                

        interactions = self._calculate_partials(order=order, distance_type=distance_type, exclude=exclude)
        objectives = self._estimate_objective_at_partials(order=order, exclude=exclude)

        correlations = {}
        raw_correlations = {}
        mean_abs_partials = {}
        mean_objectives = {}
        #mean_absolute_partial = 0
        for hps in interactions:
            parsed_hps = tuple([self.hyperparameters[hp] for hp in hps])
            mean_abs_partial = get_mean_abs_partials(interactions[hps], hps)
            mean_objective = reshape_objectives(objectives[hps], hps)
            st.write(parsed_hps)
            rs= spearmanr_n(mean_abs_partial, mean_objective)
            correlations[parsed_hps] = np.nanmean(rs)
            raw_correlations[parsed_hps] = rs
            mean_abs_partials[parsed_hps] = mean_abs_partial
            mean_objectives[parsed_hps] = mean_objective

        return correlations, raw_correlations, mean_abs_partials, mean_objectives
       
    
    def _calculate_partials(self, order=2, distance_type='graphical', exclude=0):
        """_calculate_partials

        Calculculates the partial derivatives for all combinations of 
        the hyperparameters with order unique hyperparameters in each tuple.

        Parameters
        ----------
        order : int
            The degree of the partial derivatives to calculate. If 
            order=1 then first order partial derivatives are calculated, if
            order=2 then second order partial derivatives are calculated, etc.
        distance_type : str
            Choose what kind of distance space is used to calculate distances.
            Options are 'cartesian', which normalizes the numerical parameters
            to the range [0,1] and then uses these values as the distances; and
            'graphical', which counts the minimal number of steps that would 
            need to be taken in a neighbourhood graph of the landscape.
        exclude : float
            The percentage of the landscape which you want to exclude from the
            analysis. Should be a number in [0, 100). We exclude all
            configurations in the top exclude% of solution qualities by
            replacing them with nans in the partial calculations. In this way,
            you can focus the calculations on more interesting parts of the 
            landscape.

        Returns
        -------
        partials : dict of tuples to np.ndarray(dtype=float)
            A dict mapping tuples of hyperparameter indices to numpy arrays
            containing the approximate partial derivatives with respect to
            the tuple of hyperparameter values, calculated using the method
            of finite differences.
        """
        X = self._encode_numericals(self.X, distance_type)
        # Encode the y samples as a percentage of the total variation in the
        # objective function values.
        y_samples = copy.deepcopy(self.y_samples)
        # Exclude the worst exclude% of the landscape from the analysis
        worst = self.y > np.percentile(self.y, 100-exclude)
        y_samples[worst] = np.nan
        max_y = np.nanmax(self.y)
        min_y = np.nanmin(self.y)
        y_samples -= min_y
        y_samples /= max_y - min_y
        y_samples *= 100
        # Embed the objective function samples in a grid
        y, value_lists, dimensions = self._embed_y(X, y_samples)
        # Encode categorical parameters as integers
        y, value_lists, dimensions, encodings \
            = self._encode_categoricals_as_ints(y, value_lists, dimensions)

        hyperparameters = copy.deepcopy(self.hyperparameters)

        def divide_along_axis(dividend, divisor, axis):
            # Let's avoid mutating the input for simplicity.
            dividend = copy.deepcopy(dividend)
            # In order to divide along the axis we need to swap the
            # axis along which we are dividing with the last axis, so that
            # the divisor gets broadcast properly
            dividend = dividend.swapaxes(axis, -1)
            # do the math
            quotient = dividend / divisor
            # Restore the original axis ordering.
            return quotient.swapaxes(axis, -1)
            
        def del_y_del_p(y, p, value_lists):
            """del_y_del_p
            
            Approximates the partial derivative of y with respect to 
            dimension p using the method of finite differences.
            """
            # Calculate the difference between each adjacent value along the
            # dimension p
            y_p_upper = y.take(range(1, dimensions[p]), axis=p)
            y_p_lower = y.take(range(0, dimensions[p]-1), axis=p)
            y_p_diff = y_p_upper - y_p_lower
            # Calculate the difference in p1 values
            p_upper = np.array(value_lists[p][1:])
            p_lower = np.array(value_lists[p][:-1])
            p_diff = p_upper - p_lower
            # Now we divide the first differences by the second ones,
            # which gives us the partial derivative with respect to p1
            # approximated using the method of finite differences.
            return divide_along_axis(y_p_diff, p_diff, p)
            
        interactions = {}
        # For each pair of parameters
        dim_tuples = itertools.combinations(range(len(y.shape)-1), order)
        for hps in dim_tuples:
            # Make sure axes are sorted in reverse order so that we don't
            # have to worry about the index of the axes changing.
            hps = tuple(sorted(hps, reverse=True))
            # Keep taking partial derivatives along the specified dimensions
            # until we are done.
            del_y_del_hps = copy.deepcopy(y)
            for hp in hps:
                del_y_del_hps = del_y_del_p(del_y_del_hps, hp, value_lists)
            interactions[hps]  = del_y_del_hps

        interactions = self._decode_categoricals(interactions, encodings)

        return interactions

    def _estimate_objective_at_partials(self, order=2, exclude=0):
        """_calculate_partials

        Uses linear interpolation to estimate the objective function value at
        the locations where we calculate the partial derivatives. 

        Parameters
        ----------
        order : int
            The degree of the partial derivatives for which you are calculating
            the estimates of the objective function
        exclude : float
            The percentage of the landscape which you want to exclude from the
            analysis. Should be a number in [0, 100). We exclude all
            configurations in the top exclude% of solution qualities by
            replacing them with nans in the partial calculations. In this way,
            you can focus the calculations on more interesting parts of the 
            landscape.

        Returns
        -------
        partials_objectives : dict of tuples to np.ndarray(dtype=float)
            A dict mapping tuples of hyperparameter indices to numpy arrays
            containing the approximate objective function values at the 
            locations for which we would calculate partial derivatives. The
            shape of the array will match the shape of the partial derivative
            array.
        """
        X = self._encode_numericals(self.X)
        # Encode the y samples as a percentage of the total variation in the
        # objective function values.
        y_samples = copy.deepcopy(self.y_samples)
        # Exclude the worst exclude% of the landscape from the analysis
        worst = self.y > np.percentile(self.y, 100-exclude)
        y_samples[worst] = np.nan
        max_y = np.nanmax(self.y)
        min_y = np.nanmin(self.y)
        y_samples -= min_y
        y_samples /= max_y - min_y
        y_samples *= 100
        # Embed the objective function samples in a grid
        y, value_lists, dimensions = self._embed_y(X, y_samples)
        # Encode categorical parameters as integers
        y, value_lists, dimensions, encodings \
            = self._encode_categoricals_as_ints(y, value_lists, dimensions)

        hyperparameters = copy.deepcopy(self.hyperparameters)
            
        def del_y_del_p(y, p, value_lists):
            """del_y_del_p
           
            Calculates the mean objective function value at the locations
            for which we calculate partial derivatives of y with respect to
            dimension p using the method of (centered) finite differences. 
            """
            # Calculate the mean between each adjacent value along the
            # dimension p
            y_p_upper = y.take(range(1, dimensions[p]), axis=p)
            y_p_lower = y.take(range(0, dimensions[p]-1), axis=p)
            y_p_mean = (y_p_upper + y_p_lower)/2
            return y_p_mean
            
        interactions = {}
        # For each pair of parameters
        dim_tuples = itertools.combinations(range(len(y.shape)-1), order)
        for hps in dim_tuples:
            # Make sure axes are sorted in reverse order so that we don't
            # have to worry about the index of the axes changing.
            hps = tuple(sorted(hps, reverse=True))
            # Keep taking partial derivatives along the specified dimensions
            # until we are done.
            del_y_del_hps = copy.deepcopy(y)
            for hp in hps:
                del_y_del_hps = del_y_del_p(del_y_del_hps, hp, value_lists)
            interactions[hps]  = del_y_del_hps

        interactions = self._decode_categoricals(interactions, encodings)

        return interactions

    def test_reject_convexity(self, fix=[], rtol=1e-5, atol=1e-8, max_points=None):
        """test_reject_convexity
    
        Checks to see if the numerical parameters can be rejected as not being
        convex.
        
        Parameters
        ----------
        fix: list of str
            A list of hyperparameter names to fix to their optimal values
            before performing the test for convexity.
        rtol : float
            Relative tolerance. A lower bound is considered co-planar to the
            convex hull instead of interior if it is this close to the edge
            of the hull along the metric dimension.
        atol : float
            Absolute tolerance. See rtol.
        max_points : int | float | None
            If not None, only this many randomly selected points will be tested
            to see if they are interior to the convex hull. If an integer, then
            up to that many points will be evaluated. If a float, then it must
            be between 0 and 1, and that fraction of the points will be
            evaluated. 

        Returns
        ------
        float:
            The percentage of the lower-bounds that are interior to the convex
            hull. If non-zero, this landscape is not convex.
        float:
            The percentage of the lower-bounds that were co-planar with the
            convex hull within numerical stability, as defined by rtol and atol.
            If large, the outcome of the test should be considered unreliable
            due to issues with numerical precision. 
        """

        def fix_hyperparameters(X, y_lower, y_upper, y, fix, hyperparameters):
            # Make sure that all categorical hyperparameters are fixed.
            for hp in self.config_space.get_hyperparameters():
                if isinstance(hp, CategoricalHyperparameter):
                    if hp.name not in fix:
                        fix.append(hp.name)
            # Get the optimal configuration
            minimizer = X[np.argmin(y)]
            select = np.full(len(X), True)
            select_indices = [int(round(i)) for i in range(0, X.shape[1])]
            for hp in fix:
                idx = hyperparameters.index(hp)
                # Only take those configurations with hp equal to the value
                # of the optimal configuration
                select = np.logical_and(X[:,idx] == minimizer[idx], select)
                select_indices.remove(idx)
            X = X[select,:]
            X = X[:,select_indices]
            return X, y_lower[select], y_upper[select]
     
        X, y_lower, y_upper = fix_hyperparameters(self.X, 
                                                  self.y_bounds[:,0],
                                                  self.y_bounds[:,1],
                                                  self.y, 
                                                  fix, 
                                                  self.hyperparameters)

        return self._test_reject_slice(X, y_lower, y_upper, rtol, atol, max_points)

    def _test_reject_slice(self, X, y_lower, y_upper, rtol, atol, max_points):
        if max_points is None:
            max_points = len(X)
        if max_points <= 1:
            max_points = int(len(X)*max_points)
        max_points = min(len(X), max_points)
        # First we need to append the maximum y-value to each corner of X. 
        # Get the extreme values for X
        max_x = np.max(X, axis=0)
        min_x = np.min(X, axis=0) 
        # and enumerate all corners
        corners = itertools.product(*list(zip(min_x, max_x)))
        corners = np.array(list(corners))
        # Add the corners to X as new points
        X_hull = np.append(X, corners, axis=0)
        # Now add the max value of y for these corners
        max_y = np.max(y_upper)
        y_hull = np.append(y_upper, [max_y]*len(corners))
        # And append y to X
        y_hull = y_hull.reshape((len(y_hull), 1))
        points_upper = np.append(X_hull, y_hull, axis=1)
        # Make sure each of the lower bounds are outside of the hull
        points_lower = np.append(X, np.reshape(y_lower, (len(y_lower), 1)), axis=1)
        np.random.shuffle(points_lower)

        # Idea to use linear programming from: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
        (n_points, n_dim) = points_upper.shape
        c = np.zeros(n_points)
        A = np.r_[points_upper.T,np.ones((1,n_points))]
        interior = 0
        coplanar = 0
        progress = Progress(max_points,
                            '{}% done testing slice for convexity')
        for idx, x in enumerate(points_lower[:max_points]):
            b = np.r_[x, np.ones(1)]
            lp1 = linprog(c, A_eq=A, b_eq=b)
            if lp1.success:
                # Slightly decrease the lower bound by the tolerance to
                # verify that it is still interior, if it is, we consider
                # it to be co-planar instead of interior
                b[-2] -= atol + rtol*np.abs(b[-2])
                lp2 = linprog(c, A_eq=A, b_eq=b)
                if lp2.success:
                    interior += 1
                else:
                    coplanar += 1
            progress.update()
            #print(f'Done {idx+1}/{max_points} (Actual number of poitns: {len(points_lower)})')
        return interior/max_points*100, coplanar/max_points*100

    def _test_reject_slice_slow_and_unstable(X, y_lower, y_upper):
        # First we need to append the maximum y-value to each corner of X. 
        # Get the extreme values for X
        max_x = np.max(X, axis=0)
        min_x = np.min(X, axis=0) 
        # and enumerate all corners
        corners = itertools.product(*list(zip(min_x, max_x)))
        corners = np.array(list(corners))
        # Add the corners to X as new points
        X_hull = np.append(X, corners, axis=0)
        # Now add the max value of y for these corners
        max_y = np.max(y_upper)
        y_hull = np.append(y_upper, [max_y]*len(corners))
        # And append y to X
        y_hull = y_hull.reshape((len(y_hull), 1))
        points_upper = np.append(X_hull, y_hull, axis=1)
        # Get the convex hull on the upper bound points
        hull = ConvexHull(points_upper)
        # Keep the X coordinates of only those points that are in the
        # convex hull.
        hull_points = points_upper[hull.vertices,:]
        hull_point_list = hull_points[:,:-1].tolist()
        # Make sure each of the lower bounds are outside of the hull
        points_lower = np.append(X, np.reshape(y_lower, (len(y_lower), 1)), axis=1)
        nonconvex_points = 0
        coplanar_points = 0
        checked_points = 0
        skipped = 0
        progress = Progress(max_points,
                            '{}% done testing slice for convexity')
        self._random.shuffle(points_lower)
        for point in points_lower:
            if list(point[:-1]) in hull_point_list:
                # We know that this point cannot be in the convex hull 
                # because the upper bound corresponding to this lower bound
                # is in the hull.
                skipped += 1
                #print("Skipping {}".format(skipped))
            else:
                new_points = copy.deepcopy(hull_points)
                new_points = np.append(new_points, [point], axis=0)
                new_point_idx = len(new_points)-1
                # Set 'Qc' option so that co-planar points are checked for
                new_hull = ConvexHull(new_points, qhull_options='Qc')
                # Check if the index of the lower bound is not in the list of vertices
                # of the convex hull and not in the list of coplanar points (i.e.,
                # points on the boundary of the hull).
                if (new_point_idx not in new_hull.vertices 
                    and new_point_idx not in new_hull.coplanar[:,0]):
                    # This point is in the interior of the hull, therefore it is 
                    # a lower bound that is above the convex hull of the upper bounds,
                    # and so that means that the landscape is not uni-modal. Hence,
                    # we reject uni-modality.
                    nonconvex_points += 1
                elif new_point_idx in new_hull.coplanar[:,0]:
                    coplanar_points += 1
                    #print(point)
            checked_points += 1
            #print('{}/{} points checked so far are non-convex'
            #      ''.format(nonconvex_points, checked_points))
            progress.update()
        return nonconvex_points/len(points_lower)*100, coplanar_points/len(points_lower)*100, hull
 
    def fdc(self, X=None, y=None, argmin='closest', distance_type='graphical', norm=1.0):
        """fdc

        Calculates the fitness distance correlation on the landscape defined by
        X and y. 

        Parameters
        ----------
        X : np.ndarray | None
            The configurations that have been evaluated. If None, we use
            self.X.
        y : np.ndarray | None
            The observed objective function values for the configurations. If 
            None, we used self.y
        argmin : str
            Choose how to handle the case with multiple equally optimal 
            configurations. Options are 'first', which treats the first as if
            it is the only one; 'random', which picks a random one and
            treats it as if it is the only one; and 'closest', which uses
            the distance to the closest optimal configuration.
        distance_type : str
            Choose what kind of distance space is used to calculate distances.
            Options are 'cartesian', which normalizes the numerical parameters
            to the range [0,1] and then uses these values as the distances; and
            'graphical', which counts the minimal number of steps that would 
            need to be taken in a neighbourhood graph of the landscape.
        norm : float
            Specifies what kind of norm is used when calculating distances.
            E.g., the l2-norm.

        Returns
        -------
        fdc : float
            The fitness distance correlation
        distances : np.array
            The distances to the data from the argmin(s)
        fitnesses : np.array
            The fitness of the data
        """
        def covariance(a, b):
            return np.dot(a-np.mean(a), b-np.mean(b))/(len(a) - 1)

        def get_distances(X_num, X_cat, minimizers, argmin, norm):
            distances = []
            if argmin == 'first':
                minimizers = [minimizers[0]]
            elif argmin == 'random':
                minimizers = [minimizers[np.random.randint(0, len(minimizers))]]
            for minimizer in minimizers:
                X_num_d = X_num - X_num[minimizer,:]
                X_cat_d = np.array(X_cat != X_cat[minimizer,:], dtype=float)
                # Recombine into a distance array
                X_d = np.array(np.append(X_num_d, X_cat_d, axis=1), dtype=float)
                # take the norm to get the distances
                distances.append(np.linalg.norm(X_d, axis=1, ord=norm))
            return np.min(distances, axis=0)

        if X is None or y is None:
            X = self.X
            y = self.y
        # Check if there is more than one configuration with optimal
        # performance
        minimizers = np.where(y == np.min(y))[0]
        if len(minimizers) > 1:
            print('There are {} optimal configurations!'.format(len(minimizers)))
            st.write('There are {} optimal configurations!'.format(len(minimizers)))
        # Get the indices of the categorical parameters
        categoricals = []
        # And the ranges of the numerical ones
        numerical_ranges = []
        numerical_values = []
        for hp in self.hyperparameters:
            hp = self.config_space.get_hyperparameter(hp)
            if isinstance(hp, CategoricalHyperparameter):
                 categoricals.append(True)
            else:
                 categoricals.append(False)
                 numerical_ranges.append([hp.sequence[0], hp.sequence[-1]])
                 numerical_values.append(list(hp.sequence))
        categoricals = np.array(categoricals)
        numerical_ranges = np.array(numerical_ranges)
        # Split X into categorical and numerical
        X_num = np.array(X[:,~categoricals], dtype=float)
        X_cat = X[:,categoricals]
        # Normalize the numericals
        if distance_type == 'cartesian':
            X_num -= numerical_ranges[:,0]
            X_num /= numerical_ranges[:,1] - numerical_ranges[:,0]
        else:
            for hp in range(X_num.shape[1]):
                values = numerical_values[hp]
                same = []
                for i in range(len(values)):
                    # Replace all values with their corresponding index
                    # in the ordinal sequence of values.
                    same.append(np.isclose(X_num[:,hp], values[i]))
                for i in range(len(values)):
                    X_num[same[i], hp] = i
        # Get the distances for each type of parameter
        distances = get_distances(X_num, X_cat, minimizers, argmin, norm)
        fitnesses = y
        cov = covariance(fitnesses, distances)
        sigf = np.sqrt(covariance(fitnesses, fitnesses))
        sigd = np.sqrt(covariance(distances, distances))
        return cov/(sigf*sigd), distances, fitnesses

    def init_fanova(self, max_configurations=1000):
        """init_fanova

        Initialiazes fanova by converting the landscape into the format
        required by fanova. Note that at the time of this writing fanova was
        poorly documented and determining how to use it correctly was 
        difficult. However, I beleive I have correclty determined how to use 
        it.

        NOTE: This is not the version of fANOVA used in Pushak & Hoos 2022a.
        
        Parameters
        ----------
        max_configurations : int
            A positive integer. fANOVA appears to run extremely slowly for
            large numbers of configurations. It is unclear why this is. 
            However, to make it feasable to run on many of these scenarios
            we therefore need to subsample the landscape prior to running
            fanova.
        """
        if len(self.y) > max_configurations:
            indices = np.arange(0, len(self.y))
            self._random.shuffle(indices)
            indices = indices[:max_configurations]
            X = self.X[indices,:]
            y = self.y[indices]
        else:
            X = self.X
            y = self.y
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
        # Make a new configuration space that is compatible with fANOVA
        config_space = ConfigSpace.ConfigurationSpace()
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, OrdinalHyperparameter):
                new_hp = UniformFloatHyperparameter(hp.name, min(hp.sequence), max(hp.sequence))
                config_space.add_hyperparameter(new_hp)
            else:
                new_hp = CategoricalHyperparameter(hp.name, [float(i) for i in range(len(hp.choices))])
                hp_ind = self.hyperparameters.index(hp.name)
                # While categorical parameters are supported (technically) I believe
                # they have never been tested by the authors of the code, because they
                # cause pyrfr to crash if they are strings. Furthermore, fanova logs
                # a warning that all values of X and y are expected to be floats...
                # So I guess we have to encode categoricals as numericals? I wonder 
                # whether or not these actually end up being handled appropriately by
                # the random forest...
                # Note that for the scenarios we studied in our paper it doesn't matter,
                # since our categorical parameters have only two values each, which means
                # that even if it learns an arbitrary rule 
                # (e.g., activiation_fn_1 < 0.2721), it will be functionally equivalent
                # to a meaningful rule (e.g., activation_fn_1 == tanh)
                for i in range(len(hp.choices)):
                    replace = X[:, hp_ind] == hp.choices[i]
                    X[replace, hp_ind] = float(i)
                config_space.add_hyperparameter(new_hp)
        # Sort the elements of X by the order of the hyperparameters in the
        # (annoyingly) un-sorted config_space
        hp_list = []
        for hp in self.hyperparameters:
            hp_list.append(hp)
        sort = [hp_list.index(hp.name) for hp in config_space.get_hyperparameters()]
        X = np.take(X, sort, axis=1)
        # print(config_space)
        # It crashes for some scenarios without this explicit conversion (if it can be done with no problems)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.fANOVA = fANOVA(X, y, config_space=config_space)

    def _quantify_fanova_importance(self, *args, **kwargs):
        """_quantify_fanova_importance
    
        Provides an interface to fanova's quantify_importance function. If this
        is the first time a call to this function is made then the fANOVA object
        will be initialized.

        At the time that this was created, fANOVA was poorly documented, and it 
        was somewhat difficult to discover the correct way to interface with it.
        I believe I have now done so, this script should do all of the things
        necesarry to convert your data into the correct format.

        Parameters
        ----------
        *args : any
            Arguments passed to fanova's quanitfy_importance function.
        **kwargs : any
            Keyword arguments passed to fanova's quantify_importance function.

        Returns
        -------
        output : any
            Output from fanova's quantify_importance function.
        """
        if self.fANOVA is None:
            self.init_fanova()

        return self.fANOVA.quantify_importance(*args, **kwargs)

    def fanova_importance(self, order=2):
        """fanova_importance

        Returns the importance of all tuples of hyperparameters of length order
        according to fANOVA.

        NOTE: This is not the fANOVA implemented used in Pushak & Hoos 2022a.

        Parameter
        ---------
        order : int
            The length of the tuples of hyperparameters evaluated. Determines
            the order of the parameter interactions quanitified.

        Returns
        -------
        importance : dict of tuples to floats
            A dict that maps tuples of hyperparameters to their importances
            according to fANOVA. The importance value returned is the 
            'individual importance', to use the terminology from fANOVA,
            multiplied by 100.
        total_importance : float
            The sum of all of the importance scores. This represents the 
            total fraction of the variance of the objective function that can
            be explained by parameter interactions of order order.
        """
        importance = {}
        total_importance = 0
        for hps in itertools.combinations(self.hyperparameters, order):
            importance[hps] = self._quantify_fanova_importance(hps)[hps]['individual importance']*100
            total_importance += importance[hps]

        return importance, total_importance

    def _encode_numericals(self, X, distance_type='graphical'):
        """_encode_numericals

        Encodes the numerical parameters as either Cartesian values that are
        normalized to be in the ranges [0,1] based on their original values,
        or encodes them as integers based on their position in the sequence
        of values. The latter is useful for calculate graphical-type distances
        between the values.

        Parameters
        ----------
        X : np.ndarray
            The configurations to encode
        distance_type : str
            Choose what kind of distance space is used to calculate distances.
            Options are 'cartesian', which normalizes the numerical parameters
            to the range [0,1] and then uses these values as the distances; and
            'graphical', which counts the minimal number of steps that would 
            need to be taken in a neighbourhood graph of the landscape.
        
        Returns
        -------
        X_encoded : np.ndarray
            The configurations with the numerical parameters encoded according
            to distance_type.
        """
        X = copy.deepcopy(X)
        # Get the indices of the categorical parameters
        categoricals = []
        # And the ranges of the numerical ones
        numerical_ranges = []
        numerical_values = []
        for hp in self.hyperparameters:
            hp = self.config_space.get_hyperparameter(hp)
            if isinstance(hp, CategoricalHyperparameter):
                 categoricals.append(True)
            else:
                 categoricals.append(False)
                 numerical_ranges.append([hp.sequence[0], hp.sequence[-1]])
                 numerical_values.append(list(hp.sequence))
        categoricals = np.array(categoricals)
        numerical_ranges = np.array(numerical_ranges)
        # Split X into categorical and numerical
        X_num = np.array(X[:,~categoricals], dtype=float)
        # Normalize the numericals
        if distance_type == 'cartesian':
            X_num -= numerical_ranges[:,0]
            X_num /= numerical_ranges[:,1] - numerical_ranges[:,0]
        else:
            for hp in range(X_num.shape[1]):
                values = numerical_values[hp]
                same = []
                for i in range(len(values)):
                    # Replace all values with their corresponding index
                    # in the ordinal sequence of values.
                    same.append(np.isclose(X_num[:,hp], values[i]))
                for i in range(len(values)):
                    X_num[same[i], hp] = i
        X[:,~categoricals] = X_num
        return X

    def exact_fanova_importance(self, order=1, samples=False):
        """exact_fanova_importance

        A re-implementation of fANOVA that runs directly on the grid instead
        of on a random forest.

        Parameters
        ----------
        order : int or list of int
            The order or list of orders of the interaction importances to 
            compute. Note: all lower order interactions will be calculated
            in the process of calculating the maximum order interactions, so
            the only reason to avoid inputing order=list(range(max_order)) is
            for your own convenience when you parse the output.
        samples : bool
            If samples is True, we calculate the fANOVA score for each sample
            of the objective function value and return an array of importances
            for each tuple of hyperparameters. Otherwise, we calculate the 
            fANOVA score for the best point estimate of the objective function
            values.

        Returns
        -------
        importances : dict of tuples to floats
            The importances of each tuple of hyperparameters.
        """
        def get_marginal(hp_subset):
            # Copy y so that we don't modify it
            a = copy.deepcopy(y)
            # Move all of the axes we are not touching up front
            i = 0
            for axis in hp_subset:
                a = np.swapaxes(a, axis, i)
                i += 1
            # Take the mean along all other axes
            for axis in list(range(len(hp_subset), len(a.shape)-1))[::-1]:
                a = np.mean(a, axis=axis)
            # a now gives the performance of the values of the parameters in
            # hp_subset marginalized over all other parameter values.
            return a

        def sum_f_subsets(hp_subset):
            """sum_f_subsets

            Compute the sum of the f values over all subsets of hp_subset.
            """
            hp_subset = list(hp_subset)
            dims = np.array(y.shape)
            # Define a helper for this helper! :) 
            def expand_dims(data, subset):
                """expand_dims

                We need to make the dimensions of some data with the axis
                in subset match the dimensions of some data with the axes
                hp_subset so that we can add them together such that 
                broadcasting works in the intuitive way. This function
                creates the needed dimensions for us.
                """
                
                for i in range(len(hp_subset)):
                    if hp_subset[i] not in subset:
                        data = np.expand_dims(data, i)
                return data
            # Start with nothing
            f_total = np.zeros(marginal[tuple(hp_subset)].shape)
            # For each possible size of the subsets of hp_subset
            for order in range(len(hp_subset)):
                # Get all of the subsets
                for subset in itertools.combinations(hp_subset, order):
                    # Create new axes to match the dimension of f_total
                    # and add contribution of that subset of parameters,
                    # broadcasted along each missing dimension.
                    f_total = f_total + expand_dims(f[subset], subset)

            return f_total

        if samples:
            y = self.y_samples
        else:
            y = self.y
        y, _, dimensions = self._embed_y(self.X, y)
        
        # Initialize some variables and comment on their names in 
        # "An Efficient Approach for Assessing Hyperparameter Importance"
        # variance = V
        variance = {}
        # marginal = \hat{a)
        marginal = {}
        # f = \hat(f)
        f = {}
        # importance = F
        importance = {}
        # Calculate the total_variance = V
        y_reshaped = y.reshape((-1, y.shape[-1]))
        total_variance = np.var(y_reshaped, axis=0) 
        # The total importance of the orders of interaction in order
        if len(total_variance) > 1:
            total_importance = np.zeros(total_variance.shape)
        else:
            total_importance = 0


        self.marginal = marginal
        self.f = f
        self.importance = importance

        hps = list(range(len(self.hyperparameters)))
        if not isinstance(order, list):
            order_list = [order]
        else:
            order_list = order
        max_order = max(order_list)
        # order = k
        for order in range(0,max_order+1):
            # hp_subset = U
            for hp_subset in itertools.combinations(hps, order):
                marginal[hp_subset] = get_marginal(hp_subset)
                f[hp_subset] = marginal[hp_subset] - sum_f_subsets(hp_subset)
                # Take the variance of all data, but keep the last axis
                # separate since we're compute the fANOVA score for each
                # sample of the objective function individually.
                f_reshaped = f[hp_subset].reshape((-1, y.shape[-1]))
                variance[hp_subset] = np.var(f_reshaped, axis=0)
                parsed_hps = tuple([self.hyperparameters[hp] 
                                    for hp in hp_subset])
                if len(parsed_hps) in order_list:
                    var = variance[hp_subset]/total_variance*100
                    if len(var) == 1:
                       var = var[0]
                    importance[parsed_hps] = var
                    total_importance += var

        return importance, total_importance

    def get_1d_slices(self):
        """
        Returns the raw data for the 1D hyper-parameter response slices
        centered around the global optimum.
        """
        optimizer = np.argmin(self.y)
        X_slices = []
        y_slices = []
        y_bounds_slices = []
        y_samples = []
        hyperparameters = []
        for hp in range(self.X.shape[1]):
            hp_name = self.hyperparameters[hp]
            if not isinstance(self.config_space.get_hyperparameter(hp_name), CategoricalHyperparameter):
                slice_ = np.logical_and(np.all(self.X[:,:hp] == self.X[optimizer,:hp], axis=1),
                                        np.all(self.X[:,hp+1:] == self.X[optimizer,hp+1:], axis=1))
                X_slices.append(self.X[slice_,hp])
                y_slices.append(self.y[slice_])
                y_bounds_slices.append(self.y_bounds[slice_])
                y_samples.append(self.y_samples[slice_])
                hyperparameters.append(hp_name)
        return hyperparameters, X_slices, y_slices, y_bounds_slices, y_samples

    def plot_1d_slices(self, fs=18):
        """
        Returns the names and plots (altair-viz charts) of each
        1D hyper-parameter response slice centered around the global
        optimum.
        """
        hyperparameters, X_slices, y_slices, y_bounds_slices, _ = self.get_1d_slices()
        charts = []
        for i, hp in enumerate(hyperparameters):
            steps = np.array(np.diff(X_slices[i]).squeeze(), dtype=float)
            close = np.isclose(steps, np.min(steps), rtol=1)
            linear_scale = np.all(close)
            df = pd.DataFrame({hp: X_slices[i],
                               'Loss': y_slices[i],
                               'Loss Upper': y_bounds_slices[i][:,1],
                               'Loss Lower': y_bounds_slices[i][:,0]})
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X(hp, scale=alt.Scale(type='linear' if linear_scale else 'log')),
                y=alt.Y('Loss', title='Loss + CI'),
            )
            chart += alt.Chart(df).mark_area(opacity=0.3).encode(
                x=hp,
                y='Loss Lower',
                y2='Loss Upper'
            ).interactive(bind_x=False)
            chart = chart.configure_legend(
                labelFontSize=fs,
                titleFontSize=fs,
            ).configure_axis(
                labelFontSize=fs,
                titleFontSize=fs,
            )
            charts.append(chart)
        return hyperparameters, charts

    def plot_2d_slices(self, y_title='Loss + CI', y_scale='linear', x_scale=None, fs=18):
        """
        Returns the names and plots (altair-viz charts) of each
        2D hyper-parameter response slice centered around the global
        optimum.
        """

        hyperparameters, X_slices, y_slices, y_bounds_slices, _ = self.get_2d_slices()
        charts = []
        for i, hp in enumerate(hyperparameters):
            hp1_cat = isinstance(self.config_space.get_hyperparameter(hp[0]), CategoricalHyperparameter)
            hp2_cat = isinstance(self.config_space.get_hyperparameter(hp[1]), CategoricalHyperparameter)
            if hp1_cat and hp2_cat:
               charts.append(None)
            elif hp1_cat or hp2_cat:
                hp_num, hp_cat = (1, 0) if hp1_cat else (0, 1)
                X = X_slices[i]
                values = sorted(np.unique(np.array(X[:,hp_num], dtype=float).squeeze()))
                steps = sorted(np.diff(values))
                close = np.isclose(steps, np.min(steps), rtol=1)
                linear_scale = np.all(close) if x_scale is None else x_scale
                df = pd.DataFrame({hp[0]: X[:,0],
                                   hp[1]: X[:,1],
                                   'Loss': y_slices[i],
                                   'Loss Upper': y_bounds_slices[i][:,1],
                                   'Loss Lower': y_bounds_slices[i][:,0]})
                #st.write(df)
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X(hp[hp_num], scale=alt.Scale(type='linear' if linear_scale else 'log')),
                    y=alt.Y('Loss', 
                            title=y_title,
                            scale=alt.Scale(domain=[df['Loss Lower'].min()/1.1,
                                                    df['Loss Upper'].max()*1.1],
                                            type=y_scale)),
                    color=alt.Color(hp[hp_cat] + ':N'),
                    #opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
                )#.add_selection(selection)
                chart2 = alt.Chart(df).mark_area(opacity=0.3).encode(
                    x=hp[hp_num],
                    y='Loss Lower',
                    y2='Loss Upper',
                    color=hp[hp_cat] + ':N',
                    #opacity=alt.condition(selection, alt.value(0.3), alt.value(0.05)),
                ).interactive(bind_x=False)
                chart = chart2 + chart
                chart = chart.configure_legend(
                    labelFontSize=fs,
                    titleFontSize=fs,
                ).configure_axis(
                    labelFontSize=fs,
                    titleFontSize=fs,
                )
                charts.append(chart)
            else:
                linear_scale = []
                steps = []
                vals = []
                X = np.array(X_slices[i], dtype=float)
                for hp_idx in [0, 1]:
                    values = sorted(np.unique(X[:,hp_idx].squeeze()))
                    steps.append(np.diff(values))
                    steps[hp_idx] = sorted(steps[hp_idx])
                    close = np.isclose(steps[hp_idx], np.min(steps[hp_idx]), rtol=1)
                    linear_scale.append(np.all(close) if x_scale is None else x_scale)
                    vals.append(values)
                    #st.write(values)
                hp_order = [0, 1] if len(steps[0]) > len(steps[1]) else [1, 0]
                #selection = alt.selection_multi(fields=[hp[hp_order[1]]], bind='legend')
                df = pd.DataFrame({hp[0]: X[:,0],
                                   hp[1]: X[:,1],
                                   'Loss': y_slices[i],
                                   'Loss Upper': y_bounds_slices[i][:,1],
                                   'Loss Lower': y_bounds_slices[i][:,0]})
                if not linear_scale[hp_order[1]]:
                    hp_color = 'log(' + hp[hp_order[1]] + ')'
                    df[hp_color] = df[hp[hp_order[1]]].map(np.log)
                else:
                    hp_color = hp[hp_order[1]]
                #st.write(df)
                df = df.sort_values(hp[hp_order[1]])
                chart = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X(hp[hp_order[0]], scale=alt.Scale(type='linear' if linear_scale[hp_order[0]] else 'log')),
                    y=alt.Y('Loss', 
                            title=y_title,
                            scale=alt.Scale(domain=[df['Loss Lower'].min()/1.1,
                                                    df['Loss Upper'].max()*1.1],
                                            type=y_scale)),
                    color=alt.Color(hp_color + ':Q', legend=alt.Legend(type='gradient', titleOrient='right')),
                    #opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
                )#.add_selection(selection)
                chart2 = alt.Chart(df).mark_area(opacity=0.3).encode(
                    x=hp[hp_order[0]],
                    y='Loss Lower',
                    y2='Loss Upper',
                    color=hp_color + ':Q',
                    #opacity=alt.condition(selection, alt.value(0.3), alt.value(0.05)),
                ).interactive(bind_x=False)
                chart = chart2 + chart
                chart = chart.configure_legend(
                    labelFontSize=fs,
                    titleFontSize=fs,
                ).configure_axis(
                    labelFontSize=fs,
                    titleFontSize=fs,
                )
                charts.append(chart)
           
        return hyperparameters, charts

    def get_1d_analyzers(self, permute_objective=False):
        """
        Returns the names of and benchmark analyzers for each 1D
        hyper-parameter response slice centered around the global
        optimum.
        """
        hyperparameters, X_slices, _, y_bounds_slices, y_samples = self.get_1d_slices()
        benchmarks = []
        for X, y_bounds, y_samples, hp in zip(X_slices, y_bounds_slices, y_samples, hyperparameters):
            X = X.reshape((-1, 1))
            cs = ConfigurationSpace()
            cs.add_hyperparameter(self.config_space.get_hyperparameter(hp))
            benchmark = Benchmark(
                X=X, y_samples=y_samples, hyperparameters=[hp], config_space=cs,
                y_bounds=y_bounds, scenario_name=f'{self._dataset}_{hp}',
                permute_objective=permute_objective)
            benchmarks.append(benchmark)
        return hyperparameters, benchmarks

    def get_2d_slices(self):
        """
        Returns the raw data for the 2D hyper-parameter response slices
        centered around the global optimum.
        """
        optimizer = np.argmin(self.y)
        X_slices = []
        y_slices = []
        y_bounds_slices = []
        y_samples = []
        hyperparameters = []
        for hp_1 in range(self.X.shape[1]):
            for hp_2 in range(hp_1+1, self.X.shape[1]):
                hp_1_name = self.hyperparameters[hp_1]
                hp_2_name = self.hyperparameters[hp_2]

                slice_ = np.all(self.X[:,:hp_1] == self.X[optimizer,:hp_1], axis=1)
                if hp_1+1 != hp_2:
                    slice_ = np.logical_and(slice_,
                                            np.all(self.X[:,hp_1+1:hp_2] == self.X[optimizer, hp_1+1:hp_2], axis=1))
                slice_ = np.logical_and(slice_,
                                        np.all(self.X[:,hp_2+1:] == self.X[optimizer,hp_2+1:], axis=1))
                #print(f'{hp_1} - {hp_2}')
                #print(self.X[slice_])
                X_slices.append(self.X[slice_][:,(hp_1, hp_2)])
                y_slices.append(self.y[slice_])
                y_bounds_slices.append(self.y_bounds[slice_])
                y_samples.append(self.y_samples[slice_])
                hyperparameters.append((hp_1_name, hp_2_name))
        return hyperparameters, X_slices, y_slices, y_bounds_slices, y_samples

    def get_2d_analyzers(self, permute_objective=False):
        """
        Returns the names of and benchmark analyzers for each 2D
        hyper-parameter response slice centered around the global
        optimum.
        """
        hyperparameters, X_slices, _, y_bounds_slices, y_samples = self.get_2d_slices()
        benchmarks = []
        for X, y_bounds, y_samples, (hp_1, hp_2) in zip(X_slices, y_bounds_slices, y_samples, hyperparameters):
            cs = ConfigurationSpace()
            cs.add_hyperparameter(self.config_space.get_hyperparameter(hp_1))
            cs.add_hyperparameter(self.config_space.get_hyperparameter(hp_2))
            from single_sample_analyzer import SingleSampleBenchmark
            BenchmarkClass = SingleSampleBenchmark if issubclass(type(self), SingleSampleBenchmark) else Benchmark
            benchmark = BenchmarkClass(
                X=X, y_samples=y_samples, hyperparameters=[hp_1, hp_2], config_space=cs,
                y_bounds=y_bounds, scenario_name=f'{self._dataset}_{hp_1}_{hp_2}',
                permute_objective=permute_objective)
            benchmarks.append(benchmark)
        return hyperparameters, benchmarks

    def interesting(self):
        """
        Determines if the overall variance in the response slice is 
        large enough to be considered ``interesting'', given the size
        of the confidence intervals. Originally defined in Pushak &
        Hoos, 2018, modified here to not be applied to the logarithm
        of the running times, since these measures of performance
        are better analyzed on a linear scale.
        """
        bounds = self.y_bounds
        min_upper = np.min(bounds[:,1])
        max_lower = np.max(bounds[:,0])
        mean_size = np.mean(bounds[:,1] - bounds[:,0])
        return (min_upper - max_lower) <= mean_size/2

    def optimize_independently(self, timeout=30*60, n_processes=24):
        """
        Optimize each hyper-parameter independently a single time, and
        in a random order. Returns the percentage of the orders 
        for which the lower bound of the final incument is better than
        the upper bound of the global optimum.
        """
        permutations = list(itertools.permutations(range(len(self.hyperparameters))))
        np.random.shuffle(permutations)
        optimizer = np.argmin(self.y)
        position = np.argmax(self.y)
        X = self.X
        y = self.y
        y_bounds = self.y_bounds
        start_time = time.time()

        def args(permutations):
           for permutation in permutations:
               if time.time() - start_time < timeout:
                   yield (X, y, y_bounds, optimizer, position, permutation)
               else:
                   break

        with Pool(n_processes) as p:
            r = list(tqdm.tqdm(p.imap_unordered(_optimize, args(permutations), chunksize=100), total=len(permutations)))
        return np.mean(r)*100, len(r), len(permutations)                


def _optimize(args):
    X, y, y_bounds, optimizer, position, order = args
    for hp in order:
        slice_ = np.where(X[:,hp] == X[position,hp])[0]
        position = slice_[np.argmin(y[slice_])]
    return  y_bounds[position,0] <= y_bounds[optimizer,1]

