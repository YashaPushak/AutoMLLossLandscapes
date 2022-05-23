import sys
import os
sys.path.append(os.getcwd()) 

import copy

import numpy as np
import pandas as pd

from nasbench.api import OutOfDomainError
import nasbench
from ConfigSpace.configuration_space import Configuration

import helper
from helper import Progress
from benchmark_analyzer import Benchmark

class NASBenchmark(Benchmark):
    
    def _initialize_configuration_table(self):
        """
        Creates a pandas dataframe with an entry for every valid configuration in
        self.config_space.
        """

        # Check to see if we can load the data directly, if not, compute it from scratch.
        try:
            self._configuration_table = helper.loadObj('./tables/', 'cifar10')
            samples = helper.loadObj('./samples/', 'cifar10')
        except:
            # Get a list of all of the hashed unique valid configurations
            config_ids = list(self.benchmark.dataset.hash_iterator())
            configurations = []
            progress = Progress(len(config_ids),
                                '{}% done making the configuration table')
            for config_id in config_ids:
                progress.update()
                config = self.get_configuration_from_ID(config_id)
                configurations.append(config)
            
            self._configuration_table = pd.DataFrame({'Configuration ID': config_ids,
                                                      'Configuration': configurations,
                                                      'Visited': False,
                                                      'Parent': None,
                                                      'Quality': float('inf')})
            self._configuration_table = self._configuration_table.drop_duplicates('Configuration ID')
            configurations = list(self._configuration_table['Configuration'])
            # This column doesn't save properly, and we don't really need it in what follows anyways.
            self._configuration_table = self._configuration_table.drop(columns='Configuration')
            progress = Progress(len(config_ids),
                            '{}% done extracting the sample values')
            # Save the configuration table for later use
            helper.saveObj('./tables/', self._configuration_table, 'cifar10') 
            
            # Now load the sample values for the MSE for each configuration
            samples = []
            for config in configurations:
                progress.update()
                sample = [[self.benchmark.objective_function_deterministic(config,
                                                                           index=i,
                                                                           dataset=dataset_type)[0]
                          for i in range(int(round(self._num_trials)))]
                          for dataset_type in self.dataset_type]
                samples.append(sample)
            samples = np.asarray(samples)
            helper.saveObj('./samples/', samples, 'cifar10')

        if self.multiple_test_correction:
            #Bonferroni multiple test correction
            alpha = 1 - self.confidence_level
            alpha /= len(samples)
            confidence_level = 1 - alpha
        else:
             confidence_level = self.confidence_level

        if self.confidence_interval == 'nested student-t bootstrap':
            cis = self.confidence_interval_nested_student_bootstrap(samples, confidence_level)
        elif self.confidence_interval == 'nested bootstrap':
            cis = self.confidence_interval_nested_bootstrap(samples, confidence_level)
        elif self.confidence_interval == 'student-t':
            cis = self.confidence_interval_student(samples, confidence_level)
        #elif self.confidence_interval == 'bootstrap':
        #    cis = self.confidence_interval_bootstrap(samples, confidence_level)

        if self._permute_objective:
            self._random.shuffle(cis)

        self._configuration_table['Lower Bound'] = cis[:,0]
        self._configuration_table['Upper Bound'] = cis[:,1]
        for s in range(self._num_trials):
            self._configuration_table['Validation Sample {}'.format(s)] = samples[:,0,s]
            self._configuration_table['Test Sample {}'.format(s)] = samples[:,1,s]

        self._confidence_interval \
            = dict(zip(np.array(self._configuration_table['Configuration ID']),
                       cis))

        self._mark_all_unreachable()
        self._stale_table = False


    def get_unique_configuration_identifier(self,
                                            config):
        """
        Outputs a string representation of a hash for the model.
        This representation is not unique for each configuration as specified in config, because
        there are multiple encodings in config that map to identical neural architecture graphs.
        This instead returns a unique hash for each unique graph model.
        
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

        model_spec, invalid_config = self.benchmark.get_model_spec(config)
        if invalid_config:
            raise OutOfDomainError("Invalid configuration.")
        # Raise an error if it's invalid
        self.benchmark.dataset._check_spec(model_spec)

        return self.benchmark.dataset._hash_spec(model_spec)

    def get_neighbours(self,
                       config):
        # First, we prune this configuration of all extraneous edges
        config = self._prune_edges(config)
        # Get the neighbours formed by adding or removing individual edges
        # or modifying the operations of a single layer
        possible_neighbours = super().get_neighbours(config)
        # Create a set of neighbours
        neighbours = set([])
        for neighbour in possible_neighbours:
            # Check if each neighbour is valid
            if self.is_valid(neighbour):
                # If it's valid, prune it's extraneous edges (which can only be the new edge,
                # since we just pruned it's parent).
                neighbour = self._prune_edges(neighbour)
                # Which means that it might have become it's parent. We don't want that!
                if neighbour != config:
                    neighbours.add(neighbour)
        # Now get the neighbours formed by adding a single vertex
        possible_neighbours = self._get_add_vertex_neighbours(config) 
 
        for neighbour in possible_neighbours:
            # Make sure it's valid -- although, unless I made a mistake somewhere prior to this
            # it always will be.
            if self.is_valid(neighbour):
                neighbours.add(neighbour)

        return list(neighbours)
            
    def is_valid(self,
                 config):
        model_spec, invalid_spec = self.benchmark.get_model_spec(config) 
        if invalid_spec:
           return False

        return self.benchmark.dataset.is_valid(model_spec)

    def pad_matrix(self,
                   original_matrix,
                   original_operations):
        matrix = copy.deepcopy(original_matrix)
        operations = copy.deepcopy(original_operations)
        while len(matrix) < 7:
            # Prepend some meaningless zeros
            matrix = np.insert(matrix, 1, 0, axis=0)
            matrix = np.insert(matrix, 1, 0, axis=1)
            # Insert a meaningless operation
            operations.insert(1, 'maxpool3x3')
        # We did the following once and it passed for all configurations, no need to do it again
        # Make sure that we get back the same reduced configuration in the end
        # model_spec = nasbench.api.ModelSpec(matrix, operations)
        # assert (model_spec.matrix == original_matrix).all()
        # assert model_spec.ops == original_operations
        # We're good to go!
        return matrix, operations

    def get_smallest_unvisited_minima(self):
        visited = self._get_visited_at_least_once()
        self._configuration_table.to_csv('debug.csv')
        df = self._configuration_table[~visited]
        indexer = df['Upper Bound'] == df['Upper Bound'].min()
        smallest = df[indexer].iloc[0]['Configuration ID']
        smallest = self.get_configuration_from_ID(smallest)
        if self._num_modes == 1:
            self._global_minima = smallest
        return smallest

    def get_configuration_from_ID(self,
                                  config_id):
        # Look up the "fixed stats" of the config ID (hash)
        fixed_stats = self.benchmark.dataset.get_metrics_from_hash(config_id)[0]
        # Pad the matrix and operations if necessary
        matrix, operations = self.pad_matrix(fixed_stats['module_adjacency'],
                                             fixed_stats['module_operations'])
        # Create a model_spec from the matrix and operations
        model_spec = nasbench.api.ModelSpec(matrix, 
                                            operations)
        # Convert the model_spec into a Configuration
        config = self.benchmark.get_configuration(model_spec)
        return config

    def _was_visited(self,
                     config):
        # We only care about whether or not each unique graph can be reached; however,
        # if we don't consider paths that contain duplicated configurations then we will
        # never be able to move from graphs with n nodes to graphs with n+1 nodes,
        # since a node is only added to the prunned graph if it has both an incoming and
        # an outgoing edge. Therefore, we need to allow for each duplicate encoding
        # of the graph to be searched.
        #
        # We don't have enough memory to check things for all unique configurations...
        # Instead we're going to have to try hoping that everything is fully connected
        # using a modifed neighbourhood network when we restrict ourselves to visiting
        # unique networks exactly once. This seems reasonable, since everything should
        # be symmetric.
        return super()._was_visited(config)
        # unique_id = super().get_unique_configuration_identifier(config)
        # return self._uniquely_visited.get(unique_id, False)

    def _mark_visited(self,
                      config,
                      quality,
                      parent):
        # We need to track whether or not every duplicate version of each configuration has 
        # been visited. See comment in _was_visited()
        unique_id = super().get_unique_configuration_identifier(config)
        self._uniquely_visited[unique_id] = True
        super()._mark_visited(config, quality, parent)

    def _mark_all_unvisited(self):
        self._uniquely_visited = {}
        super()._mark_all_unvisited()

    def _get_add_vertex_neighbours(self, config):
        """_get_add_vertex_neighbours
        Returns all configurations with exactly one vertex added to them
        
        Parameters
        ----------
        config : Configuration
            Must have all extraneous edges removed prior to call.
            See _prune_edges().
        
        Returns
        -------
        neighbours : list of Congifuration
            The neighbours of config.
        """
        model_spec = self.benchmark.get_model_spec(config)[0]
        original_matrix = model_spec.original_matrix
        if original_matrix.sum() > 7:
            # Adding any vertex causes us to exceed the limit on the number
            # of edges
            return []

        # Create a mapping from vertices to edge indices
        inds = list(range(0,21))[::-1]
        m = np.zeros((7,7)) - 1
        for y in range(0, 7):
            for x in range(y+1,7):
                m[y,x] = inds.pop()

        active, extraneous = self._split_vertices(original_matrix)
        neighbours = []
        for v in extraneous:
            # Find all possible source edges for v
            sources = m[:,v] >= 0
            # Get the vertices of those edges
            v_sources = np.nonzero(sources)[0]
            # Only keep those that are actually active
            v_sources = list(active.intersection(v_sources))
            # Now get the source edges corresponding to active vertices
            e_sources = m[v_sources,v]
            # Get the active destination edges
            dests = m[v,:] >= 0
            v_dests = np.nonzero(dests)[0]
            v_dests = list(active.intersection(v_dests))
            e_dests = m[v,v_dests]
            # Return all combinations of source and destination edges
            for source in e_sources:
                for dest in e_dests:
                    neighbour = copy.deepcopy(config)
                    neighbour['edge_{}'.format(int(source))] = 1
                    neighbour['edge_{}'.format(int(dest))] = 1
                    # Make sure no mistakes were made
                    new_model_spec = self.benchmark.get_model_spec(neighbour)[0]
                    # print(new_model_spec.original_matrix)
                    new_active, _ = self._split_vertices(new_model_spec.original_matrix)
                    # print(new_active)
                    # print(v)
                    new_active.remove(v)
                    assert new_active == active
                    # We're good to go!
                    neighbours.append(neighbour)

        return neighbours

    def _prune_edges(self, config):
        """_prune_edges
        Return a new configuration with any extraneous edges removed.
        Extraneous edges are edges pointing to vertices that are not
        reachable from both the source and sink and hence are not used.
        Unlike the prune method in the nasbench API, we do not prune these
        unused vertices from the matrix, because then we would be unable
        to encode the matrix properly in the configuration.        

        Parameters
        ----------
        config : Configuration
            The configuration to be pruned.

        Returns
        -------
        config : Configuration
             The configuration with its extraneous edges removed.
        """
        # Create a mapping from vertices to edge indices
        # And create the "original matrix" that would exist if there was
        # no constraint on the number of edges. This is necessary, since
        # many edges are extraneous and can be deleted anyways.
        inds = list(range(0,21))[::-1]
        m = np.zeros((7,7)) - 1
        matrix = np.zeros((7,7))
        for y in range(0, 7):
            for x in range(y+1, 7):
                idx = inds.pop()
                m[y,x] = idx
                matrix[y,x] = config['edge_{}'.format(int(idx))]
        # Determine which vertices are active and extraneous
        active, extraneous = self._split_vertices(matrix)
        extraneous = list(extraneous)
        # Delete all edges to and from extraneous vertices
        matrix[extraneous,:] = 0
        matrix[:,extraneous] = 0
        # Now create a new config with only the edges of this new matrix
        new_config = copy.deepcopy(config)
        # Start by setting all edges off
        for edge in range(0,21):
            new_config['edge_{}'.format(int(edge))] = 0
        # Then enable the ones we have leftover
        for edge in m[np.nonzero(matrix)]:
            new_config['edge_{}'.format(int(edge))] = 1
        # Make sure no mistakes were made, i.e., what we have done is
        # consistent with the original API's prunning method.
        model_spec = self.benchmark.get_model_spec(config)[0]
        new_model_spec = self.benchmark.get_model_spec(new_config)[0]
        assert (model_spec.matrix == new_model_spec.matrix).all()
        assert model_spec.ops == new_model_spec.ops

        return new_config

    def _split_vertices(self, original_matrix):
        num_vertices = np.shape(original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        active = visited_from_input.intersection(visited_from_output)  
        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(active)

        return active, extraneous

