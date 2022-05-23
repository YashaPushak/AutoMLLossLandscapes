# AutoML Loss Landscape Analyzers

Provides a collection of methods for analyzing the loss landcapes of AutoML
problem scenarios. These methods were used to analyze the landscapes studied
in Pushak & Hoos [2022a]. We showed using these methods that the
hyper-parameters of classic machine learning algorithms tend to result in
relatively benign hyper-parameter responses. In particular, these landscapes
appear to be statisticaly indistinguishable from uni-modal (or very close to it)
and they have relativey simple hyper-parameter interactions. That is, optimzing
each hyper-parameter independently a single time, and in a random order, often
yields final incumbents that are statistically tied with optimal.

This work builds on a line of research that seeks to analyze and exploit
algorithm configuration landscapes:

- \[Pushak & Hoos, 2022a\] Yasha Pushak and Holger H. Hoos.
**AutoML Loss Landscapes**
Under review at *Transactions on Evolutionary Optimization and Learning (TELO)*.
 - \[Pushak & Hoos, 2020\] Yasha Pushak and Holger H. Hoos.  
**Golden Parameter Search: Exploiting Structure to Quickly Configure Parameters
In Parallel.**  
*In Proceedings of the Twenty-Second Interntional Genetic and Evolutionary 
Computation Conference (GECCO 2020)*. pp 245-253 (2020).  
**Won the 2020 GECCO ECOM Track best paper award.**
 - \[Pushak & Hoos, 2018\] Yasha Pushak and Holger H. Hoos.  
**Algorithm Configuration Landscapes: More Benign than Expected?**  
*In Proceedings of the Fifteenth Internationl Conference on Parallel Problem 
Solving from Nature (PPSN 2018)*. pp 271-283 (2018).  
**Won the 2018 PPSN best paper award.**

# Usage

The script `benchmark_analyzer.py` defines the `Benchmark` base class,
which contains most of the logic for how to analyze an AutoML loss landscape.
There exists a separate class for each AutoML scenario that inherits from
this base class, and provides alternative implementations of the landscape
analysis methods, where appropriate. Usage of each of these analysis methods
is described in the docstrings of each method. 

For example,

   from svm_analyzer import SVMBenchmark
   analyzer = SVMBenchmark()
   is_unimodal = analyzer.test_reject_unimodality()

will run run the test for uni-modality on the SVM scenario and print some
information to the console about its progress. 

# Contact

Yasha Pushak  
ypushak@cs.ubc.ca  

PhD Student & Vanier Scholar  
Department of Computer Science  
The University of British Columbia  
