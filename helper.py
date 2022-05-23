import sys
import os
import pickle
import copy

import numpy as np
import pandas as pd
import scipy.stats as stats

def saveObj(dir, obj, name ):
    with open(dir + '/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadObj(dir, name ):
    with open(dir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def mkdir(dir):
    #Author: Yasha Pushak
    #Last updated: January 3rd, 2017
    #An alias for makeDir
    makeDir(dir)


def makeDir(dir):
    #Only creates the specified directory if it does not already exist.
    #At some point it may be worth adding a new feature to this that saves the old directory and makes a new one if one exists.
    if(not os.path.isdir(dir)):
        os.system('mkdir '  + dir)


def isDir(dir):
    #Author: Yasha Pushak
    #last updated: March 21st, 2017
    #Checks if the specified directory exists.
    return os.path.isdir(dir)


def isFile(filename):
    #Author: Yasha Pushak
    #Last updated: March 21st, 2017
    #CHecks if the specified filename is a file.
    return os.path.isfile(filename)


class Progress:

    def __init__(self,
                 n=100,
                 message='{}% done',
                 log=True,
                 end='\r'):
        self.n = n
        self.iteration = 0
        self.percent_done = 0
        self.message = message
        self.log = log
        self.end = end

    def update(self, message=None):        
        if message is not None:
            self.message = message
        self.iteration += 1
        if self.iteration*100 // self.n > self.percent_done:
            self.percent_done = self.iteration*100 // self.n
            if self.log:
                if self.percent_done < 100:
                    print('    ' + self.message.format(self.percent_done), end=self.end, flush=True)
                else:
                    print('    ' + self.message.format(self.percent_done), end='\n', flush=True)


def confidence_interval_student(samples,
                                confidence_level):
    """confidence_interval_student
    Returns confidence intervals for the samples assuming
    that all samples are IID.

    Parameters
    ----------
    samples : numpy.ndarray
        The array of samples. Rows correspond to a sample, columns correspond to
        the sampled values within each sample.
    confidence_level : float
        The confidence level for the confidence intervals. Must be in (0, 1).

    Returns
    -------
    confidence_intervals : numpy.ndarray
        The confidence intervals for the samples. Each row corresponds to a row
        in the samples array. The columns are in the order [lower bound, upper
        bound].
    """
    # Remove any nesting structure if there is any.
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


def to_cdf(data):
    data = np.array(copy.deepcopy(data))
    n = len(data)
    data = data[np.where(~np.isnan(data))[0]]
    if len(data) < n:
        print("Warning: We dropped some nans when calculating a CDF")
    if len(data) == 0:
        return [np.nan], [np.nan]
    data = list(data)
    data.append(min(data))
    data = sorted(data)
    q = np.arange(0, len(data))/(len(data)-1)*100
    return q, data

def try_numeric(v, numpy=False):
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            pass
    return to_numpy(v) if numpy else v

def to_numpy(v):
    if isinstance(v, int):
        v = np.int64(v)
    elif isinstance(v, float):
        v = np.float64(v)
    return v
