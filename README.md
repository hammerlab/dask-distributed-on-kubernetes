# mhcflurry-cloud
Infrastructure for parallelized mhcflurry model selection

This module provides [joblib](https://github.com/joblib/joblib)-enabled routines
to train and test [mhcflurry](https://github.com/hammerlab/mhcflurry) models.

## Python 3.4 only for now on a single node

When using the multiprocessing backend for joblib (the default), the 'fork' mode causes a library we use to hang. We have to instead use the 'spawn' or 'forkserver' modes. See this
[note](https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries) for more information.

## Running locally

Install the package and run tests. From the repo directory:
```
$ pip install -e .
$ nosetests .
```


Then try running the [example notebook](notebooks/example1.ipynb).


## Running on google cloud

TODO
