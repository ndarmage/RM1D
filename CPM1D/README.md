---
title: CPM1D
author: Daniele tomatis
date: 27/09/2019
description: This folder contains services to compute the 1D integral
             transport equation by the collision probability method. Services
             to compute transfer and escape probabilities are also included.
---

# CPM1D

<!-- [Demo](http://lotabout.github.io/xxx/) -->

A libraty for collision probability method in 1d reference frames (slab,
cyclinder and sphere), created by [ndarmage](https://github.com/ndarmage).

# Install (in a virtual environment)

Create an activate a virtual environment:

```sh
# create virtual env
python3 -m venv my-virtual-env
# activate virtual env
my-virtual-env/bin/activate
```

Install the package (each time package sources are modified)

```sh
pip install path_to_this_directory
```

Use it
```sh
python
```
```python
import KinPy.algo609 as algo
dir(algo)
['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_algo609_error', 'bdiff', 'bkias', 'bkisr', 'bskin', 'd1mach', 'dbdiff', 'dbkias', 'dbkisr', 'dbskin', 'dexint', 'dgamrn', 'dhkseq', 'dpsixn', 'dqckin', 'exint', 'fdump', 'gamrn', 'hkseq', 'i1mach', 'j4save', 'psixn', 'qckin', 'r1mach', 's88fmt', 'xerabt', 'xerclr', 'xerctl', 'xerdmp', 'xermax', 'xerprt', 'xerror', 'xerrwv', 'xersav', 'xgetf', 'xgetua', 'xgetun', 'xsetf', 'xsetua', 'xsetun']
```
