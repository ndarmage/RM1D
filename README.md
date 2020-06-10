---
title: RonenMethod1D
author: Daniele Tomatis
date: 30/09/2019
---

```
 ____  __  __ _ ____  
|  _ \|  \/  / |  _ \ 
| |_) | |\/| | | | | |
|  _ <| |  | | | |_| |
|_| \_|_|  |_|_|____/ 
                      
```

# RonenMethod1D (private version)

This repository contains the source files to implement the Ronen method in a 1D diffusion solver using finite differences. Currents are derived by the theory of collision probability methods following Hébert's and Lewis-and-Miller textbooks.

Reference solutions are obtained by resolving the integral transport equation by the collision probability method.

Three simple 1D geometry frames are studied: slab, cylinder and sphere.

## Installation

```
git clone https://github.com/ndarmage/RonenMethod1D
```

# References

Please consider the following bibtex entries as references. The documentation to use this program will be published soon.

```
@article{ronen2004accurate,
  title={Accurate relations between the neutron current densities and the neutron fluxes},
  author={Ronen, Yigal},
  journal={Nuclear science and engineering},
  volume={146},
  number={2},
  pages={245--247},
  year={2004},
  publisher={Taylor \& Francis}
}

@inproceedings{tomatis2011application,
  title={Application of a numerical transport correction in diffusion calculations},
  booktitle={Proc. Int. Conf. on Mathematics and Computational Methods Applied to Nuclear Science and Engineering (M\&C 2011), Rio de Janeiro, RJ Brazil},
  month={May 8--12},
  author={Tomatis, Daniele and Dall'Osso, Aldo},
  year={2011}
}

@article{gross2020high,
  title={High-accuracy neutron diffusion calculations based on integral transport theory},
  author={Gross, Roy and Tomatis, Daniele and Gilad, Erez},
  journal={The European Physical Journal Plus},
  volume={135},
  number={2},
  pages={235},
  year={2020},
  publisher={Springer}
}
```