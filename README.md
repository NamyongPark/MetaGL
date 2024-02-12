## MetaGL
This repository contains code and data used in the paper ["MetaGL: Evaluation-Free Selection of Graph Learning Models via Meta-Learning"](https://openreview.net/pdf?id=C1ns08q9jZ) (ICLR 2023).

<p align="center">
<img src="img/MetaGL.png" width="480" height="381">
</p>

### How to install
Running [install.sh](install/install.sh) will set up the conda environment for MetaGL and install required packages.

### How to run
You can run MetaGL by executing `python main.py`.


## GLEMOS Benchmark
A comprehensive benchmark environment for evaluation-free selection of graph learning models is available 
in the [GLEMOS repository](https://github.com/facebookresearch/glemos), which provides 
a suite of [model selection algorithms](https://github.com/facebookresearch/glemos/tree/main/src/model_selection_methods) including MetaGL, 
[evaluation testbeds](https://github.com/facebookresearch/glemos/tree/main/src/testbeds), and 
[meta-graph features](https://github.com/facebookresearch/glemos/tree/main/src/metafeats), among others.


## Citation
If you use code or data in this repository, please cite our paper.

    @inproceedings{park2023metagl,
      title={Meta{GL}: Evaluation-Free Selection of Graph Learning Models via Meta-Learning},
      author={Namyong Park and Ryan A. Rossi and Nesreen Ahmed and Christos Faloutsos},
      booktitle={The Eleventh International Conference on Learning Representations},
      year={2023},
      url={https://openreview.net/forum?id=C1ns08q9jZ}
    }
