- [ICLV-RBM model estimation](#iclv-rbm-model-estimation)
  * [Dataset](#dataset)
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
      - [Ubuntu (Unix)](#ubuntu--unix-)
  * [Model estimation](#model-estimation)
  * [Contributing](#contributing)
  * [Versioning](#versioning)
  * [Authors](#authors)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)

# ICLV-RBM model estimation

Modelling Latent Travel Behaviour Characteristics with Generative Machine Learning

We implement an information-
theoretic approach to travel behaviour analysis by introducing
a generative modelling framework to identify informative latent
characteristics in travel decision making. It involves developing
a joint tri-partite Bayesian graphical network model using a
Restricted Boltzmann Machine (RBM) generative modelling
framework.

## Dataset

SP and RP survey conducted for a new train service between Montreal and New York (Train Hotel).

Dataset Tech report: [Sp_TrainHotel_Draft 5_Oct 27.2016.pdf](https://github.com/LiTrans/ICLV-RBM/blob/master/Sp_TrainHotel_Draft%205_Oct%2027.2016.pdf)

## Getting Started

This is a starting point if you are new to this project where you will use the python project as-is to generate the estimation model.

### Prerequisites

Python 3.5+ (with pip3), Numpy, Pandas, Theano

### Installation

These are the installation instructions. Consider them work-in-progress and feel free to make suggestions for improvement.

1. Clone or download the git repository and navigate to the project folder

#### Ubuntu (Unix)

The following system packages are required to be installed

```
apt-get install python3 python3-dev pip3
python3 --version
>>> Python 3.X.X
```

Install requirements with pip with `--user` option

```
cd /root-project-folder
pip3 install --user -r requirements.txt
```

The above command also installs the latest Theano from github.com/Theano/Theano

## Model estimation

To run MNL model:
``` python3 run_mnl.py ```

To run MXL model:
```python3 run_mxl.py```

To run ICLV model:
```python3 run_iclv.py```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

0.1 Initial version

## Authors

* **Melvin Wong** [Github](https://github.com/mwong009)

See also the list of [contributors](https://github.com/LiTrans/ICLV-RBM/contributors) who participated in this project.

## License

This project is licensed under the MIT - see [LICENSE.md](LICENSE.md) for details

## Acknowledgments

* BIOGEME http://biogeme.epfl.ch/
