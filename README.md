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

SP and RP survey conducted for a new train service between Montreal and New York called Train Hotel.

## Getting Started

### Prerequisites

Python 3.5+ (with pip3), Numpy, Pandas, Theano

navigate to project folder

```
>>> apt-get install python3-dev pip3
>>> pip3 install -r requirements.txt
```

This command installs Theano from github.com/Theano/Theano

## Model estimation

To run MNL model:
* python3 run_mnl.py

To run MXL model:
* python3 run_mxl.py

To run ICLV model:
* python3 run_iclv.py

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

0.1 Initial version

## Authors

* **Melvin Wong** - *Initial work* -

See also the list of [contributors](https://github.com/mwong009/iclv_rbm/contributors) who participated in this project.

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* BIOGEME http://biogeme.epfl.ch/
