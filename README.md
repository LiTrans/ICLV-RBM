# ICLV-RBM model estimation

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

# Dataset

SP and RP survey conducted for a new train service between Montreal and New York called Train Hotel.

### Prerequisites

Python 3.5+, Numpy, Pandas

```
>>> apt-get install python3-dev pip3
>>> pip3 install numpy pandas
```

Theano

```
>>> git clone https://github.com/Theano/Theano.git
>>> cd Theano
>>> pip3 install .
```

## Deployment

To run MNL model:
* python3 run_mnl.py

To run MXL model:
* python3 run_mxl.py

To run ICLV model:
* python3 run_iclv.py

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

TODO

## Authors

* **Melvin Wong** - *Initial work* -

See also the list of [contributors](https://github.com/mwong009/iclv_rbm/contributors) who participated in this project.

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* BIOGEME http://biogeme.epfl.ch/
