# StatsML

StatsML is a machine learning and data processing library for Python. It was developed by myself in an attempt to further consolidate my machine learning education and gain experience through implementation of state-of-the-art processes. The library contains many features common to most machine learning libraries, but does not implement any specific hardware-level optimizations. These features range from simple statistical methods, to neural networks, to data processing, and much more.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Technologies](#technologies)
5. [License](#license)

## Getting Started

The following instructions will help you to get StatsML up and running on your local machine for development and testing purposes.
I encourage you to give it a look, experiment with it and give me any feedback you have. 

*All command line documentation below is specifically for the Linux system.*

### Prerequisites

You may choose to run the project either in your machine's local environment or to setup a [virtual
environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) and install all packages there.


The packages found in *requirements.txt* will then need to be installed:

```
$ pip3 install -r requirements.txt 
```

### Installation

To create a local copy of the codebase, navigate to your directory of choice and clone this repository:

```
$ git clone https://github.com/GiovanniPasserello/StatsML.git
```

## Project Structure

StatsML is split into several distinct sections found as a set of directories within 'statsml', each implementing a different machine learning algorithm.
Within each directory is a suite of classes used to implement the specific algorithm, along with an example demonstrating how to interact with the package on a fake dataset.

## Features

* __Neural Network__ - a backpropagating artificial neural network implementation
    * Multi Layer Network
    * Layers
        * Linear
        * Sigmoid
        * ReLu
        * MSE Loss
        * Cross Entropy Loss
    * Automated Training Suite
* __Decision Classifier__ - a decision tree classifier implementation 
    * Decision Tree Classifier    
    * Random Forest Classifier
    * Decision Tree Pruning
    * Cross Validation Suite
* __Clustering__ - a suite of scripts used to cluster multi-dimensional data
    * K Means
    * Gaussian Mixture Model
* __Metrics Evaluation__ - a set of extractable metrics from confusion matrices

## Technologies

StatsML is built entirely from scratch without the use of external packages, aside from NumPy for performance and data handling purposes.

* [Python 3](https://docs.python.org/3/) - StatsML implementation programming language of choice
* [NumPy](https://numpy.org/) - Python library adding efficient support for large, multi-dimensional arrays and matrices

## License

This project is licensed under the MIT License - see [LICENSE](https://github.com/GiovanniPasserello/StatsML/blob/master/LICENSE) for details.
