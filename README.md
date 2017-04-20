Sesame-Paste-Noodle(SPN)
===================

[![Build Status](https://travis-ci.org/aissehust/sesame-paste-noodle.svg?branch=master)](https://travis-ci.org/aissehust/sesame-paste-noodle)
[![Documentation Status](https://readthedocs.org/projects/sesame-paste-noodle/badge/?version=latest)](http://sesame-paste-noodle.readthedocs.io/en/latest/?badge=latest)))]]

SPN is a library to build and train deep networks based on Theano.
Its main features are:

* Support common layers and forward networks such as Convolutional Neural Networks.
* Include popular SGD methods such as RMSprop and Adam.
* Allow variety of input/output type, cost function.
* Human readable disk image of saved model based on YAML.
* Easy extendable to have user defined layer.

Installation
------------

    pip install https://github.com/aissehust/sesame-paste-noodle/archive/master.zip

Development Setup
-----------------

    virtualenv ENV
    . ENV/bin/activate
    git clone https://github.com/aissehust/sesame-paste-noodle.git
    cd sesame-paste-noodle
    pip install -e .
