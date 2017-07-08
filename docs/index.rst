.. SPN documentation master file, created by
   sphinx-quickstart on Thu Apr 20 16:19:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SPN's documentation!
===============================

SPN is library to build, train and save neural networks based on Theano.

SPN defines a neural network image on hard disk to reuse and modify.

#########
Tutorials
#########

.. toctree::
   :maxdepth: 1

   tutorial/mnist
   tutorial/unet
   tutorial/binarynet
   tutorial/vbn
   tutorial/gan


############
User's Guide
############

.. toctree::
   :maxdepth: 2

   user-guide/layer
   user-guide/network
   user-guide/activation-unit
   user-guide/optimization


#############
API Reference
#############

The following is the document extracted from code.

.. toctree::
   :maxdepth: 2

   modules/network	      
   modules/layers
   modules/cost
   modules/gradient_optimizer
   modules/regularization
   modules/util
	     

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
