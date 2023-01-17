# -*- coding: utf-8 -*-
"""Custom module name or brief description.

Enter description of this module

__author__ = ['Lishan Cai']
__copyright__ = Copyright 2019
__version__ = 1.0
__maintainer__ = ['Ivan Vogel']
__email__ = ['ivogel@sund.ku.dk']
__status__ = Dev
"""
import pickle
from .MachineLearning import Trainer, SerializedTrainer, starting_procedure_GM12878,tm_routine,evaluate_metrics
import pandas as pd
from . import DataLoader
from . import MachineLearning 




def loader(filename):
	"""Helper function for loading the serialized classifiers

	Args:
		filename (pickle): path to serialized classifier

	Returns:
		SerializedTrainer: instance of classifier
	"""
	with open(filename,'rb') as input_file:
		classif = SerializedTrainer(pickle.load(input_file,encoding='latin1'))
	return classif
