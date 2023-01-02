import pickle
from .MachineLearning import Trainer, SerializedTrainer, starting_procedure_GM12878,tm_routine,evaluate_metrics
import pandas as pd
from . import DataLoader
from . import MachineLearning 




def loader(filename):
	"""load serialized classifier

	Args:
		filename (pickle): path to serialized classifier

	Returns:
		SerializedTrainer: instance of classifier
	"""
	with open(filename,'rb') as input_file:
		classif = SerializedTrainer(pickle.load(input_file,encoding='latin1'))
	return classif
