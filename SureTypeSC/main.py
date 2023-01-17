import sys
import numpy as np
import os
from pathlib import Path

cwd = str(Path(__file__).parent)
sys.path.insert(0, cwd)



#from . import MachineLearning
#from . import Config
import logging
import SureTypeSC.MachineLearning as MachineLearning
import SureTypeSC.Config as Config

#import SureTypeSC
#from SureTypeSC import MachineLearning
#from SureTypeSC import Config


import pickle


#config file
config_path='CONFIG/GM7228_training_mlp.conf'

#train a model
if __name__ == "__main__":
  np.warnings.filterwarnings('ignore')
  #jb.Parallel._print = _print


  #load config
  Config.load(config_path)
  os.chdir(os.path.dirname(os.path.abspath(config_path)))

  #init logger
  logging.basicConfig(format='%(asctime)s %(message)s',filename=Config.D["LOGFILE"], level=logging.DEBUG)
  logging.info("Started using " + config_path)

  sc,gdna=MachineLearning.starting_procedure(decimal=',')

  training = MachineLearning.Trainer(sc,clfname=Config.D['CLFNAME'],outer='output')
  training.train()

  with open(Config.D["CLASSIFIER_PATH"], "wb") as output_file:
    # cPickle.dump(training, output_file)
    pickle.dump(training.clf, output_file)
  print(1)












