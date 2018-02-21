from __future__ import division, print_function, absolute_import
from builtins import range

import opto
from opto.opto.classes.OptTask import OptTask
import opto.utils as rutils
from opto.functions import *
from opto.opto.acq_func import *
import opto.regression as rregression

from objective_functions import *

import numpy as np
import matplotlib.pyplot as plt
from dotmap import DotMap

import time
import logging

logger = logging.getLogger()
fh = logging.FileHandler('example.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

init_vrep()
obj_f = generate_f(parameter_mode='discovery', objective_mode='moo', steps=400)
task = OptTask(f=obj_f, n_parameters=8, n_objectives=2, name='MOO', bounds=rutils.bounds(min=[1e-10,0,0,0,0,0,0,0], max=[60, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi]), vectorized=False)
stopCriteria = opto.opto.classes.StopCriteria(maxEvals=100)

p = DotMap()
p.verbosity = 1
p.acq_func = EI(model=None, logs=None)
p.optimizer = opto.CMAES
p.model = rregression.GP
opt = opto.PAREGO(task=task, stopCriteria=stopCriteria, parameters=p)
opt.optimize()
logs = opt.get_logs()
print(logs.get_objectives())

exit_vrep()
