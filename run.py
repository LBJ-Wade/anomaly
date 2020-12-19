#!/usr/bin/evn python3
## See anomalysolutions-class.ipynb for details
#from solutions import *
import itertools
import sys
import numpy as np
import pandas as pd

import solutions as solutions
#Private functions are not imported
_get_chiral=solutions._get_chiral
z=solutions.z
import warnings
import time
warnings.filterwarnings("ignore")
s=solutions.solutions(parallel=True,n_jobs=5)
st=time.time()
l=s(6,verbose=True,print_step=500000,nmin=-21,nmax=21,zmax=32)
print(time.time()-st)
