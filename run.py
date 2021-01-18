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
#import warnings
import time
#warnings.filterwarnings("ignore")
n=6
s=solutions.solutions(parallel=True,n_jobs=5)
st=time.time()
l=s(6,verbose=True,print_step=500000, nmin=-21,nmax=21,zmax=32,max_size=100000000)
#l =s(9,verbose=True,print_step=1000000,nmin=-12,nmax=12,zmax=32)
print(time.time()-st)
l=[d for d in l if d]
df=pd.DataFrame(l)
df['strz']=df['z'].astype(str)
df.drop_duplicates('strz').drop('strz',axis='columns').reset_index(drop=True).to_json('s{}.json'.format(n))
