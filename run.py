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
s=solutions.solutions(parallel=True,n_jobs=5)
st=time.time()
n=7
#l=s(n,verbose=True,print_step=500000, nmin=-10,nmax=10,zmax=32)
l =s(n,verbose=True,print_step=500000,nmin=-10,nmax=10,zmax=32,max_size= 900000000) #7
#l =s(n,verbose=True,print_step=500000,nmin=-15,nmax=15,zmax=32,max_size=900000000) #8
#l =s(n,verbose=True,print_step=1000000,nmin=-12,nmax=12,zmax=32)
print(time.time()-st)
l=[d for d in l if d]
df=pd.DataFrame(l)
df['strz']=df['z'].astype(str)
df.drop_duplicates('strz').drop('strz',axis='columns').reset_index(drop=True).to_json('s{}.json'.format(n))
