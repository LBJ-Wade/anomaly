#TODO: inherit from free class
import numpy as np
import itertools
import sys
from joblib import Parallel, delayed
from anomalies import anomaly
z=anomaly.free
import warnings
warnings.filterwarnings("ignore")
def _get_chiral(q,q_max=np.inf):
    #Normalize to positive minimum
    if q[0]<0 or (q[0]==0 and q[1]<0):
        q=-q
    #Divide by GCD
    GCD=np.gcd.reduce(q)
    q=(q/GCD).astype(int)
    if ( #not 0 in z and 
          0 not in [ sum(p) for p in itertools.permutations(q, 2) ] and #avoid vector-like and multiple 0's
          #q.size > np.unique(q).size and # check for at least a duplicated entry
          np.abs(q).max()<=q_max
           ):
        return q,GCD
    else:
        return None,None
    
def get_solution(l,k,sols=[],zmax=32):
    q,gcd=_get_chiral( z(l,k) )
    #if q is not None and np.abs(q).max()<=zmax:#
    if q is not None and list(q) not in sols and np.abs(q).max()<=zmax:
        sols.append(list(q))
        return {'l':l,'k':k,'z':list(q),'gcd':gcd}
    else:
        return {}    
    
class solutions(object):
    '''
    Obtain anomaly free solutions with N chiral fields
    
    Call the initialize object with N and get the solutions:
    Example:
    >>> s=solutions()
    >>> s(6) # N = 6
    
    Redefine the self.chiral function to implement further restrictions:
    inherit from this class and define the new chiral function
    '''
    def __init__(self,nmin=-2,nmax=2,zmax=np.inf,parallel=False,n_jobs=1):
        self.nmin=nmin
        self.nmax=nmax
        self.zmax=zmax
        self.CALL=False
        self.free=[]
        self.parallel=parallel
        self.n_jobs=n_jobs


    def __call__(self,N,*args,**kwargs):
        if kwargs.get('nmin'):
            self.nmin=kwargs['nmin']
        if kwargs.get('nmax'):
            self.nmax=kwargs['nmax']            
        if kwargs.get('zmax'):
            self.zmax=kwargs['zmax']                        
        self.CALL=True
        self.N=N
        if N%2!=0: #odd
            N_l=(N-3)//2
            N_k=(N-1)//2
        else: #even
            N_l=N//2-1
            N_k=N_l
        r=range(self.nmin,self.nmax+1)
        self.ls=list(itertools.product( *(r for i in range(N_l)) ))
        self.ks=list(itertools.product( *(r for i in range(N_k)) ))
        if not self.parallel:
            return self.chiral(*args,**kwargs)
        else:
            sols=[]
            kk=1
            ll=[]
            #limit for 70 jobs and 96GB of RAM
            ksmax=900000000 #900E6
            if len(self.ls)*len(self.ks)>ksmax:
                kk=(len(self.ls)*len(self.ks)+ksmax)//ksmax
            for kq in range(kk):
                if kk>1:
                    ll=[d for d in ll if d]
                    sols=[] # May be too big
                ll=ll+Parallel(n_jobs=self.n_jobs)(
                     delayed(get_solution)(l, k,sols,self.zmax) 
                     for k in self.ks for l in self.ls[len(self.ls)*kq//kk:len(self.ls)*(kq+1)//kk] )
            return ll
        
    def chiral(self,*args,**kwargs):
        if not self.CALL:
            sys.exit('Call the initialized object first:\n>>> s=solutions()\n>>> self(5)')
        self.list=[]
        solt=[]
        for l in self.ls:
            for k in self.ks:
                l=list(l)
                k=list(k)
                q,gcd=_get_chiral( z(l,k) )
                #print(z(l,k))
                if q is not None and list(q) not in self.list and np.abs(q).max()<=zmax:
                    self.list.append(list(q))
                    solt.append({'l':l,'k':k,'z':list(q),'gcd':gcd})
        return solt
