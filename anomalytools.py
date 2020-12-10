import pandas as pd
import numpy as np
from astropy.table import Table
import itertools
import sys

def get_conditions(uf,nu_R,l=0,Dim=5):
    #uf=np.sort(np.unique(xdf))
    Ls=-l    
    Type='X'
    if Ls==0:
        Type='D'
        
    if Dim==5:
        d=2
    elif Dim==6:
        d=4
    
    md=[ {'{}→nu_R+{}fi'.format(Type,d):[nu_R,p]} for p in uf if nu_R+d*p-Ls==0 ]
    prmts=get_permutations(uf,3)
    if Dim==5:
        md=md+[ {'{}→nu_R+fi+fj'.format(  Type):list(p)} for p in prmts if nu_R in p and   p.sum()     -Ls==0 ]
    elif Dim==6:
        md=md+[ {'{}→nu_R+2fi+2fj'.format(Type):list(p)} for p in prmts if nu_R in p and 2*p.sum()-nu_R-Ls==0 ]        
    if md:
        return md
    else:
        return []
    
def get_permutations(l,n=3):
    prmts=[]
    for p in itertools. permutations(l, n):
        if sorted(p) not in prmts:
            prmts.append(sorted(p))
    return [ np.asarray(p) for p in prmts if isinstance(p,list) ]

def repeated(f):
    r={}
    r[1]=f
    for ri in range(2,f.size+2):
        r[ri]=r[ri-1][pd.Series(r[ri-1]).duplicated()]
        if ri>2:
            for a in np.unique(r[ri]):
                r[ri-1]=r[ri-1][ r[ri-1]!=a ]
                if len(r[ri-1])>0:
                    r[ri-1][ r[ri-1]!=a ]
        if len(r[ri])<=1:
            r[ri]=r[ri]
            r.pop(1)
            break
    return dict( (k,list(v)) for k,v in r.items() if len(v)!=0 )

def anomalies(row):
    f=np.array( [  row['f{}'.format(i)] for i in range(1,10) if row['f{}'.format(i)]!=0    ]  )
    rptd=repeated(f)
    # it triplets → remove triplet and check subanomaly cancelation:  Σsubf', 'Σsubf³' 
    # check four conditions
    return {'Σf':f.sum(), 'Σf³':(f**3).sum(),'multiplets':rptd}

#row=df.loc[28]
def get_solutions(row,solution='',Dim=5):
    if not solution:
        f=np.array( [  row['f{}'.format(i)] for i in range(1,10) if row['f{}'.format(i)]!=0    ]  )
    else:
        f=np.array(row[solution])
    rptd=repeated(f)

    nu_Rs=[]
    for i in rptd.keys():
        nu_Rs=nu_Rs+rptd.get(i)
    
    #if not rptd use all unique charges
    uf=np.sort(np.unique(f))    

    if not nu_Rs:
        nu_Rs=uf
        
    XD=[]
    for nu_R in nu_Rs:
        Ds=get_conditions(uf,nu_R,l=0,Dim=Dim)
        for D in Ds:
            if D not in XD:
                XD.append(D)
        #If nu_R in triplet → drop it and check the rest for X solutions
        if ( (rptd.get(3) and nu_R in rptd.get(3)) or
             (rptd.get(4) and nu_R in rptd.get(4)) ):
            l=nu_R # nu_R → -L
            xf=uf[uf!=l] #Drop nu_R
            # Drop l from nu_Rs and search new_nu_R there
            new_nu_Rs=nu_Rs.copy()
            new_nu_Rs.remove(l)
            for new_nu_R in new_nu_Rs:
                Xs=get_conditions(xf,new_nu_R,l,Dim=Dim)
                #print(l,new_nu_R,Xs)            
                for X in Xs:
                    if X not in XD:
                        XD.append(X)
        #TODO if rptd.get(4)
    return XD

def extract_multiplets(df,i,greater_than=0,column='solution'):
    dff=df[df[column].apply(
             lambda f: repeated(np.asarray(f))).str[i].apply( #get charges in multiplet i
             lambda l: len(l) if isinstance(l,list) else 0)>greater_than  # count charges in multiplets and filter for >
           ].reset_index(drop=True)
    return dff

def label_solutions(dsd,Dim=5):
    ds=dsd.copy()
    if Dim==5:
        dd=''
        dm=1
    elif Dim==6:
        dd='2'
        dm=2
    ds['nu_R']=ds.apply(lambda row: get_solutions(row,solution='solution',Dim=Dim)
                        ,axis='columns')
    ds['DarkDirac']=ds['nu_R'].apply(lambda l: [ 'D→nu_R+{}fi+{}fj'.format(dd,dd) in d.keys() for d in l] 
                         ).apply(lambda l: [True for T in l if T] ).apply(len)
    ds['DarkMajor']=ds['nu_R'].apply(lambda l: [ 'D→nu_R+{}fi'.format(2*dm) in d.keys() for d in l] 
                         ).apply(lambda l: [True for T in l if T] ).apply(len)
    ds['XDirac']=ds['nu_R'].apply(lambda l: [ 'X→nu_R+{}fi+{}fj'.format(dd,dd) in d.keys() for d in l] 
                         ).apply(lambda l: [True for T in l if T] ).apply(len)
    ds['XMajor']=ds['nu_R'].apply(lambda l: [ 'X→nu_R+{}fi'.format(2*dm) in d.keys() for d in l] 
                         ).apply(lambda l: [True for T in l if T] ).apply(len)
    return ds
def filter_solution(dsd,nmax=32):
    tmp=dsd.copy()
    tmp['sltn']=tmp['solution'].apply(lambda s: repeated(np.asarray(s)))
    cl=tmp[( ( (tmp['solution'].apply(np.asarray).apply(np.abs).apply(np.max)<=nmax) & 
              (tmp['sltn'].apply(lambda d:list(d.keys())).apply(len)>0) 
             ) &
         ((tmp['DarkDirac']>0) | (tmp['DarkMajor']>0) | (tmp['XDirac']>0) | (tmp['XMajor']>0))
       )].reset_index(drop=True)
    cl['-n']=-cl['n']

    cl=cl.sort_values(['-n','DarkDirac','DarkMajor','XDirac','XMajor'],ascending=False
            ).reset_index(drop=True)
    cl=cl.drop('-n',axis='columns')
    return cl

def get_condition(nuR,rls,label):
    CONDITION=False
    if label=='D→nu_R+fi+fj':
        if len(rls)==2 and nuR:
            if sum( nuR+rls )==0:
                CONDITION=True
    if label=='D→nu_R+2fi+2fj':
        if len(rls)==2 and nuR:
            if sum( nuR+[2*r for r in rls] )==0:
                CONDITION=True
    if label=='D→nu_R+2fi':
        if len(rls)==1 and nuR:
            if sum( nuR+[2*r for r in rls] )==0:
                CONDITION=True
    if label=='D→nu_R+4fi':
        if len(rls)==1 and nuR:
            if sum( nuR+[4*r for r in rls] )==0:
                CONDITION=True
    #TODO → IMPLEMENT SOLUTIONS FOR X:            
    if label=='X→nu_R+fi+fj':
        CONDITION=True
    if label=='X→nu_R+2fi+2fj':
        CONDITION=True
    if label=='X→nu_R+2fi':
        CONDITION=True
    if label=='X→nu_R+4fi':
        CONDITION=True
        
    return CONDITION 

def get_nuR_i(nrow,i,model,label):
    """
    Get the nu_R member of the solution
    """
    if nrow[model]>0 and nrow['sltn'].get(i) and nrow['nu_R'] and len( [ d.get(label) for d in nrow['nu_R'] if d.get(label)] )>0:
        M=[m  for m in  nrow['nu_R'] if m.get(label)]
        nuR=[n for n in nrow['sltn'][i] for m in M  if n in m.get(label)  ]
        nuR=list(np.unique(nuR))
        NUR=[]
        for i in range(len(nuR)):
            nu_R=nuR[i:i+1]
            try:
                #Get the proper solution
                sltn=[d.get(label) for d in M if nu_R in d.get(label)][0]
                rls=[rl for rl in sltn if rl not in nu_R]
            except:
                rls=[]
            condition=get_condition(nu_R,rls,label)
            if condition:
                NUR=NUR+nu_R
        
        if NUR:
            return NUR
        else:
            return []
    else:
        return []
    

def get_nuR(row,model='DarkDirac',Dim=5):
    '''
    Get nu_R for a model
    The main function is get_nbu
    '''
    nrow=row.copy() 
    if Dim==5:
        dd=''
        dm=1
    elif Dim==6:
        dd='2'
        dm=2    
    if model=='DarkDirac':
        label='D→nu_R+{}fi+{}fj'.format(dd,dd)
    elif model=='DarkMajor':
        label='D→nu_R+{}fi'.format(2*dm)
    elif model=='XDirac':
        label='X→nu_R+{}fi+{}fj'.format(dd,dd)
    else:
        label='X→nu_R+{}fi'.format(2*dm)
        
    if model=='XDirac' or model=='XMajor':
        if nrow[model]>0:
            if nrow['sltn'].get(3)  and len( nrow['sltn'].get(3)  ) <= 1:
                nrow['sltn']=dict( ((k,v) for k,v in nrow['sltn'].items() if k!=3) )
                i=2
                nuR=get_nuR_i(nrow,i,model,label)
            elif nrow['sltn'].get(3)  and len( nrow['sltn'].get(3)  ) == 2:
                nu_R=[]
                x,y=nrow['sltn'].get(3)
                l=[ d.get(label) for d in nrow['nu_R'] if label in d.keys()  ]
                l=list(itertools.chain(*l))
                if x in l: #remove y
                     nu_R.append(x)
                if y in l: #remove y
                     nu_R.append(y)
                return nu_R        
                #print('WARNING: two triplets in X solution not yet implemented in X solutions: {}'.format(nrow['solution']))
            elif nrow['sltn'].get(3)  and len( nrow['sltn'].get(3)  ) > 2:
                #Remove not nu_R
                #Check nu_R in solution
                print( dict( ((k,v) for k,v in nrow['sltn'].items() if k!=3) ) )
                print('WARNING: More than one triplet in X solution not yet implemented in X solutions: {}'.format(nrow['solution']))
        
    # model=='DDirac' or model=='DMajor'
    else:
        i=3
        nuR=get_nuR_i(nrow,i,model,label)
        if nuR:
            return nuR
        else:
            i=4
            nuR=get_nuR_i(nrow,i,model,label)
            if nuR:
                return nuR

    i=2
    nuR=get_nuR_i(nrow,i,model,label)
    if nuR:
        return nuR    
    else: 
        return 0

def extract(l,x,νmax=5):
    for i in range(νmax):
        try:
            l.remove(x)
        except:
            break
    return l

def remove_list(l,pr):
    for x in pr:
        try:
            l=extract(l,x)
        except:
            pass
    return l

def remove(l,pr):
    if pr:
        pr=np.concatenate(pr).ravel()
    l=remove_list(l,pr)
    return l

def get_massless_fermions(l,ν,m=0,Dim=5):
    '''
    Zero determinant check implemented.
    '''
    if Dim==5:
        d=1
    elif Dim==6:
        d=2
    s=-(ν+m)//d
    l=extract(l,ν)
    lold=l.copy()
    if m:
        l=extract(l,m)
    #Dirac masses
    det0=0
    pr=[ p for  p in get_permutations( l,n=2  ) if np.append( p,s).sum()==0 or np.append( p,-s).sum()==0 ]
    #print(pr,s)
    for i in range(len(pr)):
        if pr and sum( [ lold.count(x) for x in pr[i] ] )%2!=0:
            det0=det0+1
        l=remove(l,pr)
        if det0!=0:
            l.append(-1000)

    #Majorana masses
    pr=[ll for ll in l if 2*ll+s==0 or 2*ll-s==0]
    l=remove_list(l,pr)
    #print(pr,s)
    #If repeated an extra S' gives at least degenerate Majorana mass
    return list(np.unique(l))
    #return l


def count_massless_fermions(l,ν,m=0,Dim=5):
    return len( get_massless_fermions(l,ν,m,Dim=Dim) )

def get_label(model,Dim=5):
    if Dim==5:
        dd=''
        dm=1
    elif Dim==6:
        dd='2'
        dm=2    
    if model=='DarkDirac':
        label='D→nu_R+{}fi+{}fj'.format(dd,dd)
    elif model=='DarkMajor':
        label='D→nu_R+{}fi'.format(2*dm)
    elif model=='XDirac':
        label='X→nu_R+{}fi+{}fj'.format(dd,dd)
    else:
        label='X→nu_R+{}fi'.format(2*dm)
    return model,label

def sltn_intersection(row,model='DarkDirac',Dim=5):
    model,label=get_label(model,Dim=Dim)
    sltn_int=0
    l=list(itertools.chain(*row['sltn'].values() ) )
    #print(l,label)
    for d in row['nu_R']:
        if d.get(label):
            #print(d)
            intrs=len(np.intersect1d(d.get(label), l ))
            if intrs>sltn_int:
                sltn_int=intrs
    return sltn_int

def massfer(row,label='DarkDirac',nmax=10,Dim=5):        
    # 
    if label=='DarkDirac' or label=='DarkMajor':
        if label=='DarkDirac':
            nuR='DD'
        else:
            nuR='DM'
        if row.get(label)>0:
            cdd=[]
            for ν in row.get(nuR):
                l=row.get('solution').copy()
                cdd.append( count_massless_fermions(l , ν,Dim=Dim)  )
            return min(cdd)
        else:
            return nmax
    if label=='XDirac' or label=='XMajor':
        if label=='XDirac':
            nuR='XD'
        else:
            nuR='XM'
        if row.get(label)>0:
            cddmax=100
            for m in row['sltn'].get(3):
                cdp=[]
                for ν in row.get(nuR):
                    l=row.get('solution').copy()
                    l=[ll for ll in l if ll!=m]
                    cdp.append( count_massless_fermions(l , ν,m,Dim=Dim)  )
                if min(cdp)<cddmax:
                    cdd=cdp
            return min(cdd)
        else:
            return nmax
        
def get_permutations(l,n=3):
    prmts=[]
    for p in itertools. permutations(l, n):
        if sorted(p) not in prmts:
            prmts.append(sorted(p))
    return [ np.asarray(p) for p in prmts if isinstance(p,list) ]
def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x
