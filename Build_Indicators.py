#!/usr/bin/env python
__title__   =  'Title'
__author__  =  'Jean-Pierre Stander'
__contact__ =  'jeanpierre.stander@gmail.com'
__date__    =  '26/07/2018'

#% Load Packages
import os
import re
import talib
import pandas as pd
import numpy as np

os.chdir(r'C:\Users\jeanp\Desktop\Universiteit\CoinPred\JP')
#% 
def ReturnClass(x):
  x = np.array(x).reshape(-1,1)
  cutoffs = np.array([-0.11,-0.09,-0.07,-0.05,-0.03,-0.01,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.01,0.03,0.05,0.07,0.09,0.11]).reshape(1,-1)
  clas = np.array([10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
  return np.array([clas[int(a)] for a in np.sum(x<cutoffs,axis=1)]).reshape(x.shape)

def get_builtin_args(obj):
   docstr = obj.__doc__
   args = []
   
   if docstr:
      items = docstr.split('\n')
      if items:
         func_descr = items[0]
         s = func_descr.replace(obj.__name__,'')
         idx1 = s.find('(')
         idx2 = s.find(')',idx1)
         if idx1 != -1 and idx2 != -1 and (idx2>idx1+1):
            full_sig_str = s[idx1+1:idx2]
            required_args_str = re.sub(r'(\[.*?\])', '', full_sig_str)
            args = [a.strip() for a in required_args_str.split(',')]
            return args
   raise Exception('Could not get arguments, no doc string.')

def BuildInd(df):
  ListOfFunc = talib.get_functions()
  ListOfFunc = [a for a in ListOfFunc if a not in ['MAVP','OBV']]

  my_special_args = {'low':df['LOW'],'high':df['HIGH'],'open':df['OPEN'],'close':df['CLOSE'],'volume':df['VOLUME']}
  Nums=0
  Inds = pd.DataFrame()
  for func in ListOfFunc:
     func_args = get_builtin_args(getattr (talib,func))
     if isinstance(func_args, list):
       if 'real0' not in func_args: 
         try:
           kwargs = {k: v for k, v in my_special_args.items() if k in func_args}
           tempfunc = getattr (talib,func)
           res = tempfunc(**kwargs)
           Nums += 1
           if len(res)>10:
             Inds[func] = res         
           else:
    #       if len(res)<10:
             for i in range(len(res)):
               Inds[func+str(i)] = res[i]
               Nums += 1  
         except:
           tempfunc = getattr (talib,func)
           res = tempfunc(my_special_args['close'])
           Nums += 1
           if len(res)>10:
             Inds[func] = res
    #       if len(res)<10:
           else:
             for i in range(len(res)):
               Inds[func+str(i)] = res[i]
               Nums += 1
  Inds = Inds.fillna(method='backfill')
  dell = []
  for i in range(Inds.shape[1]):
    if Inds.iloc[:,i].unique().shape[0]<2:
        dell.append(list(Inds)[i])
  Inds = Inds.drop(columns=dell)
        
  return Inds
  

BTC = pd.read_csv('Candles.csv')
BTC = BTC.drop(columns=['Unnamed: 0'])
BTC = BTC.rename(columns={a:a[9:] for a in list(BTC)})


Indic = BuildInd(BTC)  
  