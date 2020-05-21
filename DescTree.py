#!/usr/bin/env python
__title__   =  'Title'
__author__  =  'Jean-Pierre Stander'
__contact__ =  'jeanpierre.stander@gmail.com'
__date__    =  '26/07/2018'

#% Load Packages
import os
import talib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
os.chdir(r'C:\Users\jeanp\Desktop\Universiteit\CoinPred')
#% 
def ReturnClass(x):
  x = np.array(x).reshape(-1,1)
  cutoffs = np.array([-0.11,-0.09,-0.07,-0.05,-0.03,-0.01,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.01,0.03,0.05,0.07,0.09,0.11]).reshape(1,-1)
  clas = np.array([10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10])
  return np.array([clas[int(a)] for a in np.sum(x<cutoffs,axis=1)]).reshape(x.shape)


BTC = pd.read_csv('Aug4.csv',header=None)
BTC = BTC.iloc[1000:1100,:]
Test = BTC.iloc[1100:1200,:]
Test = np.array(Test)
BTC = np.array(BTC)
Returns = (BTC[1:,1]-BTC[:-1,1])/BTC[:-1,1]
TestReturns = (Test[1:,1]-Test[:-1,1])/Test[:-1,1]
ReturnClassesTest = ReturnClass(TestReturns)
ReturnClasses = ReturnClass(Returns)
inputs = {'close':np.array(BTC[:-1,1],dtype=np.float64)}
tests = {'close':np.array(Test[:-1,1],dtype=np.float64)}


ListOfGroups = ['Cycle Indicators', 'Pattern Recognition', 'Momentum Indicators', 
                'Overlap Studies', 'Volatility Indicators']

Indicators = pd.DataFrame()
Indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(inputs['close'])
Indicators['HT_DCPHASE'] = talib.HT_DCPHASE(inputs['close'])
Indicators['HT_PHASOR1'] = talib.HT_PHASOR(inputs['close'])[0]
Indicators['HT_PHASOR2'] = talib.HT_PHASOR(inputs['close'])[1]
Indicators['HT_SINE1'] = talib.HT_SINE(inputs['close'])[0]
Indicators['HT_SINE2'] = talib.HT_SINE(inputs['close'])[1]
Indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(inputs['close'])
Indicators = Indicators.iloc[63:,:]
ReturnClasses = ReturnClasses[63:,:]
Indicatorst = pd.DataFrame()
Indicatorst['HT_DCPERIOD'] = talib.HT_DCPERIOD(tests['close'])
Indicatorst['HT_DCPHASE'] = talib.HT_DCPHASE(tests['close'])
Indicatorst['HT_PHASOR1'] = talib.HT_PHASOR(tests['close'])[0]
Indicatorst['HT_PHASOR2'] = talib.HT_PHASOR(tests['close'])[1]
Indicatorst['HT_SINE1'] = talib.HT_SINE(tests['close'])[0]
Indicatorst['HT_SINE2'] = talib.HT_SINE(tests['close'])[1]
Indicatorst['HT_TRENDMODE'] = talib.HT_TRENDMODE(tests['close'])
Indicatorst = Indicatorst.iloc[63:,:]
ReturnClassTest = ReturnClassesTest[63:,:]
from sklearn import tree
DT = tree.DecisionTreeClassifier()
DT.fit(Indicators,ReturnClasses)

yhat = DT.predict(Indicatorst)



CylceInds = ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE']
#ListOfFunc = talib.get_functions()
ListOfFunc = []
[ListOfFunc.append(talib.get_function_groups()[a]) for a in ListOfGroups]
ListOfFunc = [val for sublist in ListOfFunc for val in sublist]

from talib.abstract import *

a = ['SMA','BBANDS','STOCH']
