import os
import re
import talib
import pandas as pd
import numpy as np
os.chdir(r'C:\Users\jeanp\Desktop\Universiteit\CoinPred\JP')

def jitter(df, cols = 'All',replications = 10,lamb=1):
    if cols == 'All':
        cols = df.columns
    n = df.shape[0]
    k = len(cols)
    diag_sigma = np.sqrt(lamb)*np.diag(df[cols].std(axis = 0, skipna = True).fillna(1).values)
    df_out = pd.DataFrame({})
    for i in range(replications):
        print(round(100*i/replications))
        d = df.copy()
        d.loc[:,cols] = d[cols].values + np.random.multivariate_normal(np.repeat(0,k), diag_sigma, n)
        df_out = pd.concat([df_out,d])
    df_out = df_out.reset_index(drop=True)
    return df_out
    
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
BTC = BTC.drop(columns='Unnamed: 0')
BTC = BTC.rename(columns={a:a[9:].upper() for a in list(BTC)})
BTC = BTC.drop(columns='MTS')

Indic = BuildInd(BTC)
Indic['Return'] = BTC['CLOSE'].iloc[:-1]/BTC['CLOSE'].shift(-1)
Indic['LogReturn'] = np.log(Indic['Return'])

Train = Indic.iloc[:-100,:]
Test = Indic.iloc[-100:,:]
#Train = jitter(Train,['LogReturn'],30,1.1)

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

x = [a for a in list(Indic) if 'Return' not in a]
y = 'LogReturn'
h2o.init(nthreads = 7, max_mem_size = '1g')

train =  h2o.H2OFrame(Train)
valid = h2o.H2OFrame(Test)

DeepLearn = H2ODeepLearningEstimator(model_id='DeepLearn',
                               hidden=[10,20,20,10],
                               epochs=30,
                               seed=1111,
                               nfolds=5,
                               stopping_rounds=0,
                               keep_cross_validation_predictions = True)
DeepLearn.train(x=x, y=y, training_frame=train)
# Eval performance:
DLperf = DeepLearn.model_performance()

RandomForest = H2ORandomForestEstimator(model_id='RandomForest',
                                        ntrees=10,
                                        max_depth=5,
                                        min_rows=10,
                                        seed=1111,
                                        nfolds=5,
                                        binomial_double_trees=True,
                                        keep_cross_validation_predictions=True)
RandomForest.train(x=x, y=y, training_frame=train)
# Eval performance:
RFperf = RandomForest.model_performance()

GradientBoost = H2OGradientBoostingEstimator(model_id = 'GradientBoost',
                                             nfolds=5,
                                             seed=1111,
                                             keep_cross_validation_predictions=True)
GradientBoost.train(x=x, y=y, training_frame=train)
GBperf = GradientBoost.model_performance()


Ensemble = H2OStackedEnsembleEstimator(model_id="Ensemble",
                                       base_models=['DeepLearn', 'RandomForest',
                                                    'GradientBoost'])
Ensemble.train(x=x, y=y, training_frame=train)

Performance = Ensemble.model_performance()


predic = Ensemble.predict(valid).as_data_frame()
yhat = np.array(predic).reshape(-1,1)
ytrue = np.array(Test['LogReturn']).reshape(-1,1)
yy = np.concatenate((np.exp(yhat),np.exp(ytrue)),axis=1)
yy = yy[:99,:]
R2 = np.corrcoef(yy.T)[0,1]**2

h2o.cluster().shutdown()


