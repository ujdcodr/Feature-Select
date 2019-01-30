import pandas as pd
import numpy as np
import time
from sklearn import preprocessing

np.set_printoptions(suppress=True)

df = pd.read_csv('kddcup.data_10_percent_corrected')

#print(df)
from skfeature.function.statistical_based import gini_index

df.flag = pd.Categorical(df.flag)
df['flag'] = df.flag.cat.codes

df.protocol_type = pd.Categorical(df.protocol_type)
df['protocol_type'] = df.protocol_type.cat.codes

df.service = pd.Categorical(df.service)
df['service'] = df.service.cat.codes

df.attack = pd.Categorical(df.attack)
df['attack'] = df.attack.cat.codes

#df['attack'] = (df['attack'] == 'normal').astype(int)
target = df['attack']
df = df.drop(['attack'],axis=1)

X = df.as_matrix()
y = target.as_matrix()
print(X)
print (y)

scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(X)

start = time.time()
a=gini_index.gini_index(robust_scaled_df,y)
print(gini_index.feature_ranking(a))
print(time.time()-start)


