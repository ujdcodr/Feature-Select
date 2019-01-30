import pandas as pd
import numpy as np
import time
np.set_printoptions(suppress=True)

df = pd.read_csv('fullreducedL1.csv')
'''
#print(df)
df.flag = pd.Categorical(df.flag)
df['flag'] = df.flag.cat.codes

df.protocol_type = pd.Categorical(df.protocol_type)
df['protocol_type'] = df.protocol_type.cat.codes

df.service = pd.Categorical(df.service)
df['service'] = df.service.cat.codes
df.attack = pd.Categorical(df.attack)
df['attack'] = df.attack.cat.codes
'''
print df.columns
target = df['attack']
df = df.drop(['attack'],axis=1)


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

clf = MultinomialNB()

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(clf, df, target, cv=6)
print 'Cross-validated scores:', scores
predictions = cross_val_predict(clf, df, target, cv=6)
accuracy = metrics.accuracy_score(target, predictions)
print 'Cross-Predicted Accuracy:', accuracy

