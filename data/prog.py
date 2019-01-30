import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

df = pd.read_csv('kddcup.data_10_percent_corrected')

#print(df)


df.flag = pd.Categorical(df.flag)
df['flag'] = df.flag.cat.codes

df.protocol_type = pd.Categorical(df.protocol_type)
df['protocol_type'] = df.protocol_type.cat.codes

df.service = pd.Categorical(df.service)
df['service'] = df.service.cat.codes

df.attack = pd.Categorical(df.attack)
df['attack'] = df.attack.cat.codes

target = df['attack']
df = df.drop(['attack'],axis=1)

'''
#VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
x = pd.DataFrame(sel.fit_transform(df))
print (sel.get_support(indices=True))
'''

'''
#SelectKbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selector = SelectKBest(chi2, k=15)
selector.fit(df, target)

print(selector.get_support(indices=True))
'''

'''
#L1
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(df, target)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(df)
print(X_new.shape)
print(model.get_support(indices=True))
'''

'''
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(df, target)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
'''


#Tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=20)
clf = clf.fit(df, target)
#print(clf.feature_importances_)  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(df)
print(X_new.shape)  
print(model.get_support(indices=True))





