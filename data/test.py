import pandas as pd
import numpy as np
from sklearn import preprocessing
np.set_printoptions(suppress=True)

df = pd.read_csv('kddcup.data_10_percent_corrected')
print(df.shape)

col_names = df.columns.get_values()

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
#print(df)
scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(df)
select = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = pd.DataFrame(select.fit_transform(robust_scaled_df))
sel = select.get_support(indices=True)
cols = X_new.shape[1]
print sel

#SelectKbest

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selector = SelectKBest(f_classif, k=15)
X_new= pd.DataFrame(selector.fit_transform(robust_scaled_df, target))

sel = selector.get_support(indices=True)
cols = X_new.shape[1]

print sel


# #L1
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel
# lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(df, target)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(df)
# print(X_new)
# print(X_new.shape)
# cols = X_new.shape[1]
# sel = model.get_support(indices=True)

'''

#Tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=300)
clf = clf.fit(df, target)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(df)
print(X_new.shape)
cols = X_new.shape[1]
sel = model.get_support(indices=True)
print sel
print len(sel)

'''
names = [i for i in range(cols)]

df1 = pd.DataFrame(X_new)
df1.columns = names

print({i:sel[i] for i in range(cols)})
features = [names[sel[i]] for i in range(cols)]
print(features)

df1.columns = features
df1['attack'] = target
df1.to_csv("varthresh2.csv", sep='\t')
'''
# #Recursive
# from sklearn.feature_selection import RFE
# from sklearn.svm import SVR
# estimator = SVR(kernel="linear")
# selector = RFE(estimator, 5, step=1)
#
# from sklearn.feature_selection import RFECV
# selector = RFECV(estimator, step=1, cv=5)
# selector = selector.fit(df, target)
# print(selector.support_)
# print(selector.ranking_)
