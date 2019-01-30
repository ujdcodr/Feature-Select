from sklearn.naive_bayes import GaussianNB
import pandas as pd
df = pd.read_csv('svm0p001.csv')
final_X = df.as_matrix()[:, 1:-1]
final_y = df.as_matrix()[:,(-1)]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(final_X, final_y, test_size=0.2, random_state=0)

print(X_train.shape, Y_train.shape, X_test.shape)

gnb = GaussianNB()
y_pred = gnb.fit(X_train.shape, Y_train.shape).predict(X_test.shape)
print(gnb.fit(X_train.shape, Y_train.shape))
