import pandas as pd
df = pd.read_csv('tree25.csv')
print(df)
final_X = df.as_matrix()[:, 1:-1]
final_y = df.as_matrix()[:,(-1)]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(final_X, final_y, test_size=0.2, random_state=0)

print(X_train.shape, Y_train.shape, X_test.shape)

from sklearn.svm import OneClassSVM

clf_oneclass = OneClassSVM(kernel='rbf')
clf_oneclass.fit(X_train, Y_train)

y_pred = clf_oneclass.predict(X_test)
# for num in range(len(y_pred)):
#     if (y_pred[num] == -1):
#         y_pred[num] = 0
#
# correct = 0
# total = 0
#
# for i in range(len(y_pred)):
#     if y_pred[i] == 1:
#         total += 1
#     if (y_pred[i] == Y_test[i] and y_pred[i] == 1):
#         correct += 1
# print("Precision: ", correct / total)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(Y_test, y_pred)

print('Average precision-Recall score: {0:0.2f}'.format(average_precision))


from sklearn.metrics import recall_score
average_precision = recall_score(Y_test, y_pred,average='weighted')

print('Average precision-Recall score: {0:0.2f}'.format(average_precision))
