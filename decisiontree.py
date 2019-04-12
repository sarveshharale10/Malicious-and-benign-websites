import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

clf = LogisticRegression()
scaler = MinMaxScaler()
df = pd.read_csv("dataset-cleaned-train.csv")

X = df.iloc[:,:-1]

Y = df.iloc[:,-1]

clf.fit(X,Y)

df_test = pd.read_csv("dataset-cleaned-test.csv")
X_test = df_test.iloc[:,:-1]
Y_test = df_test.iloc[:,-1]
Y_pred = clf.predict(X_test)

count = 0
for i in range(len(Y_test)):
	if(Y_test[i] == Y_pred[i]):
		count += 1

accuracy = count / len(Y_test)
print(accuracy)

tn,fp,fn,tp = confusion_matrix(Y_test,Y_pred).ravel()
print("True Positives:",tp)
print("True Negatives:",tn)
print("False Positives:",fp)
print("False Negatives:",fn)
