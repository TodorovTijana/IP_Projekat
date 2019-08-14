import pandas as pd
from sklearn.naive_bayes import  GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
import numpy as np
import sklearn.preprocessing as prep

df = pd.read_csv("SPAM.csv")

booleandf = df.select_dtypes(include=[bool])
booleanDictionary = {True: "tacno", False: "netacno"}
for column in booleandf:
    df[column] = df[column].map(booleanDictionary)

features1 = df.columns[0]
features5 = df.columns[4]
features9 = df.columns[8]
features10 = df.columns[9]
features12 = df.columns[11]
features16 = df.columns[15]
features17 = df.columns[16]
features19 = df.columns[18]
features26 = df.columns[25]

features = [features5, features9, features10, features12, features16, features17, features19, features26]
x_original = df[features]
x = pd.DataFrame(prep.MinMaxScaler().fit_transform(x_original))
x.columns = features
y = df["spam"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

print("GaussianNB")

clf_gnb = GaussianNB()
clf_gnb.fit(x_train, y_train)

y_pred = clf_gnb.predict(x_test)

cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Matrica konfuzije", cnf_matrix, sep="\n")
print("\n")

accuracy = met.accuracy_score(y_test, y_pred)
print("Preciznost", accuracy)
print("\n")

class_report = met.classification_report(y_test, y_pred, target_names=df["spam"].unique())
print("Izvestaj klasifikacije", class_report, sep="\n")