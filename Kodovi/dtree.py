import pandas as pd
from sklearn.tree import  DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.preprocessing as prep
import subprocess
import sklearn.metrics as met
from sklearn import tree

def calculate_metrics(part, true_values, predicted_values):

    print('Skup ', part)
    print('Matrica konfuzije')
    cnf_matrix = met.confusion_matrix(true_values, predicted_values)
    df_cnf_matrix = pd.DataFrame(cnf_matrix, index=dt.classes_, columns=dt.classes_)
    print(df_cnf_matrix)
    accuracy = met.accuracy_score(true_values, predicted_values)
    print('Preciznost', accuracy)

    print('Preciznost po klasama', met.precision_score(true_values, predicted_values, average=None))

    print('Odziv po klasama', met.recall_score(true_values, predicted_values, average=None))

    class_report = met.classification_report(true_values, predicted_values)
    print('Izvestaj klasifikacije', class_report, sep='\n')

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
x=pd.DataFrame(prep.MinMaxScaler().fit_transform(x_original))

x=df[features]
x.columns = features
y=df["spam"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)

dt = DecisionTreeClassifier(max_depth=7)
dt.fit(x_train, y_train)

print('Klase', dt.classes_)
print('feature_importances_', '\n', pd.Series(dt.feature_importances_, index=features))

print('Predvidjena verovatnoca', dt.predict_proba(x_test), sep='\n')
print('\n\n')

y_pred = dt.predict(x_train)
calculate_metrics('Trening ',y_train, y_pred )

y_pred = dt.predict(x_test)
calculate_metrics('Test',y_test, y_pred )

