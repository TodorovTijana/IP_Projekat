import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import sklearn.metrics as met
from termcolor import colored
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

scaler = preprocessing.StandardScaler().fit(x)
x =pd.DataFrame(scaler.transform(x))
x.columns = features
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y)
params = [{'solver':['sgd'],
           'learning_rate':['constant', 'invscaling', 'adaptive'],
           'learning_rate_init':[0.01, 0.005, 0.002, 0.001],
            'activation' : ['identity', 'logistic', 'tanh', 'relu' ],
            'hidden_layer_sizes' : [(10,3), (10,10),],
           'max_iter': [500]
           }]
clf = GridSearchCV(MLPClassifier(), params, cv=5)
clf.fit(x_train, y_train)

print("Izvestaj za test skup:")
y_true, y_pred = y_test, clf.predict(x_test)
cnf_matrix = met.confusion_matrix(y_test, y_pred)
print("Matrica konfuzije", cnf_matrix, sep="\n")
print("\n")

accuracy = met.accuracy_score(y_test, y_pred)
print("Preciznost", accuracy)
print("\n")

class_report = met.classification_report(y_test, y_pred, target_names=clf.classes_)
print("Izvestaj klasifikacije", class_report, sep="\n")

print('Broj iteracija: ', clf.best_estimator_.n_iter_)
print('Broj slojeva: ', clf.best_estimator_.n_layers_)
