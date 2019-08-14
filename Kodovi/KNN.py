import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
import sklearn.metrics as met


def class_info(clf_arg, x_train_arg, y_train_arg, x_test_arg, y_test_arg):

    clf.fit(x_train_arg, y_train_arg)
    distances, indices = clf.kneighbors(x_test_arg)

    y_pred = clf.predict(x_test_arg)

    cnf_matrix = met.confusion_matrix(y_test_arg, y_pred)
    print("Matrica konfuzije", cnf_matrix, sep="\n")

    accuracy = met.accuracy_score(y_test_arg, y_pred)
    print("Preciznost", accuracy)
    class_report = met.classification_report(y_test_arg, y_pred, target_names=clf.classes_)
    print("Izvestaj klasifikacije", class_report, sep="\n")

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


k_values = range(3,10)
p_values = [1, 2]
weights_values = ['uniform', 'distance']

for k in k_values:
    for p in p_values:
        for weight in weights_values:
            clf = KNeighborsClassifier(n_neighbors=k,
                                        p=p,
                                        weights=weight)

            print("k="+ str(k))
            print("p="+str(p))
            print("weight=" + weight)

            class_info(clf, x_train, y_train, x_test, y_test)