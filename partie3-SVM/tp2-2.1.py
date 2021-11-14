import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
rdforest = RandomForestClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, rdforest, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression', 'SVM']

# lst_classif = [dummycl, gmb, dectree, rdforest, logreg]
# lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Random Forest', 'Logistic regression']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))



data_test = pd.read_csv('data/bank.test.csv', sep=';')

# print nb of instances and features
print(data_test.shape)

# print feature types
print(data_test.dtypes)

data_train = pd.read_csv('data/bank.train.csv', sep=';')

# print nb of instances and features
print(data_train.shape)

# print feature types
print(data_train.dtypes)


        
# Replace missing values by mean and scale numeric values
data_num = data_train.select_dtypes(include=['float64', 'int64'])
simple_imputer = SimpleImputer(strategy='median')
data_num_median = simple_imputer.fit_transform(data_num)

standard_scaler = StandardScaler()
data_num_scaler = standard_scaler.fit_transform(data_num_median)
print(data_num_scaler)

# la classe à prédire
classs_predict = data_train['y']

# accuracy_score(lst_classif,lst_classif_names,data_num_scaler,classs_predict)
#
# confusion_matrix(lst_classif,lst_classif_names,data_num_scaler,classs_predict)

data_columns_names = data_num.columns.values
ros = RandomOverSampler(sampling_strategy=0.5)
data_resample, data_labels_resample = ros.fit_resample(data_num, data_columns_names)
print(data_resample)

accuracy_score(lst_classif,lst_classif_names,data_resample,classs_predict)

confusion_matrix(lst_classif,lst_classif_names,data_resample,classs_predict)

# # Replace missing values by mean and discretize categorical values
# data_cat = data_train.select_dtypes(exclude='float64').drop('class',axis=1)
#
#
# # Disjonction with OneHotEncoder
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(handle_unknown="ignore")
# # encoder.fit(X_cat)
# # X_cat = encoder.transform(X_cat).toarray()


