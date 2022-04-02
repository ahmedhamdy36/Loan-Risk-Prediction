import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.svm import SVC



#Encode Coulomns
def Feature_Encoder(X, cols):
    for c in cols:
        label = preprocessing.LabelEncoder()
        label.fit(list(X[c].values))
        X[c] = label.transform(list(X[c].values))
    return X


#read data
data = pd.read_csv('LoanRiskClassification.csv')


#preprocessing
data.dropna(axis = 1, how = "any", thresh = 80000, inplace = True)
data.dropna(subset = ['ProsperRating (Alpha)'], axis = 0, how = 'any', inplace = True)

data.fillna(method = 'bfill', axis = 0, inplace = True)

X = data.iloc[:, 1:21]
Y = data["ProsperRating (Alpha)"]

scaling_data = ["RevolvingCreditBalance", "AvailableBankcardCredit", "StatedMonthlyIncome", "LoanNumber"]
encoding_data = ["LoanStatus", "BorrowerState", "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange"]

X = Feature_Encoder(X, encoding_data)
for i in scaling_data:
    scale = StandardScaler().fit(X[[i]])
    X[i] = scale.transform(X[[i]])


#Graph of feature gane
mutual_info = mutual_info_classif(X, Y)
info = pd.Series(mutual_info)
info.index = X.columns
info = info.sort_values(ascending = False)
info.plot(kind = 'bar', color = 'red')


#Feature Selection
X = []
X = data.iloc[:, 3:5]


# dataset split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True)


# Training
Training_time = []

s1 = time.time()
linear_kernel = SVC(kernel = 'linear', C = 0.1).fit(X_train, y_train)
e2 = time.time()
training_time1 = e2 - s1
Training_time.append(training_time1)

s2 = time.time()
poly_kernel = SVC(kernel = 'poly', degree = 3, C = 0.2).fit(X_train, y_train)
e2=time.time()
training_time2 = e2 - s2
Training_time.append(training_time2)

s3 = time.time()
rbf_kernel = SVC(kernel = 'rbf', gamma = 0.7, C = 0.3).fit(X_train, y_train)
e3 = time.time()
training_time3 = e3 - s3
Training_time.append(training_time3)


#Testing
Testing_time = []

s1 = time.time()
linear_pred = linear_kernel.predict(X_test)
e1 = time.time()
testing_time1 = e1 - s1
Testing_time.append(testing_time1)

s2 = time.time()
linear_k_pred = poly_kernel.predict(X_test)
e2 = time.time()
testing_time2 = e2 - s2
Testing_time.append(testing_time2)

s3 = time.time()
poly_pred = rbf_kernel.predict(X_test)
e3 = time.time()
testing_time3=e3 - s3
Testing_time.append(testing_time3)


#Accuracy
Accuracy = []

linear_accuracy = accuracy_score(y_test, linear_pred)
Accuracy.append(linear_accuracy*100)

linear_k_accuracy = accuracy_score(y_test, linear_k_pred)
Accuracy.append(linear_k_accuracy*100)

poly_accuracy = accuracy_score(y_test, poly_pred)
Accuracy.append(poly_accuracy*100)


#print data
print('Accuracy of (Linear SVC Kernel) = ', (linear_accuracy * 100), '%')
print('Training time of (Linear SVC Kernel) = ', training_time1, 's')
print('Testing time of (Linear SVC Kernel) = ', testing_time1, 's')
print('#########################################################')
print('Accuracy of (poly SVC Kernel) = ', (linear_k_accuracy * 100), '%')
print('Training time of (poly SVC Kernel) = ', training_time2, 's')
print('Testing time of (poly SVC Kernel) = ', testing_time2, 's')
print('#########################################################')
print('Accuracy of (rbf SVC Kernel) = ', (poly_accuracy * 100), '%')
print('Training time of (rbf SVC Kernel) = ', training_time3, 's')
print('Testing time of (rbf SVC Kernel) = ', testing_time3, 's')


#data plot
labels=['Linear Kernel', 'Polynomial', 'RBF']
X_axis=np.arange(len(labels))
plt.bar(X_axis, Accuracy, 0.3, label = 'Accuracy', color = 'red')
plt.bar(X_axis+0.3 , Training_time, 0.3, label = 'Traning Time', color = 'green')
plt.bar(X_axis-0.3 , Testing_time, 0.3, label = 'Testing Time', color = 'blue')
plt.xticks(X_axis, labels)
plt.xlabel("Models")
plt.ylabel("Values")
plt.legend()
plt.show()


#pickle
filename = 'linear kernel'
pickle.dump(linear_kernel, open(filename, 'wb'))

filename = 'poly kernel'
pickle.dump(poly_kernel, open(filename, 'wb'))

filename = 'rbf kernel'
pickle.dump(rbf_kernel, open(filename, 'wb'))






