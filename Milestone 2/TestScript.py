import pandas as pd
import pickle


#read data
data = pd.read_csv('LoanRiskTestForTAsClassification.csv')


#select features
X_test = data.iloc[:,4:6]
Y_test = data["ProsperRating (Alpha)"]

#preprocessing
X_test.fillna(value = X_test.mean(axis = 0), inplace = True)
Y_test.fillna(method='bfill', axis = 0, inplace = True)


#test data
filename = 'linear kernel'
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(X_test)
result = model.score(X_test, Y_test)
print('Predictions of Linear Kernel = ', predictions)
print('Accuracy of Linear Kernel = ', result * 100, '%')

filename = 'poly kernel'
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(X_test)
result = model.score(X_test, Y_test)
print('Predictions of Polynomial Kernel = ', predictions)
print('Accuracy of Polynomial Kernel = ', result * 100, '%')

filename = 'rbf kernel'
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(X_test)
result = model.score(X_test, Y_test)
print('Predictions of RBF Kernel = ', predictions)
print('Accuracy of RBF Kernel = ', result * 100, '%')








