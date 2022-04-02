import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics



#Encode Coulomns
def Feature_Encoder(X, cols):
    for c in cols:
        label = preprocessing.LabelEncoder()
        label.fit(list(X[c].values))
        X[c] = label.transform(list(X[c].values))
    return X


#read file
data = pd.read_csv('LoanRiskScore.csv')


#Corolation between features
plt.figure(figsize = (20, 13))
corolation = data.corr()
sns.heatmap(corolation, annot = True)
plt.show()


#drop colums
data.dropna(axis = 1, how = "any", thresh = 80000, inplace = True)
data.dropna(axis = 0, how = "any", inplace = True)


#scaling & Encoding
scaling_data = ["RevolvingCreditBalance", "AvailableBankcardCredit", "StatedMonthlyIncome", "LoanNumber"]
encoding_data = ["LoanStatus", "BorrowerState", "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange"]

X = data.iloc[:, 1:21]
Y = data["LoanRiskScore"]

for i in scaling_data:
    scale = StandardScaler().fit(X[[i]])
    X[i] = scale.transform(X[[i]])

X = Feature_Encoder(X, encoding_data)


#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)


#traing model
Linear_Model = linear_model.LinearRegression()
start_time = time.time()
Linear_Model.fit(X_train, y_train)
end_time = time.time()
y_train_predection = Linear_Model.predict(X_train)
y_predection = Linear_Model.predict(X_test)


#print data
print("Co-efficient of linear regression: ", Linear_Model.coef_)
print("/////////////////////////////////")
print("Intercept of linear regression model: ", Linear_Model.intercept_)
print("/////////////////////////////////")
print("Training time: ", end_time - start_time)
print("/////////////////////////////////")
print("MSE for testing: ", metrics.mean_squared_error(np.asarray(y_test), y_predection))
print("/////////////////////////////////")
print("MSE for training: ", metrics.mean_squared_error(np.asarray(y_train), y_train_predection))
print("/////////////////////////////////")
print("Accuracy: ", (Linear_Model.score(X_test, y_test)) * 100, "%")


#pickle
filename = 'Linear'
pickle.dump(Linear_Model, open(filename, 'wb'))





