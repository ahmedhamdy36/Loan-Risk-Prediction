import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import time
from sklearn.preprocessing import PolynomialFeatures
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


#scaling & Encoding
scaling_data = ["RevolvingCreditBalance", "AvailableBankcardCredit", "StatedMonthlyIncome", "LoanNumber"]
encoding_data = ["LoanStatus", "BorrowerState", "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange"]

X = data.iloc[:, 1:21]
Y = data["LoanRiskScore"]

for i in scaling_data:
    scale = StandardScaler().fit(X[[i]])
    X[i] = scale.transform(X[[i]])

X = Feature_Encoder(X, encoding_data)

X.fillna(value = X.mean(axis = 0), inplace = True)
Y.fillna(value = Y.mean(axis = 0), inplace = True)


#split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle = True)


#training model
poly = PolynomialFeatures(degree = 2)
train_poly = poly.fit_transform(X_train)
test_poly = poly.fit_transform(X_test)
poly_model = linear_model.LinearRegression()

start_poly = time.time()
poly_model.fit(train_poly, y_train)
end_poly = time.time()
training_time_poly = end_poly - start_poly

y_train_predicted = poly_model.predict(train_poly)
prediction = poly_model.predict(test_poly)


#print data
print("Co-efficient of linear regression : ", poly_model.coef_)
print("/////////////////////////////////")
print("Intercept of linear regression model : ", poly_model.intercept_)
print("/////////////////////////////////")
print("Training time: ", training_time_poly)
print("/////////////////////////////////")
print("MSE for testing: ", metrics.mean_squared_error(y_test, prediction))
print("/////////////////////////////////")
print("MSE for training: ", metrics.mean_squared_error(y_train, y_train_predicted))
print("/////////////////////////////////")
print("Accuracy: ", (poly_model.score(test_poly, y_test)) * 100, "%")


#pickle
filename = 'Polynomial'
pickle.dump(poly, open(filename, 'wb'))





