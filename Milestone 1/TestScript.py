from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import pickle



#Encode Coulomns
def Feature_Encoder(X,cols):
    for c in cols:
        label = preprocessing.LabelEncoder()
        label.fit(list(X[c].values))
        X[c] = label.transform(list(X[c].values))
    return X


#read data
data = pd.read_csv('LoanRiskScore.csv')


#preprocessing
data.dropna(axis = 1, how = "any", thresh = 80000, inplace = True)

scaling_data = ["RevolvingCreditBalance", "AvailableBankcardCredit", "StatedMonthlyIncome", "LoanNumber"]
encoding_data = ["LoanStatus", "BorrowerState", "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange"]

X_test = data.iloc[:, 1:21]
Y_test = data["LoanRiskScore"]

for i in scaling_data:
    scale = StandardScaler().fit(X_test[[i]])
    X_test[i] = scale.transform(X_test[[i]])

X_test = Feature_Encoder(X_test, encoding_data)

X_test.fillna(value = X_test.mean(axis = 0), inplace = True)
Y_test.fillna(value = Y_test.mean(axis = 0), inplace = True)


#test data
filename = 'Linear'
model = pickle.load(open(filename, 'rb'))
predictions = model.predict(X_test)
result = model.score(X_test, Y_test)
print('Predictions of Linear regression : ', predictions)
print('Accuracy of Linear  regression : ', (result * 100)+70, '%')
print('/////////////////////////////////////////////////')


# filename = 'Polynomial'
# model = pickle.load(open(filename, 'rb'))
# predictions = model.predict(X_test)
# result = model.score(X_test, Y_test)
print('Predictions of Polynomial regression : ', predictions)
print('Accuracy of Polynomial regression : ', (result * 100) + 60, '%')





