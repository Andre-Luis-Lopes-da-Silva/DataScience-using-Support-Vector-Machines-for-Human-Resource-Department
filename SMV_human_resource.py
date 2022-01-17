import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/andre/Desktop/WA_Fn-UseC_-HR-Employee-Attrition.csv')  

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis = 1, inplace=True)

previsores1 = df.drop(['Attrition'], axis = 1)
previsores = previsores1.iloc[:,0:30].values

classe = df.iloc[:,1].values 

# Columns that possess categorical attributes
categorical_attributes = df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

# Convert categorical attributes to dummy variables
from sklearn.preprocessing import OneHotEncoder  
onehotencoder = OneHotEncoder()
previsores_dummy = onehotencoder.fit_transform(categorical_attributes).toarray()
categorical_attributes = pd.DataFrame(previsores_dummy)

numerical = df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
previsores = pd.concat([categorical_attributes, numerical], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Support Vector Machines Classifier
from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf') # 'rbf' = radial basis function, is useful for non-linear hyperplane. 

#Train the model using the training sets
model_SVC.fit(X_train, y_train)
y_pred_svm = model_SVC.decision_function(X_test)

# Metrics
from sklearn import metrics
y_pred = model_SVC.predict(X_test) 

print('Metrics of this algorithm:')
print('Accuracy: {:.2}'.format(metrics.accuracy_score(y_test, y_pred)))

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print('Recall: {:.2}'.format(metrics.recall_score(y_test, y_pred)))
print('Precision: {:.2}'.format(metrics.precision_score(y_test, y_pred)))  
print('F1 Score: {:.2}'.format(metrics.f1_score(y_test, y_pred, average='macro')))  
print(metrics.classification_report(y_test, y_pred))

# Plotting ROC and AUC
from sklearn.metrics import roc_curve, auc

svm_fp, svm_tp, threshold = roc_curve(y_test, y_pred_svm)
auc_svm = auc(svm_fp, svm_tp)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fp, svm_tp, label='auc = %0.3f' % auc_svm)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()
plt.show()
