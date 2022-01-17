# DataScience-using-Support-Vector-Machines-for-Human-Resource-Department
This study presents a predictive machine learning model using Support vector machines to classify the profile of employees who will leave the company.

Support vector machines so called as SVM is a supervised learning algorithm which can be used for classification and regression problems as support vector classification (SVC) and support vector regression (SVR). It is a non-probabilistic binary linear classifier.

This datascience solution was performed using dataset by “IBM HR Analytics Employee Attrition & Performance, Predict attrition of your valuable employees”. Available at: https:www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
Hiring a new employee involves costs. Therefore, the identifying of a candidate who has a profile that will not stay with the company will be useful to the company optimize your decisions.

After the clearing of the data and the data exploratory analysis (EDA), some insights were perfomed.
The columns: Attrition, OverTime and Over18 possessed “y” or “n” and they were replaced by “0” or “1”, because machine learning does not process variables like “string”.
Variables that have a unique attribute for each employee are not useful for analysis. Therefore, these variables were removed, as the columns: EmployeeCount, StandardHours, Over18 and EmployeeNumber.

The classes of this algorithm are class 1 (the employee will leave from company) and class 0 (the employee will stay in company). These will be the answers.

The metrics obtained in this study were: Accuracy: 0.87, Precision: 0.18, Recall: 0.58, F1 Score: 0.6 and AUC: 0.775. These metrics vary each time we run the code. This variation is not very expressive.
