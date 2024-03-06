import sklearn
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Download dataset and convert to dataframe
diabetes_data = datasets.load_diabetes()
df = pd.DataFrame(diabetes_data['data'], columns = diabetes_data['feature_names'])
df['Diabetes'] = pd.DataFrame(diabetes_data['target'])

#Data Processing

print(df.info())
#No non numerical values

#Fill potential missing values
for column in df.columns.values:
    df[column].fillna(df[column].median(), inplace=True)

#Replace outliers with median value
for column in df.columns.values:
    median = df[column].median()
    std_dev = df[column].std()
    outliers = (df[column] - median).abs() > 2 * std_dev
    df.loc[outliers, column] = median

#Get a correlation matrix and display to the user
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr)
plt.show()

#Get features with at least 20% correlation
correlation_threshold = 0.25
corr_target = abs(corr['Diabetes'])
relevant = []
for i in range(len(corr.columns)):
    if abs(corr.iloc[i, 10]) > correlation_threshold:
        name = corr.columns[i]
        relevant.append(name)

#Create new dataframe using these features by copying the old one and dropping any not in the relevant list
filtered = df.copy()
for column in df.columns.values:
    if column in relevant:
        break
    filtered = filtered.drop(column, axis = 1)
info = filtered.drop('Diabetes', axis = 1).values
target = filtered['Diabetes'].values

#Split the data
x_train, x_test, y_train, y_test = train_test_split(info, target, test_size = .2)

#GridSearch
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf', 'poly']}

grid_model = SVR()
grid = GridSearchCV(grid_model, param_grid, cv = 5, scoring = 'r2')
grid.fit(x_train, y_train)
best_params = grid.best_params_

#Make a new model and run the tests
model = SVR(**best_params)
model.fit(x_train, y_train)
results = model.predict(x_test)

#Results display
print("Mean Squared Error:", (mean_squared_error(y_test, results)))
print("Mean Absolute Error:", (mean_absolute_error(y_test, results)))
print("R-squared:", (r2_score(y_test, results)))
