#!/usr/bin/env python
# coding: utf-8
#THIS IS A PROGRAM MADE BY AND BEST USED IN JUPYTER NOTEBOOK
#EACH INSTANCE OF THIS -----> "# In[ ]:" IS AN INDICATION THAT IT IS SUPPOSED TO BE IN A SEPERATE JUPYTER NOTEBOOK BLOCK
# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from IPython.display import Image
import sklearn.linear_model as linear
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, roc_auc_score
#I'm kinda just importing everything, ignore most of it 


# In[ ]:


df = pd.read_csv("Customer-Churn-Records.csv")
df2 = df


# In[ ]:


print("Variables:", df.columns)


# In[ ]:


df


# In[ ]:

#Dropping any missing value datapoints(There are none)
df = df.dropna(subset=['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited',
       'Satisfaction Score', 'Card Type', 'Point Earned'])


# In[ ]:


df= df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 'columns')


# In[ ]:

#Making the catergorical variables into dummy variables
df = pd.get_dummies(df, drop_first=True)


# In[ ]:


df


# In[ ]:


y = df['Exited']
X = df.drop(['Exited'], axis = 'columns')
X


# In[ ]:

#Splitting the dataset into training and testing dataset
Xtrain, Xtest, ytrain, ytest= train_test_split(X,y,test_size=0.1, random_state=0)
Xtest


# In[ ]:


""" #This is the grid search used to find the right hyperparameters, the grid was manually changed until the best were found
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 6, 7]
}

model = RandomForestClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')

grid_search.fit(Xtrain, ytrain)

best_params = grid_search.best_params_
best_roc_auc = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best ROC AUC Score:", best_roc_auc)
"""


# In[ ]:

#Making the model
model = RandomForestClassifier(n_estimators=125,max_depth=6).fit(Xtrain, ytrain)


# In[ ]:


predictions = model.predict(Xtest)


# In[ ]:

#Calculating Accuracy Score and Precision Score
precision = precision_score(ytest, predictions)
accuracy = accuracy_score(ytest, predictions)

print("Precision:", precision)
print("Accuracy:", accuracy)


# In[ ]:


y_train_probs = model.predict_proba(Xtrain)[:, 1]
y_test_probs = model.predict_proba(Xtest)[:, 1]

roc_auc_train = roc_auc_score(ytrain, y_train_probs)
roc_auc_test = roc_auc_score(ytest, y_test_probs)

print("ROC AUC Score on Training Data:", roc_auc_train)
print("ROC AUC Score on Testing Data:", roc_auc_test)
y_train_probs = model.predict_proba(Xtrain)[:, 1]
y_test_probs = model.predict_proba(Xtest)[:, 1]

# Calculating ROC curve for training data
fpr_train, tpr_train, thresholds_train = roc_curve(ytrain, y_train_probs)
roc_auc_train = roc_auc_score(ytrain, y_train_probs)

# Calculating ROC curve for testing data
fpr_test, tpr_test, thresholds_test = roc_curve(ytest, y_test_probs)
roc_auc_test = roc_auc_score(ytest, y_test_probs)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Training ROC Curve (AUC = {roc_auc_train:.2f})')
plt.plot(fpr_test, tpr_test, label=f'Testing ROC Curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


importances = model.feature_importances_
indices = np.argsort(importances)[::-1] 

f, ax = plt.subplots(figsize=(3, 8))
plt.title("Variable Importance - Random Forest")
sns.set_color_codes("pastel")
sns.barplot(y=[Xtrain.columns[i] for i in indices], x=importances[indices], 
            label="Total", color="b")
ax.set(ylabel="Variable",
       xlabel="Variable Importance (Gini)")
sns.despine(left=True, bottom=True)


# In[ ]:


churned = df[df['Exited'] == 1]  
average_age = churned["Age"].median()
print("Average age of those who left the bank:", average_age)


# In[ ]:


not_churned = df[df['Exited'] == 0]  
average_age = not_churned["Age"].median()
print("Average age of those who stayed at the bank:", average_age)


# In[ ]:


churned = df[df['Exited'] == 1]  
average_age = churned["NumOfProducts"].mean()
print("Average number of products of those who left the bank:", average_age)


# In[ ]:


not_churned = df[df['Exited'] == 0]  
average_age = not_churned["NumOfProducts"].mean()
print("Average number of products of those who stayed at the bank:", average_age)


# In[ ]:





# In[ ]:


not_churned = df[df['Exited'] == 0]  
average_age = not_churned["Balance"].mean()
print("Average bank balance of those who stayed at the bank:", average_age)


# In[ ]:


churned = df[df['Exited'] == 1]  
average_age = churned["Balance"].mean()
print("Average bank balance of those who left the bank:", average_age)


# In[ ]:


#PLotting the pie chart of the target variable
fig = plt.figure(figsize=(5,5))
data=df['Exited'].value_counts()
plt.pie(data,
labels=[f'{"Retained"}: {data.values[0]}', f'{"Churned"}: {data.values[1]}'], autopct='%1.1f%%')
plt.title('Bank Clients: Retained and Churned');


# In[ ]:


df['Satisfaction Score'].value_counts().sort_values()


# In[ ]:





# In[ ]:


#Plotting Churn by Geography 
country_counts = df2.groupby('Geography')['Exited'].value_counts().unstack()

plt.figure(figsize=(10, 6))
colors = ['blue', 'orange']  
labels = ['Retained', 'Churned'] 

for i, country in enumerate(country_counts.index):
    plt.subplot(1, 3, i+1) 
    plt.pie(country_counts.loc[country], labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'{country} - Churned vs. Retained')  

plt.tight_layout()
plt.show()


# In[ ]:


#Plotting the line chart of churn by age
age_groups = df.groupby(pd.cut(df['Age'], bins=range(0, 101, 5)))['Exited'].mean()


plt.figure(figsize=(10, 6))
age_groups.plot(kind='line', marker='o')
plt.xlabel('Age Group')
plt.ylabel('Likelihood of Churning')
plt.title('Age vs. Likelihood of Churning')
plt.xticks(range(len(age_groups)), [f'{i*5}-{(i+1)*5-1}' for i in range(len(age_groups))], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


df2['Exited'].value_counts().sort_values()


# In[ ]:


#Plotting Confusion Matrix
confusion = confusion_matrix(ytest, predictions)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0.5, 1.5], labels=['Retained', 'Churned'])
plt.yticks(ticks=[0.5, 1.5], labels=['Retained', 'Churned'])
plt.show()


# In[ ]:


#Plotting the line chart of churn by bank balance
min_balance = df['Balance'].min()
max_balance = df['Balance'].max()


bin_edges = pd.cut(df['Balance'], bins=8, precision=0)
balance_groups = df.groupby(bin_edges)['Exited'].mean()

plt.figure(figsize=(10, 6))
balance_groups.plot(kind='line', marker='o')
plt.xlabel('Bank Balance Group')
plt.ylabel('Likelihood of Churning')
plt.title('Bank Balance vs. Likelihood of Churning')
plt.xticks(range(len(balance_groups)), balance_groups.index.astype(str), rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


#Plotting bar graph of churn by number of products 
product_groups = df.groupby('NumOfProducts')['Exited'].mean()

plt.figure(figsize=(10, 6))
product_groups.plot(kind='bar')
plt.xlabel('Number of Products')
plt.ylabel('Likelihood of Churning')
plt.title('Number of Products vs. Likelihood of Churning')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()



# In[ ]:




