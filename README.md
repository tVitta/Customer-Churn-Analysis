# Customer-Churn-Analysis
Analysing and predicting customer churn dataset using a random forest classification model.

## The Dataset
- 10,000+ datapoints
- Found on Kaggle

| Variable            | Meaning                                         |
|---------------------|-------------------------------------------------|
| RowNumber           | Record number (not affecting output)            |
| CustomerId          | Random customer identifier (not affecting output)|
| Surname             | Customer's surname (not impacting churn prediction) |
| CreditScore         | Credit score of the customer                   |
| Geography           | Customer's location (country)                  |
| Gender              | Customer's gender                               |
| Age                 | Customer's age                                  |
| Tenure              | Number of years as a client of the bank        |
| Balance             | Account balance                                 |
| NumOfProducts       | Number of bank products the customer uses      |
| HasCrCard           | Whether the customer has a credit card         |
| IsActiveMember      | Whether the customer is an active bank member  |
| EstimatedSalary     | Estimated annual salary of the customer        |
| Exited              | Whether the customer left the bank (target variable) |
| Satisfaction Score  | Customer's score for complaint resolution     |
| Card Type           | Type of credit card held by the customer       |
| Point Earned        | Points earned by the customer for using a credit card |


## Methodology
As previously mentioned, a random forest classification model was fitted to the data. 
  
This was chosen due to the fact that the target variable was a binary categorical variable.
  
The hyperparameters, were tuned using grid search cross validation, each iteration was assessed by its ROCAUC score. The grid search returned the best n_estimators at 150 and the best max_depth at 10, these hyperparameters carried with them an AUC score of 0.95. The max_depth was then manually reduced to 6 in order to prevent overfitting to the training data. The final hyperparameters would be set at n_estimators = 150 and max_depth = 6. 

## The Results

The ROC AUC score graphed:  
![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/5001168d-5c88-4478-8aed-b4f4e9675d00)
  
The variable importance was also measured:  
![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/b80a7f28-16e2-4606-aef9-002a627610a2)

## Key Takeaways
**Age**  
The average age of churned customers was 45, and the average age of retained customers was 36. This would seem to suggest that the oldest a customer is, the more likely they are to churn. However, if a line graph is applied showing the correlation between likelihood of churning and age, the distribution follows somewhat of a bell curve, with the likelihood of customer churning peaking between the ages of 50 and 54.  

![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/b82dc3b1-af1b-4ae8-a8ed-4f70b2f08729)  

**Number of Products**  
The second most important feature was the number of products each customer owned. This could be interpreted as the number of lines of credit/debit the customer had with the bank. When a bar graph is applied a correlation is shown. 

Customers least likely to churn own either one or two products, with customers owning two being by far the least likely to churn. While customers who own three or four products are vastly more likely to churn. 

![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/94ff407f-d2e5-4e72-bd19-c27155438fa9)  

**Bank Balance**  
Another Important Variable is the customer’s bank balance When a line chart is applied, we can see customers, from those with a negative bank balance all the way up to those with a balance of 180,000, have very similar rates of churning. Once a customer has a larger balance than 180,000, they are massively more likely to churn.

![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/25f835d8-95ba-4a3f-84bd-5b1799ed65cb)  

**Geography**  
The last key variable I’ll examine is geography, specifically, whether the customer is located within Germany or not. When examining the data, we can find that churn rates for the other two countries, Spain and France, are fairly similar at 16.7% and 16.2% respectively. The churn rate for Germany, however, is much higher at 32.4%. Although the data does not suggest why.

![image](https://github.com/tVitta/Customer-Churn-Analysis/assets/143434462/c2531602-fec6-4b7e-9a32-7851a9082e9f)






