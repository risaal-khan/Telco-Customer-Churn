# Telco Customer Churn

## Context
* "Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

## Content
* Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
* The data set includes information about:
    * Customers who left within the last month – the column is called Churn
    * Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
    * Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
    * Demographic info about customers – gender, age range, and if they have partners and dependents

## Problem Understanding
* **Objective**: Predict customer churn (binary classification: "Yes" or "No") and identify actionable insights to reduce churn.
* **Target Column**: Churn (There are total 1869 churn value and 5174 non-churn value)

## Data Transformation
1. **Label encoding for binary categories**: gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod
2. **Scaling numerical variables using MinMaxScaler**: tenure, MonthlyCharges, TotalCharges (Use MinMaxScaler because columns are not normally distributed)

## Model Building

### Starting with a simple Logistic Regression as a baseline:
* After training and testing got 81.61 as accuracy score
* After adding parameters (C=[0.5, 1.0], penalty=['l1', 'l2'] , solver=['liblinear', 'newton-cg']), got same score
* After using gridsearchcv got same score and best parameters as ({'C': 1, 'penalty': 'l1', 'solver': 'saga'}).

### Random Forest
Using the model with gridsearchcv and got following
* Best parameters: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200}
* Best score: 0.8044010962945685
* accuracy score: 0.8055358410220014

### XGBoost
* Best parameters found:  {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8}
* Best cross-validation score: 0.8044010962945686
* Test accuracy of the best XGBoost model: 0.8126330731014905

### Gradient Boosting
* Best parameters found:  {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}
* Best cross-validation score: 0.8004969826056089
* Test accuracy of the best Gradient Boosting model: 0.808374733853797

### LightGBM
* Best parameters found:  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50, 'num_leaves': 31}
* Best cross-validation score: 0.8008519701810437
* Test accuracy of the best LightGBM model: 0.8097941802696949

#### Among all models, Logistic Regression & XGBoost perform best

## Analyzing feature importance using SHAP
* SHAP gives us a detailed understanding of how each feature contributes to predictions.
* SHAP is useful because it shows us the importance of each feature in making predictions. Providing Shapley values, helps us understand complex models and how input features affect predictions.

### SHAP summary plot Analysis
![image](https://github.com/user-attachments/assets/7833cb78-6e8d-41a0-a596-baacd7068f1a)

#### SHAP Value Interpretation
* **SHAP Value**: Shows whether a feature pushes the prediction toward churn (positive SHAP value) or not churn (negative SHAP value).
* **Color**: Indicates the feature value:
    * Red/Pink: High feature value.
    * Blue: Low feature value.

#### Top Features:
* **Contract**: Highly influential; shorter contracts (e.g., month-to-month) likely contribute to higher churn.
* **tenure**: Customers with lower tenure (shorter time with the company) tend to churn more.
* **MonthlyCharges**: Higher monthly charges are associated with churn.
* **OnlineSecurity and TechSupport**: Customers without these services (low values in blue) are more likely to churn.

#### Relationships Between Features and Target
* **tenure**: Shorter tenures (low feature values in blue) are associated with higher churn.
* **OnlineSecurity and TechSupport**: Absence of these services (blue dots) increases churn probability.
* **PaperlessBilling**: Customers with Paperless Billing (pink/red) tend to churn more.

### Insights for Churn Reduction
To reduce churn, focus on:
* **Contract**: Encourage customers to switch to long-term contracts to reduce churn.
* **OnlineSecurity and TechSupport**: Promote or bundle these services to customers who do not have them.
* **High Monthly Charges**: Provide discounts or lower-cost plans to high-spending customers to retain them.
* **New Customers (Low Tenure)**: Improve onboarding experiences to retain customers early in their lifecycle.
