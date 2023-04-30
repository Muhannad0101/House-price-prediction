import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from haversine import haversine
import xgboost
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
%matplotlib inline 


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test.head()

train.dtypes
train.describe(include=['int64'])
train.describe(include=['float64']).T
train.describe(include=['object'])
avg_stroke = train["SalePrice"].astype("float").mean(axis = 0)
print("Averge of Sale Price:", avg_stroke)

# Probability Plot 
plt.subplots(figsize=(6,4))
sns.distplot(train['SalePrice'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()

train['SalePriceLog'] = np.log1p(train['SalePrice'])
plt.subplots(figsize=(6,4))
sns.distplot(train['SalePriceLog'], fit=stats.norm)
(mu, sigma) = stats.norm.fit(train['SalePriceLog'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
fig = plt.figure()
stats.probplot(train['SalePriceLog'], plot=plt)
plt.show()

del train['SalePriceLog']

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.head()

# Exclude ID column 
numeric_features = numeric_features.drop('Id', axis=1)

# Loop through each feature 
for col in numeric_features:
    plt.figure(figsize=(5,4))
    plt.scatter(train[col], train['SalePrice'])
    
    sns.regplot(x=col, y='SalePrice', data=train)
    
    plt.title(col + ' vs Sale Price')
    plt.xlabel(col)
    plt.ylabel('Sale Price')
    
    plt.show() 

# Explanatory data analysis
figure= plt.subplots(figsize=(9,6))
plt.bar(train['RoofStyle'], train['SalePrice'])

figure= plt.subplots(figsize=(9,6))
plt.bar(train['RoofMatl'], train['SalePrice'])

plt.figure(figsize=(20,5))
sns.lineplot(data=train, x="YearBuilt", y="SalePrice")

plt.bar(train['YrSold'], train['SalePrice'])

plt.bar(train['MoSold'], train['SalePrice'])

plt.bar(train['MSZoning'], train['SalePrice'])

import matplotlib.pyplot as plt

plt.figure(figsize=(8,10))
train['SaleCondition'].value_counts(normalize=True).plot.pie(autopct='%1.2f%%')

plt.figure(figsize=(10,10))
plt.bar(train['Neighborhood'], train['SalePrice'])
plt.xticks(rotation= 45)

# Missing values and Remove outliers
miss_col_train = train.isnull().sum()[train.isnull().sum() > 0].sort_values(ascending=True)
percent_miss_train = round((miss_col_train / len(train) * 100) , 2)
missing_train = pd.DataFrame([miss_col_train, percent_miss_train]).T.rename(columns={0:'Feature', 1:'missing'})

train.fillna(value=0, inplace=True)

# Replace infinite values with a large finite value or np.nan, and then handle the np.nan values as before
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train.fillna(value=0, inplace=True)

train.isnull().sum()

# Remove outliers
cat_features = np.array([i for i in train.columns.tolist() if train[i].dtype == 'object'])
num_features = np.array([i for i in train.columns.tolist() if train[i].dtype != 'object'])
print("Number features column =" , len(num_features))
print("Categorial features column =" , len(cat_features))

# plot a boxplot for the label by each numerical feature  
for col in train.describe().columns:
  fig = plt.figure(figsize=(9, 6))
  ax = fig.gca()
  train.boxplot(column = col, ax = ax)
  ax.set_ylabel(col)
plt.title("Box plot for trip_duration")
plt.show()

# Function to detect outliers
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


## function to remove outliers
def replace_with_thresholds(dataframe,columns):
    for col in columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
        dataframe.loc[(dataframe[col] > up_limit), col] = up_limit

replace_with_thresholds(train, num_features)

# Corralation
train_corr = train.select_dtypes(include=[np.number])

corr = train_corr.corr()
plt.subplots(figsize=(17,9))
sns.heatmap(corr, annot=True)

print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice

top_feature = corr.index[abs(corr['SalePrice']>0.5)]

col = ['SalePrice','OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
       'YearRemodAdd']

numcols = train[train.columns.intersection(top_feature)]
numcols.head()

numcols = train[train.columns.intersection(top_feature)]
plt.figure(figsize=(8,8))
sns.heatmap(numcols.corr(), annot=True)

sns.set(style='ticks')
sns.pairplot(train[col], size=2, kind='reg')

train[col].replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

train[col].info()

# Split and Normalize Data

from matplotlib import legend
# Function for evaluation metric for regression
def EvaluationMetric(Xt,yt,yp,disp="on"):
    ''' Take the different set of parameter and prints evaluation metrics '''
    MSE=round(mean_squared_error(y_true=yt,y_pred=yp),4)
    RMSE=(np.sqrt(MSE))
    R2=(r2_score(y_true=yt,y_pred=yp))
    Adjusted_R2=(1-(1-r2_score(yt, yp))*((Xt.shape[0]-1)/(Xt.shape[0]-Xt.shape[1]-1)))
    if disp=="on":
        print("MSE :",MSE,"RMSE :", RMSE)
        print("R2 :",R2,"Adjusted R2 :",Adjusted_R2)
    
    #Plotting Actual and Predicted Values
    plt.figure(figsize=(18,6))
    plt.plot((yp)[:100]) 
    plt.plot((np.array(yt)[:100]))
    plt.legend(["Predicted","Actual"])
    plt.title('Actual and Predicted Time Duration')
    
    return (MSE,RMSE,R2,Adjusted_R2) 

X = numcols.drop('SalePrice', axis=1)
y = numcols['SalePrice']

X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Regression models

## Linear Regression
# Instance the linear regression object
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)

y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

EvaluationMetric(X_train,y_train,y_pred_train)

EvaluationMetric(X_test,y_test,y_pred_test)

residuals=y_pred_test-y_test

plt.figure(figsize=(10, 6), dpi=120, facecolor='w', edgecolor='b')
f = range(0,len(y_test))
k = [0 for i in range(0,len(y_test))]
plt.scatter( f, residuals, label = 'residuals')
plt.plot( f, k , color = 'red', label = 'regression line' )
plt.xlabel('fitted points ')
plt.ylabel('residuals')
plt.title('Residual plot')
plt.legend()

## LightGBM
n_estimator=[5,10,20] # No. of tree
max_depth=[5,7,9] # max depth of tree
min_samples_split=[40,50]

params={"n_estimator":n_estimator,
        "max_depth":max_depth,
        "min_samples_split":min_samples_split}

lgb = LGBMRegressor()
gs_lgb = GridSearchCV(lgb,params,cv=3,verbose=2,scoring='r2')
gs_lgb.fit(X_train,y_train)

print(gs_lgb.best_score_)
print(gs_lgb.best_params_)

gs_lgb.best_estimator_
gs_lgb_opt_model = gs_lgb.best_estimator_

y_preds_lgb_test = gs_lgb_opt_model.predict(X_test)
y_pred_lgb_train = gs_lgb_opt_model.predict(X_train)

#Evaluation metrics for Train set
EvaluationMetric(X_train,y_train,y_pred_lgb_train)

EvaluationMetric(X_test, y_test, y_preds_lgb_test)

# best_estimator
lgb_best = LGBMRegressor(max_depth=9, 
                         min_samples_split=40, 
                         n_estimator=5)
lgb_best.fit(X_train,y_train)

y_pred_best = lgb_best.predict(X_test)
r2_score(y_test,y_pred_best)

importances = gs_lgb_opt_model.feature_importances_

importance_dict = {'Feature' : list(X.columns),
                   'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)

importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)

plt.figure(figsize=(15,5))
plt.title('Top 10 Features')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])

sns.distplot(y_test - y_preds_lgb_test ).set_title("error distribution between actual and predicted values")
plt.show()

## Random Forest 
rf_reg = RandomForestRegressor(random_state=7)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
r2_score(y_pred,y_test)

n_estimators = [90,100]
max_depth = (1,7,1)
min_samples_leaf = (1,7,1)
min_samples_split = (1,7,1)
max_features = ['auto','log2']

param = {
    "n_estimators" : n_estimators,
    'max_depth' : max_depth,
    'min_samples_leaf' : min_samples_leaf,
    'min_samples_split': min_samples_split,
    'max_features' : max_features
}
rf_grid = GridSearchCV(rf_reg,param)
rf_grid.fit(X_train,y_train)

print(rf_grid.best_score_)
print(rf_grid.best_params_)

rf_grid.best_estimator_
gs_rf_opt_model = rf_grid.best_estimator_

y_preds_rf_test = gs_lgb_opt_model.predict(X_test)
y_pred_rf_train = gs_lgb_opt_model.predict(X_train)

#Evaluation metrics for Train set
EvaluationMetric(X_train,y_train,y_pred_rf_train)

EvaluationMetric(X_test, y_test, y_preds_rf_test)

rf_best = RandomForestRegressor(max_depth=7, max_features='log2', min_samples_leaf=7,
                      min_samples_split=7, random_state=7)
rf_best.fit(X_train,y_train)
y_pred_best = rf_best.predict(X_test)
r2_score(y_test,y_pred_best)

importances = rf_best.feature_importances_

importance_dict = {'Feature' : list(X.columns),
                   'Feature Importance' : importances}

importance_df = pd.DataFrame(importance_dict)

importance_df.sort_values(by=['Feature Importance'],ascending=False,inplace=True)

plt.figure(figsize=(15,5))
plt.title('Top 10 Features')
sns.barplot(x='Feature',y="Feature Importance",data=importance_df[:10])

sns.distplot(y_test - y_preds_rf_test).set_title("error distribution between actual and predicted values")
plt.show()

# test data and Submission
test.fillna(value=0, inplace=True)

# Replace infinite values with a large finite value or np.nan, and then handle the np.nan values as before
test.replace([np.inf, -np.inf], np.nan, inplace=True)
test.fillna(value=0, inplace=True)

cat_features_test = np.array([i for i in test.columns.tolist() if test[i].dtype == 'object'])
num_features_test = np.array([i for i in test.columns.tolist() if test[i].dtype != 'object'])

replace_with_thresholds(test, num_features_test)

X_cols = top_feature.copy()
X_test = test[X_cols[1:]]
X_test.info()

rf_best = RandomForestRegressor(max_depth=7, max_features='log2', min_samples_leaf=7,
                      min_samples_split=7, random_state=7)
rf_best.fit(X,y)
y_pred = rf_best.predict(X_test)
submission = pd.DataFrame({"Id":test.index, "SalePrice":y_pred})
submission.to_csv('submission.csv', index=False)