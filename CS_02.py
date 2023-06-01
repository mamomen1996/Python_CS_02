#!/usr/bin/env python
# coding: utf-8

# # Case-Study Title: Sports Analytics (Regression Methods in Python)
# ###### Data Analysis methodology: CRISP-DM
# ###### Dataset: Hitters dataset (Major League Baseball Data from the 1986 and 1987 seasons in US)
# ###### Case Goal: Annual Salary prediction of each Player in 1987 base on his performance in 1986

# # Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from tensorflow import keras


# # Read Data from File

# In[2]:


data = pd.read_csv('CS_02.csv')


# In[3]:


data.shape  # 322 records, 21 variables


# # Business Understanding
#  * know business process and issues
#  * know the context of the problem
#  * know the order of numbers in the business

# # Data Understanding
# ## Data Inspection (Data Understanding from Free Perspective)
# ### Dataset variables definition

# In[4]:


data.columns  # KPI (Key Performance Indicator) variables


# 1. KPI variables in 1986:
# * **AtBat**:      Number of times at bat in 1986
# * **Hits**:       Number of hits in 1986
# * **HmRun**:      Number of home runs in 1986
# * **Runs**:       Number of runs in 1986
# * **RBI**:        Number of runs batted in in 1986
# * **Walks**:      Number of walks in 1986
# * **PutOuts**:    Number of put outs in 1986
# * **Assists**:    Number of assists in 1986
# * **Errors**:     Number of errors in 1986
# 
# 2. KPI variables in whole career life:
# * **Years**:      Number of years in the major leagues
# * **CAtBat**:     Number of times at bat during his career
# * **CHits**:      Number of hits during his career
# * **CHmRun**:     Number of home runs during his career
# * **CRuns**:      Number of runs during his career
# * **CRBI**:       Number of runs batted in during his career
# * **CWalks**:     Number of walks during his career
# 
# 3. Categorical variables:
# * **League**:     A factor with levels A and N indicating player's league at the end of 1986 (american league|national league)
# * **Division**:   A factor with levels E and W indicating player's division at the end of 1986 (west|east)
# * **NewLeague**:  A factor with levels A and N indicating player's league at the beginning of 1987
# * **Name**:       name of players
# 
# 4. Outcome variable:
# * **Salary**:     1987 annual salary on opening day in thousands of dollars

# ## Data Exploring (Data Understanding from Statistical Perspective)
# ### Overview of Dataframe

# In[5]:


type(data)


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.info()


# In[9]:


# Do we have any NA in our Variables?
data.isna().sum()  # count NAs in columns

# We have 56 NAs in 'Salary'


# In[10]:


# Check for abnormality in data
data.describe(include='all')


# ### Dealing with MVs

# In[11]:


# Remove records with MVs (remove records with NA in their 'Salary' column)
data2 = data.dropna(subset = ['Salary'], inplace = False, axis = 0)


# In[12]:


# Remove Players' Name
data2 = data2.iloc[:, 1:]


# In[13]:


data2.head()


# In[14]:


data2.shape


# In[15]:


data2.describe()


# ### Univariate Profiling (check each variable individually)
# #### Categorical variables
# Check to sure that have good car distribution in each category
# 
# **Rule of Thumb**: we must have atleast 30 observation in each category

# In[16]:


data2.League.value_counts()


# In[17]:


data2.Division.value_counts()


# In[18]:


data2.NewLeague.value_counts()


# #### Continuous variables
# distribution: plot Histogram

# In[19]:


var_ind = list(range(13)) + list(range(15, 19))
plot = plt.figure(figsize = (12, 6))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 18):
    a = plot.add_subplot(3, 6, i)
    a.hist(data2.iloc[:, var_ind[i - 1]], alpha = 0.7)
    a.title.set_text(data2.columns[var_ind[i - 1]])


# In[20]:


# Box plot of 'Salary'
plt.boxplot(data2['Salary'], showmeans = True)  # Outlier detection by Tukey method
plt.title('Boxplot of Salary')

# the outliers in this case are Nature of this problem


# ### Bivariate Profiling (measure 2-2 relationships between variables)
# #### Two Continuous variables (Correlation Analysis)

# In[21]:


# correlation table between Salary and continuous variables
corr_table = round(data2.iloc[:, var_ind].corr(method = 'pearson'), 2)
corr_table  # two-by-two Pearson Correlation between variables


# > **Salary** has good correlations with features

# In[22]:


# Correlation plot
plt.figure(figsize = (12, 6))
sns.heatmap(corr_table, annot = True)


# > **Multicollinearity** problem: we have high correlations between features with each other

# In[23]:


# Scatter Plot (between Salary and other continuous variables 2 by 2)
var_ind = list(range(13)) + list(range(15, 18))
plot = plt.figure(figsize = (12, 12))
plot.subplots_adjust(hspace = 0.8, wspace = 0.5)
for i in range(1, 17):
    a = plot.add_subplot(4, 4, i)
    a.scatter(x = data2.iloc[:, var_ind[i - 1]], y = data2.iloc[:, 18], alpha = 0.5)
    a.title.set_text('Salary vs. ' + data2.columns[var_ind[i - 1]])


# > There is a good linear relationship between **Career** variables and **Salary**

# # Data PreProcessing
# ## Divide Dataset into Train and Test randomly
# * Learn model in Train dataset
# * Evaluate model performance in Test dataset

# In[24]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data2, test_size = 0.2, random_state = 1234)

# according to the dataset size: 80% - 20% 


# In[25]:


# train data distribution must be similar to test data distribution
train.shape


# In[26]:


train.describe()


# In[27]:


test.shape


# In[28]:


test.describe()


# # Modeling
# ## Model 1: Simple Linear Regression

# In[29]:


# Create binary dummy variables for Categorical variables (One-Hot Encoding)
dummies = pd.get_dummies(train[['League', 'Division', 'NewLeague']])
dummies.head()


# In[30]:


# Define the features set X
X_ = train.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_train = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_train = sm.add_constant(X_train)  # adding a constant column (a column of 1)

# Define response variable
y_train = train['Salary']


# In[31]:


X_train.head()  # features matrix


# In[32]:


y_train.head()  # response matrix


# In[33]:


# Linear Regression model
lm = sm.OLS(y_train, X_train).fit()
lm.summary()


# **results**:
# 1. Y is so skewed
# 2. Multicollinearity problem
# 3. Not-Normality of Errors

# Check Assumptions of Regression:
# 
# 1. Normality of residuals (Errors)

# In[34]:


# Plot Histogram of residuals
sns.histplot(lm.resid, stat = 'probability',
            kde = True, alpha = 0.7, color = 'green',
            bins = 20)

# skewed to right


# In[35]:


# QQ-plot
qqplot_lm = sm.qqplot(lm.resid, line = 's')
plt.show()


# In[36]:


# Jarque-Bera Test (Normal Skewness = 0)
  # H0: the data is normally distributed
  # if p-value < 0.05, then reject normality assumption

# Omnibus K-squared normality test (Normal Kurtosis = 3)
  # H0: the data is normally distributed
  # if p-value < 0.05, then reject normality assumption

print(lm.summary())


# > **result**: Residuals are not Normally Distributed -> reject first Assumption of Regression
# 
# 2. Residuals independency

# In[37]:


# Diagnostic plot for checking Heroscedasticity problem

sns.regplot(x = lm.fittedvalues, y = lm.resid, lowess = True,
               scatter_kws = {'color': 'black'}, line_kws = {'color': 'red'})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()


# > **result**: We see Heteroscedasticity problem in model (variance of residuals is not constant)

# In[38]:


# Check Cook's distance
sum(lm.get_influence().summary_frame().cooks_d > 1)


# > **result**: there is no Cook's Distance > 1
# 
# Check having Multicollinearity problem via VIF

# In[39]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):  # X: features matrix
    vif = pd.DataFrame()
    vif['variables'] = X.columns  # column names
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

calc_vif(X_train.iloc[:, 1:]) # calculate VIF for each variable (if VIF > 10 then Multicollinearity problem is serious)


# **Conclusion**:
# * Severe violation of Regression Assumptions
# * t-test results were not reliable for features selection
# * Weak prediction power (bad model)

# In[40]:


# Linear Regression model based on t-test results feature selection (just with significant variables)
lm = sm.OLS(y_train, X_train[['const', 'AtBat', 'Hits', 'Walks', 'PutOuts', 'Division_W']]).fit()
lm.summary()


# * R-squared decreased
# * the Errors distribution is not Normal
# * still have Multicollinearity problem
# * weak model for Prediction

# ### Prediction on Test dataset
# use Model 1 for prediction on Test dataset

# In[41]:


test.head()


# In[42]:


# Create dummy variables for Categorical variables
dummies = pd.get_dummies(test[['League',
                               'Division',
                               'NewLeague']])

# Define the feature set X
X_ = test.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_test = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis = 1)
X_test = sm.add_constant(X_test)  # adding a constant column

# Define response variable
y_test = test['Salary']


# In[43]:


X_test.head()


# In[44]:


y_test.head()


# In[45]:


# use lm model for prediction
pred_lm = lm.predict(X_test[['const', 'AtBat', 'Hits', 'Walks', 'PutOuts', 'Division_W']])


# Absolute Error

# In[46]:


abs_err_lm = abs(y_test - pred_lm)


# Absolute Error mean, median, sd, IQR, max, min

# In[47]:


from scipy.stats import iqr

model_comp = pd.DataFrame({'Mean of AbsErrors': abs_err_lm.mean(),
                           'Median of AbsErrors': abs_err_lm.median(),
                           'SD of AbsErrors': abs_err_lm.std(),
                           'IQR of AbsErrors': iqr(abs_err_lm),
                           'Min of AbsErrors': abs_err_lm.min(),
                           'Max of AbsErrors': abs_err_lm.max()}, 
                          index = ['LM_t-test'])

model_comp  # Comparison different models based-on Absolute Error indexes in Test dataset


# Actual vs. Prediction

# In[48]:


plt.scatter(x = y_test, y = pred_lm)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# Add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ### Box-Cox Transformation

# In[49]:


from scipy.stats import boxcox

box_results = boxcox(y_train, alpha = 0.05)
box_results  # transformed y_train with optimum lambda and its 95% confidence interval


# > optimum lambda (point estimate) is **0.13847** and confidence interval 95% is **(-0.01298, 0.28931)**
# 
# > **0** is in this confidence interval, so **0.13847** has not statistically significant difference from **0**, so we can consider **lambda = 0** and then **y' = log(y)**

# In[50]:


# log transformation (changing variable to become Y variable Normal)
logy_train = np.log(y_train)
logy_train


# In[51]:


# Plot Histogram of 'Salary'
sns.histplot(y_train, stat = 'probability', kde = True, alpha = 0.7, color = 'green', bins = 20)

# skewed


# In[52]:


# QQ-plot
qqplot_lm_bc = sm.qqplot(y_train, line = 's')
plt.show()


# In[53]:


# Plot Histogram of 'log(Salary)'
sns.histplot(logy_train, stat = 'probability', kde = True, alpha = 0.7, color = 'green', bins = 20)


# In[54]:


# QQ-plot
qqplot_lm_bc = sm.qqplot(logy_train, line = 's')
plt.show()


# > from here, we will make our models on **log(Y)** to have Errors with constant variance and Normal distribution
# 
# > there is strong **Multicollinearity** between variables

# ## Model 2: Linear Regression Using the Best Subset Selection

# In[55]:


X_train.head()  # 19 predictor variables


# In[56]:


# Define function to fit Linear Regression
def fit_lm(feature_set):
    reg_model = sm.OLS(logy_train, X_train[['const'] + list(feature_set)]).fit()
    return {'model': reg_model, 'RSquared': reg_model.rsquared}


# In[57]:


# Get all possible 3-variable combinations of 19 variables
import itertools
list(itertools.combinations(X_train.columns[1:], 3))  # 969 different possible 3-variable regressions


# Best Subset Selection

# In[58]:


def bestsubset_func(k):
    res = []
    for features in itertools.combinations(X_train.columns[1:], k):
        res.append(fit_lm(features))  # fit regression and save results for each state
        
    models = pd.DataFrame(res)
    best_model = models.iloc[models['RSquared'].argmax()]  # extract the best model from dataframe
    return best_model  # return the best model for k-variable regression


# In[59]:


import time

models_bestsub = pd.DataFrame(columns = ['RSquared', 'model'])

start_time = time.time()
for i in range(1, len(X_train.columns[1:]) + 1):
    models_bestsub.loc[i] = bestsubset_func(i)  # save the best k-variable regression model
end_time = time.time()

print('The Processing time is:', end_time - start_time, 'seconds')


# In[60]:


models_bestsub  # 19 different Regression models


# In[61]:


print(models_bestsub.loc[4, 'model'].summary())  # 4-variables regression model


# Comparison models based-on Adj. R-squared

# In[62]:


# Extract Adj. R-squared of all 19 models
models_bestsub_adjrs = models_bestsub.apply(lambda row: row[1].rsquared_adj, axis = 1)
models_bestsub_adjrs


# In[63]:


# Adj. R-squared plot
plt.plot(models_bestsub_adjrs)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('Adj. R-squared')
plt.axvline(models_bestsub_adjrs.argmax() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > best model based-on **Adj. R-squared**: 11-variables Linear Regression

# In[64]:


# AIC plot
models_bestsub_aic = models_bestsub.apply(lambda row: row[1].aic, axis = 1)  # Extract AIC of all 19 models
plt.plot(models_bestsub_aic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('AIC')
plt.axvline(models_bestsub_aic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > best model based-on **AIC**: 8-variables Linear Regression

# In[65]:


# BIC plot
models_bestsub_bic = models_bestsub.apply(lambda row: row[1].bic, axis = 1)  # Extract BIC of all 19 models
plt.plot(models_bestsub_bic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('BIC')
plt.axvline(models_bestsub_bic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > best model based-on **BIC**: 3-variables Linear Regression

# In[66]:


# Linear Regression model with 11 variables
models_bestsub.loc[11, 'model'].params  # extract model parameters


# In[67]:


models_bestsub.loc[11, 'model'].model.exog_names  # name of features


# ### Prediction on Test dataset
# use Model 2 for prediction on Test dataset

# In[68]:


# Extract appropriate columns from Test dataset which are same to variables used in 11-variables Regression model
X_test[models_bestsub.loc[11, 'model'].model.exog_names].head()


# In[69]:


pred_bestsub = models_bestsub.loc[11, 'model'].predict(X_test[models_bestsub.loc[11, 'model'].model.exog_names])
pred_bestsub.head()  # prediction of log(Salary)


# In[70]:


pred_bestsub = np.exp(pred_bestsub)
pred_bestsub.head()  # prediction of Salary


# In[71]:


y_test.head()  # real value of Salary


# Absolute Error

# In[72]:


abs_err_bestsub = abs(y_test - pred_bestsub)


# Absolute Error mean, median, sd, IQR, max, min

# In[73]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_bestsub.mean(),
                                             'Median of AbsErrors': abs_err_bestsub.median(),
                                             'SD of AbsErrors': abs_err_bestsub.std(),
                                             'IQR of AbsErrors': iqr(abs_err_bestsub),
                                             'Min of AbsErrors': abs_err_bestsub.min(),
                                             'Max of AbsErrors': abs_err_bestsub.max()},
                                            index = ['BestSubset']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[74]:


plt.scatter(x = y_test, y = pred_bestsub)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 3: Forward and Backward Stepwise Selection Regression

# In[75]:


def forward_func(features):
    res = []
    
    remaining_features = [_ for _ in X_train.columns[1:] if _ not in features]
    
    for f in remaining_features:
        res.append(fit_lm(features + [f]))
    
    models = pd.DataFrame(res)
    
    best_model = models.iloc[models['RSquared'].argmax()]  # choose the model with the Highest R-Squared
    
    return best_model


# In[76]:


# Forward Stepwise Selection
import time

models_fw = pd.DataFrame(columns = ['RSquared', 'model'])
start_time = time.time()
features = []
for i in range(1, len(X_train.columns[1:]) + 1):
    models_fw.loc[i] = forward_func(features)
    features = models_fw.loc[i, 'model'].model.exog_names[1:]
end_time = time.time()
print('The Processing time is:', end_time - start_time, 'seconds')


# In[77]:


models_fw  # 19 different Regression models


# In[78]:


print(models_fw.loc[4, 'model'].summary())  # 4-variable model


# Adj. R-squared

# In[79]:


models_fw_adjrs = models_fw.apply(lambda row: row[1].rsquared_adj, axis = 1)
models_fw_adjrs


# In[80]:


# Adj. R-squared plot
plt.plot(models_fw_adjrs)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('Adj. R-squared')
plt.axvline(models_fw_adjrs.argmax() + 1, color = 'red', linewidth = 2, linestyle = '--')


# In[81]:


# AIC plot
models_fw_aic = models_fw.apply(lambda row: row[1].aic, axis = 1)
plt.plot(models_fw_aic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('AIC')
plt.axvline(models_fw_aic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# In[82]:


# BIC plot
models_fw_bic = models_fw.apply(lambda row: row[1].bic, axis = 1)
plt.plot(models_fw_bic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('BIC')
plt.axvline(models_fw_bic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# Linear Regression model with 12 variables

# In[83]:


models_fw.loc[12, 'model'].params


# In[84]:


def backward_func(features):
    res = []
    
    for features in itertools.combinations(features, len(features) - 1):
        res.append(fit_lm(features))
        
    models = pd.DataFrame(res)
    
    best_model = models.iloc[models['RSquared'].argmax()]
    
    return best_model


# In[85]:


# Backward Stepwise Selection
models_bw = pd.DataFrame(columns = ['RSquared', 'model'])
start_time = time.time()
features = X_train.columns
while(len(features) > 1):
    models_bw.loc[len(features) - 1] = backward_func(features)
    features = models_bw.loc[len(features) - 1]['model'].model.exog_names[1:]
end_time = time.time()
print('The Processing time is:', end_time - start_time, 'seconds')


# In[86]:


models_bw  # 19 different Regression models


# In[87]:


print(models_bw.loc[4, 'model'].summary())  # 4-variable model


# Adj. R-squared

# In[88]:


models_bw_adjrs = models_bw.apply(lambda row: row[1].rsquared_adj, axis = 1)
models_bw_adjrs = models_bw_adjrs.sort_index()
models_bw_adjrs


# In[89]:


# Adj. R-squared plot
plt.plot(models_bw_adjrs)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('Adj. R-squared')
plt.axvline(models_bw_adjrs.argmax() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > the best model based-on **Adj. R-squared** is 11-variable model

# In[90]:


# AIC plot
models_bw_aic = models_bw.apply(lambda row: row[1].aic, axis = 1)
models_bw_aic = models_bw_aic.sort_index()
plt.plot(models_bw_aic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('AIC')
plt.axvline(models_bw_aic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > the best model based-on **AIC** is 8-variable model

# In[91]:


# BIC plot
models_bw_bic = models_bw.apply(lambda row: row[1].bic, axis = 1)
models_bw_bic = models_bw_bic.sort_index()
plt.plot(models_bw_bic)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('BIC')
plt.axvline(models_bw_bic.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# > the best model based-on **BIC** is 2-variable model

# ### Prediction on Test dataset
# use Model 3 for prediction on Test dataset

# In[92]:


# Forward Stepwise Selection model with 12 variables (based-on Adj. R-squared)
models_fw.loc[12, 'model'].params


# In[93]:


pred_fw = models_fw.loc[12, 'model'].predict(X_test[models_fw.loc[12, 'model'].model.exog_names])  # prediction of log(Salary)
pred_fw = np.exp(pred_fw)  # prediction of Salary
pred_fw.head()


# Absolute Error

# In[94]:


abs_err_fw = abs(y_test - pred_fw)


# Absolute Error mean, median, sd, IQR, max, min

# In[95]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_fw.mean(),
                                             'Median of AbsErrors': abs_err_fw.median(),
                                             'SD of AbsErrors': abs_err_fw.std(),
                                             'IQR of AbsErrors': iqr(abs_err_fw),
                                             'Min of AbsErrors': abs_err_fw.min(),
                                             'Max of AbsErrors': abs_err_fw.max()},
                                           index = ['Forward Stepwise']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[96]:


plt.scatter(x = y_test, y = pred_fw)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 4: Stepwise Regression Using k-fold Cross-Validation approach
# use k-fold Cross-Validation instead of AIC, BIC and Adj. R-squared (statistical indexes) to choose the best model

# In[97]:


k = 10  # create 10 folds
np.random.seed(123)
folds = np.random.randint(low = 1, high = k + 1, size = X_train.shape[0])
folds


# In[98]:


cv_errors = pd.DataFrame(index = range(1, k + 1), columns = range(1, 20))
cv_errors


# In[99]:


# Forward Stepwise Selection results
models_fw


# In[100]:


# Forward Stepwise Selection using k-fold Cross-Validation
for i in range(1, models_fw.shape[0] + 1):
    for j in range(1, k + 1):
        reg_model = sm.OLS(logy_train[folds != j], X_train.loc[folds != j, models_fw.loc[i, 'model'].model.exog_names]).fit()
        pred = reg_model.predict(X_train.loc[folds == j, models_fw.loc[i, 'model'].model.exog_names])
        cv_errors.iloc[j - 1, i - 1] = ((logy_train[folds == j] - pred) ** 2).mean()
        
cv_errors


# In[101]:


mean_cv_errors = cv_errors.mean(axis = 0)
mean_cv_errors


# In[102]:


# Mean of CV Errors plot
plt.plot(mean_cv_errors)
plt.xlabel('# Predictors')
plt.xticks(range(1, 20))
plt.ylabel('Mean of CV Errors')
plt.axvline(mean_cv_errors.argmin() + 1, color = 'red', linewidth = 2, linestyle = '--')


# ### Prediction on Test dataset
# use Model 4 for prediction on Test dataset

# In[103]:


# Forward Stepwise Selection model with 5 variables
models_fw.loc[5, 'model'].params


# In[104]:


pred_fw_cv = models_fw.loc[5, 'model'].predict(X_test[models_fw.loc[5, 'model'].model.exog_names])
pred_fw_cv = np.exp(pred_fw_cv)
pred_fw_cv.head()


# Absolute Error

# In[105]:


abs_err_fw_cv = abs(y_test - pred_fw_cv)


# Absolute Error mean, median, sd, IQR, max, min

# In[106]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_fw_cv.mean(),
                                             'Median of AbsErrors': abs_err_fw_cv.median(),
                                             'SD of AbsErrors': abs_err_fw_cv.std(),
                                             'IQR of AbsErrors': iqr(abs_err_fw_cv),
                                             'Min of AbsErrors': abs_err_fw_cv.min(),
                                             'Max of AbsErrors': abs_err_fw_cv.max()},
                                           index = ['Forward Stepwise CV']),
                              ignore_index = False)
model_comp


# ## Model 5: Ridge Regression

# In[107]:


lambda_grid = 10 ** np.linspace(5, -2, 100)
lambda_grid


# In[108]:


from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
ridgereg = Ridge()
scaler = StandardScaler()
scaler.fit(X_train)

models = pd.DataFrame(index = lambda_grid, columns = X_train.columns)
coefs = []
for i in lambda_grid:
    ridgereg.set_params(alpha = i)
    ridgereg.fit(scaler.transform(X_train), logy_train)
    models.loc[i] = ridgereg.coef_

models.shape


# In[109]:


models.head()


# In[110]:


models.tail()


# Plot Results

# In[111]:


plot_ridge = plt.gca() 
plot_ridge.plot(models)
plot_ridge.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('weights')


# k-fold Cross-Validation to choose the Best Model

# In[112]:


ridgecv = RidgeCV(alphas = lambda_grid, cv = 10)
ridgecv.fit(scaler.transform(X_train), logy_train)
ridgecv.alpha_


# ### Prediction on Test dataset
# use Model 5 for prediction on Test dataset

# In[113]:


# the Best Model Coefs
ridgecv.coef_


# In[114]:


ridgereg = Ridge(alpha = ridgecv.alpha_)
ridgereg.fit(scaler.transform(X_train), logy_train)
pred_ridge = ridgereg.predict(scaler.transform(X_test))
pred_ridge = np.exp(pred_ridge)
pred_ridge


# Absolute Error

# In[115]:


abs_err_ridge = abs(y_test - pred_ridge)


# Absolute Error mean, median, sd, IQR, max, min

# In[116]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_ridge.mean(),
                                             'Median of AbsErrors': abs_err_ridge.median(),
                                             'SD of AbsErrors': abs_err_ridge.std(),
                                             'IQR of AbsErrors': iqr(abs_err_ridge),
                                             'Min of AbsErrors': abs_err_ridge.min(),
                                             'Max of AbsErrors': abs_err_ridge.max()},
                                           index = ['Ridge Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[117]:


plt.scatter(x = y_test, y = pred_ridge)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 6: LASSO Regression

# In[118]:


lambda_grid = 10 ** np.linspace(1, -3, 100)


# In[119]:


from sklearn.linear_model import Lasso, LassoCV
lassoreg = Lasso()

models = pd.DataFrame(index = lambda_grid, columns = X_train.columns)
coefs = []
for i in lambda_grid:
    lassoreg.set_params(alpha = i)
    lassoreg.fit(scaler.transform(X_train), logy_train)
    models.loc[i] = lassoreg.coef_

models.shape


# In[120]:


models.head()


# In[121]:


models.tail()


# Plot Results

# In[122]:


plot_lasso = plt.gca()
plot_lasso.plot(models)
plot_lasso.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('weights')


# k-fold Cross-Validation to choose the Best Model

# In[123]:


lassocv = LassoCV(alphas = lambda_grid, cv = 10)
lassocv.fit(scaler.transform(X_train), logy_train)
lassocv.alpha_


# ### Prediction on Test dataset
# use Model 6 for prediction on Test dataset

# In[124]:


# the Best Model Coefs
lassocv.coef_


# In[125]:


lassoreg = Lasso(alpha = lassocv.alpha_)
lassoreg.fit(scaler.transform(X_train), logy_train)
pred_lasso = lassoreg.predict(scaler.transform(X_test))
pred_lasso = np.exp(pred_lasso)
pred_lasso


# Absolute Error

# In[126]:


abs_err_lasso = abs(y_test - pred_lasso)


# Absolute Error mean, median, sd, IQR, max, min

# In[127]:


from scipy.stats import iqr
model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_lasso.mean(),
                                             'Median of AbsErrors': abs_err_lasso.median(),
                                             'SD of AbsErrors': abs_err_lasso.std(),
                                             'IQR of AbsErrors': iqr(abs_err_lasso),
                                             'Min of AbsErrors': abs_err_lasso.min(),
                                             'Max of AbsErrors': abs_err_lasso.max()},
                                           index = ['LASSO Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[128]:


plt.scatter(x = y_test, y = pred_lasso)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 7: Decision Tree

# In[129]:


from sklearn.tree import DecisionTreeRegressor, plot_tree


# In[130]:


# settings of tree model for Regression problem
reg_tree = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 5, ccp_alpha = 0.01)


# In[131]:


# train the tree model
tree_res = reg_tree.fit(X_train[['Years', 'Hits', 'League_N']], logy_train)


# In[132]:


# Plot the tree
fig = plt.figure(figsize = (25, 20))
plot_tree(tree_res, feature_names = ['Years', 'Hits', 'League_N'])


# > consider this Tree, the **Years** is the most important variable to predict **Salary**, because it comes in the root node

# In[133]:


# another tree
reg_tree = DecisionTreeRegressor(max_depth = 5, min_samples_leaf = 2, ccp_alpha = 0.001)
tree_res = reg_tree.fit(X_train[['Years', 'Hits', 'League_N']], logy_train)

# Plot the tree
fig = plt.figure(figsize = (25, 20))
plot_tree(tree_res, feature_names = ['Years', 'Hits', 'League_N'])


# Decision Tree model using All variables

# In[134]:


reg_tree = DecisionTreeRegressor(max_depth = 6, min_samples_leaf = 5, ccp_alpha = 0.001)
tree_res = reg_tree.fit(X_train.iloc[:, 1:], logy_train)

# Plot the tree
fig = plt.figure(figsize = (25, 20))
plot_tree(tree_res, feature_names = X_train.iloc[:, 1:].columns)


# use k-fold Cross-Validation for tunning model's hyper-parameters

# In[135]:


# create hyper-parameters grid
max_depth = [5, 7, 10]
min_samples_leaf = [5, 10, 15]
ccp_alpha = [0.0001, 0.001, 0.01]
grid = list(itertools.product(max_depth, min_samples_leaf, ccp_alpha))  # all possible combinations of hyper-parameters

grid = pd.DataFrame(data = grid, index = range(1, 28), columns = ['max_depth', 'min_samples_leaf', 'ccp_alpha'])
grid  # 27 different state


# In[136]:


# 10-fold Cross-Validation approach
k = 10
cv_errors = pd.DataFrame(index = range(1, k + 1), columns = range(1, 28))
cv_errors


# In[137]:


for i in range(1, grid.shape[0] + 1):
    for j in range(1, k + 1):
        reg_tree = DecisionTreeRegressor(max_depth = grid.loc[i, 'max_depth'],
                                         min_samples_leaf = grid.loc[i, 'min_samples_leaf'],
                                         ccp_alpha = grid.loc[i, 'ccp_alpha'])
        tree_res = reg_tree.fit(X_train.iloc[folds != j, 1:], logy_train[folds != j])
        pred = tree_res.predict(X_train.iloc[folds == j, 1:])
        cv_errors.iloc[j - 1, i - 1] = ((logy_train[folds == j] - pred) ** 2).mean()
        
cv_errors  # Cross-Validation Errors for 27 different tree model


# In[138]:


cv_errors.mean(axis = 0)


# In[139]:


cv_errors.mean(axis = 0).argmin() + 1


# In[140]:


# the Best model's hyper-parameters
grid.iloc[cv_errors.mean(axis = 0).argmin()]


# ### Prediction on Test dataset
# use Model 7 for prediction on Test dataset

# In[141]:


reg_tree = DecisionTreeRegressor(max_depth = 5, min_samples_leaf = 15, ccp_alpha = 0.01)
tree_res = reg_tree.fit(X_train.iloc[:, 1:], logy_train)

# Plot the tree
fig = plt.figure(figsize = (25, 20))
plot_tree(tree_res, feature_names = X_train.iloc[:, 1:].columns)


# In[142]:


pred_tree = tree_res.predict(X_test.iloc[:, 1:])
pred_tree = np.exp(pred_tree)
pred_tree


# Absolute Error

# In[143]:


abs_err_tree = abs(y_test - pred_tree)


# Absolute Error mean, median, sd, IQR, max, min

# In[144]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_tree.mean(),
                                             'Median of AbsErrors': abs_err_tree.median(),
                                             'SD of AbsErrors': abs_err_tree.std(),
                                             'IQR of AbsErrors': iqr(abs_err_tree),
                                             'Min of AbsErrors': abs_err_tree.min(),
                                             'Max of AbsErrors': abs_err_tree.max()},
                                           index = ['Decision Tree Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[145]:


plt.scatter(x = y_test, y = pred_tree)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 8: Bagging
# ### Prediction on Test dataset
# use Model 8 for prediction on Test dataset

# In[146]:


from sklearn.ensemble import RandomForestRegressor

bagging_reg = RandomForestRegressor(max_features = 19, random_state = 123, n_estimators = 500)
bagging_res = bagging_reg.fit(X_train.iloc[:, 1:], logy_train)
pred_bagging = bagging_res.predict(X_test.iloc[:, 1:])
pred_bagging = np.exp(pred_bagging)
pred_bagging


# Absolute Error

# In[147]:


abs_err_bagging = abs(y_test - pred_bagging)


# Absolute Error mean, median, sd, IQR, max, min

# In[148]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_bagging.mean(),
                                             'Median of AbsErrors': abs_err_bagging.median(),
                                             'SD of AbsErrors': abs_err_bagging.std(),
                                             'IQR of AbsErrors': iqr(abs_err_bagging),
                                             'Min of AbsErrors': abs_err_bagging.min(),
                                             'Max of AbsErrors': abs_err_bagging.max()},
                                           index = ['Bagging Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[149]:


plt.scatter(x = y_test, y = pred_bagging)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 9: Random Forest

# In[150]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(max_features = 6, random_state = 123, n_estimators = 500)
rf_res = rf_reg.fit(X_train.iloc[:, 1:], logy_train)


# Importance of Variables: the percentage of increasing MSE if we remove the variable from the Tree

# In[151]:


importance = pd.DataFrame({'Importance': rf_res.feature_importances_ * 100}, index = X_train.iloc[:, 1:].columns)
importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# ### Prediction on Test dataset
# use Model 9 for prediction on Test dataset

# In[152]:


pred_rf = rf_res.predict(X_test.iloc[:, 1:])
pred_rf = np.exp(pred_rf)
pred_rf


# Absolute Error

# In[153]:


abs_err_rf = abs(y_test - pred_rf)


# Absolute Error mean, median, sd, IQR, max, min

# In[154]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_rf.mean(),
                                             'Median of AbsErrors': abs_err_rf.median(),
                                             'SD of AbsErrors': abs_err_rf.std(),
                                             'IQR of AbsErrors': iqr(abs_err_rf),
                                             'Min of AbsErrors': abs_err_rf.min(),
                                             'Max of AbsErrors': abs_err_rf.max()},
                                           index = ['Random Forest Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[155]:


plt.scatter(x = y_test, y = pred_rf)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 10: GBoost Regression

# In[156]:


from sklearn.ensemble import GradientBoostingRegressor

boosting_reg = GradientBoostingRegressor(learning_rate = 0.1,
                                         n_estimators = 1000,
                                         subsample = 1.0,
                                         max_depth = 5,
                                         min_samples_leaf = 5,
                                         random_state = 1234)
boosting_res = boosting_reg.fit(X_train.iloc[:, 1:], logy_train)


# use k-fold Cross-Validation for tunning model's hyper-parameters

# In[157]:


# create hyper-parameters grid
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 0.9]
max_depth = [1, 3, 5]
grid = list(itertools.product(learning_rate, subsample, max_depth))
grid = pd.DataFrame(data = grid, index = range(1, 28), columns = ['learning_rate', 'subsample', 'max_depth'])
grid


# In[158]:


k = 10  # 10-fold Cross-Validation
cv_errors = pd.DataFrame(index = range(1, k + 1), columns = range(1, 28))
cv_errors


# In[159]:


for i in range(1, grid.shape[0] + 1):
    for j in range(1, k + 1):
        boosting_reg = GradientBoostingRegressor(learning_rate = grid.loc[i, 'learning_rate'],
                                                 subsample = grid.loc[i, 'subsample'],
                                                 max_depth = grid.loc[i, 'max_depth'],
                                                 min_samples_leaf = 5,
                                                 n_estimators = 100,
                                                 random_state = 1234)
        boosting_res = boosting_reg.fit(X_train.iloc[folds != j, 1:], logy_train[folds != j])
        pred = boosting_res.predict(X_train.iloc[folds == j, 1:])
        cv_errors.iloc[j-1, i-1] = ((logy_train[folds == j] - pred) ** 2).mean()
cv_errors


# In[160]:


cv_errors.mean(axis = 0)


# In[161]:


cv_errors.mean(axis = 0).argmin() + 1


# In[162]:


grid.iloc[cv_errors.mean(axis = 0).argmin()]


# ### Prediction on Test dataset
# use Model 10 for prediction on Test dataset

# In[163]:


boosting_reg = GradientBoostingRegressor(learning_rate = 0.1,
                                         n_estimators = 100,
                                         subsample = 0.9,
                                         max_depth = 5,
                                         min_samples_leaf = 5,
                                         random_state = 1234)
boosting_res = boosting_reg.fit(X_train.iloc[:, 1:], logy_train)


# In[164]:


pred_boosting = boosting_res.predict(X_test.iloc[:, 1:])
pred_boosting = np.exp(pred_boosting)
pred_boosting


# Absolute Error

# In[165]:


abs_err_boosting = abs(y_test - pred_boosting)


# Absolute Error mean, median, sd, IQR, max, min

# In[166]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_boosting.mean(),
                                             'Median of AbsErrors': abs_err_boosting.median(),
                                             'SD of AbsErrors': abs_err_boosting.std(),
                                             'IQR of AbsErrors': iqr(abs_err_boosting),
                                             'Min of AbsErrors': abs_err_boosting.min(),
                                             'Max of AbsErrors': abs_err_boosting.max()},
                                           index = ['SGB Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[167]:


plt.scatter(x = y_test, y = pred_boosting)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 11: XGBoost Regression

# In[168]:


from xgboost import XGBRegressor

xgb_reg = XGBRegressor(n_estimators = 1000,
                       max_depth = 5,
                       learning_rate = 0.01,
                       subsample = 0.9,
                       colsample_bytree = 0.3,
                       reg_alpha = 0.1,
                       n_jobs = -1,
                       random_state = 1234)
xgb_res = xgb_reg.fit(X_train.iloc[:, 1:], logy_train)


# use k-fold Cross-Validation for tunning model's hyper-parameters

# In[169]:


# create hyper-parameters grid
learning_rate = [0.001, 0.01, 0.1]
colsample_bytree = [0.3, 0.6, 0.9]
max_depth = [1, 3, 5]
reg_alpha = [0, 0.1, 0.01]
reg_lambda = [0, 0.1, 0.01]
grid = list(itertools.product(learning_rate, colsample_bytree, max_depth, reg_alpha, reg_lambda))
grid = pd.DataFrame(data = grid, index = range(1, 244), columns = ['learning_rate', 
                                                                   'colsample_bytree',
                                                                   'max_depth',
                                                                   'reg_alpha',
                                                                   'reg_lambda'])
grid


# In[170]:


k = 10  # 10-fold Cross-Validation
cv_errors = pd.DataFrame(index = range(1, k + 1), columns = range(1, 244))
cv_errors


# In[171]:


for i in range(1, grid.shape[0] + 1):
    for j in range(1, k + 1):
        xgb_reg = XGBRegressor(n_estimators = 1000,
                               max_depth = grid.loc[i, 'max_depth'],
                               learning_rate = grid.loc[i, 'learning_rate'],
                               subsample = 0.9,
                               colsample_bytree = grid.loc[i, 'colsample_bytree'],
                               reg_alpha = grid.loc[i, 'reg_alpha'],
                               reg_lambda = grid.loc[i, 'reg_lambda'],
                               n_jobs = -1,
                               random_state = 1234)
        xgb_res = xgb_reg.fit(X_train.iloc[folds != j, 1:], logy_train[folds != j])
        pred = xgb_res.predict(X_train.iloc[folds == j, 1:])
        cv_errors.iloc[j - 1, i - 1] = ((logy_train[folds == j] - pred) ** 2).mean()
cv_errors


# In[172]:


cv_errors.mean(axis = 0)


# In[173]:


cv_errors.mean(axis = 0).argmin() + 1


# In[174]:


grid.iloc[cv_errors.mean(axis = 0).argmin()]


# ### Prediction on Test dataset
# use Model 11 for prediction on Test dataset

# In[175]:


xgb_reg = XGBRegressor(n_estimators = 1000,
                        max_depth = 5,
                        learning_rate = 0.01,
                        subsample = 0.9,
                        colsample_bytree = 0.3,
                        reg_alpha = 0,
                        reg_lambda = 0,
                        n_jobs = -1,
                        random_state = 1234)
xgb_res = xgb_reg.fit(X_train.iloc[:, 1:], logy_train)


# In[176]:


pred_xgb = xgb_res.predict(X_test.iloc[:, 1:])
pred_xgb = np.exp(pred_xgb)
pred_xgb


# Absolute Error

# In[177]:


abs_err_xgb = abs(y_test - pred_xgb)


# Absolute Error mean, median, sd, IQR, max, min

# In[178]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_xgb.mean(),
                                             'Median of AbsErrors': abs_err_xgb.median(),
                                             'SD of AbsErrors': abs_err_xgb.std(),
                                             'IQR of AbsErrors': iqr(abs_err_xgb),
                                             'Min of AbsErrors': abs_err_xgb.min(),
                                             'Max of AbsErrors': abs_err_xgb.max()},
                                           index = ['XGB Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[179]:


plt.scatter(x = y_test, y = pred_xgb)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, alpha = 0.9, linewidth = 2, color = 'red')


# ## Model 12: ANN

# In[180]:


# Create binary dummy variables for Categorical variables (One-Hot Encoding)
dummies = pd.get_dummies(train[['League', 'Division', 'NewLeague']])
dummies.head()


# In[181]:


# Define the features set X
X_ = train.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_train = pd.concat([X_, dummies], axis = 1)

# Define response variable
y_train = np.log(train['Salary'])


# In[182]:


X_train.head()  # features matrix


# In[183]:


y_train.head()  # response matrix


# In[184]:


# Min-Max Normalization to scale the train data
from sklearn.preprocessing import MinMaxScaler
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_train_scaled.head()


# In[185]:


X_train_scaled.describe()

# 'min' of all columns should be 0
# 'max' of all columns should be 1


# In[186]:


X_train_scaled.shape


# In[187]:


# Define the model architecture
model = keras.Sequential()  # define general structure of model
model.add(keras.layers.Dense(22, input_dim = 22, activation = 'relu'))  # input layer
model.add(keras.layers.Dense(11, activation = 'relu'))  # hidden layer 1
model.add(keras.layers.Dense(5, activation = 'relu'))  # hidden layer 2
model.add(keras.layers.Dense(1))  # output layer
model.summary()  # structure of model


# In[188]:


# Configure the model
model.compile(optimizer = 'SGD', loss = 'mean_squared_error', metrics = ['mean_squared_error'])


# In[189]:


# Fit the model on train data
model.fit(X_train_scaled, y_train, epochs = 400)


# ### Prediction on Test dataset
# use Model 12 for prediction on Test dataset

# In[190]:


# Create binary dummy variables for Categorical variables (One-Hot Encoding)
dummies = pd.get_dummies(test[['League', 'Division', 'NewLeague']])
dummies.head()


# In[191]:


# Define the features set X
X_ = test.drop(['Salary', 'League', 'Division', 'NewLeague'], axis = 1)
X_test = pd.concat([X_, dummies], axis = 1)

# Define response variable
y_test = test['Salary']


# In[192]:


X_test.head()


# In[193]:


y_test.head()


# In[194]:


# Min-Max Normalization to scale the test data
X_test_scaled = MinMaxScaler().fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
X_test_scaled.head()


# In[195]:


X_test_scaled.describe()


# In[196]:


# Evaluate model performance
model.evaluate(X_test_scaled, np.log(y_test))


# In[197]:


pred_ann = model.predict(X_test_scaled)
pred_ann = np.exp(pred_ann)
pred_ann = pd.Series(pred_ann[:, 0], index = y_test.index)
pred_ann.head()


# Absolute Error

# In[198]:


abs_err_ann = abs(y_test - pred_ann)


# Absolute Error mean, median, sd, IQR, max, min

# In[200]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors': abs_err_ann.mean(),
                                             'Median of AbsErrors': abs_err_ann.median(),
                                             'SD of AbsErrors': abs_err_ann.std(),
                                             'IQR of AbsErrors': iqr(abs_err_ann),
                                             'Min of AbsErrors': abs_err_ann.min(),
                                             'Max of AbsErrors': abs_err_ann.max()},
                                           index = ['ANN Reg']),
                              ignore_index = False)
model_comp


# Actual vs. Prediction

# In[201]:


plt.scatter(x = y_test, y = pred_ann)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Prediction')

# add 45' line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')


# > Consider the results, we should use **Ensemble-Learning** models for prediction in this Case-Study.
