#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all libraries and filters

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ### LOADING THE DATA :

# In[3]:


# Reading the file :

housing_df = pd.read_csv('E:\Housing real estate DELHI.csv')
housing_df.head()


# ###  Shape of the dataframe:

# In[4]:


housing_df.shape


# ### info of dataframe:

# In[5]:


housing_df.info()


# ### describe the dataframe:

# In[6]:


housing_df.describe (percentiles = [0.10,0.25, 0.50, 0.75, 0.90, 0.99])
                                


# ## Visualizing the data :

# In[7]:


# a pair plot of numerical variable :

sns.pairplot(data = housing_df)


# In[8]:


# Categorical varibale:

plt.figure(figsize = (20,8))

plt.subplot(2,3,1)
sns.boxplot(x='mainroad', y = 'price' , data = housing_df)

plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom' , y = 'price' , data = housing_df)

plt.subplot(2,3,3)
sns.boxplot( x= 'basement' , y = 'price' , data = housing_df)

plt.subplot(2,3,4)
sns.boxplot(x= 'hotwaterheating', y = 'price' , data = housing_df)

plt.subplot(2,3,5)
sns.boxplot(x= 'airconditioning' , y = 'price', data = housing_df)
            
plt.subplot(2,3,6)
sns.boxplot( x= 'prefarea' , y = 'price' , data = housing_df)


# In[9]:


sns.boxplot(x = 'furnishingstatus', y = 'price' , data = housing_df)


# In[10]:


plt.figure(figsize=(10,5))
sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing_df)


# In[11]:


sns.heatmap(housing_df.corr(), annot= True)


# #### here we see high correlation between price  , area and bathrooms. 

# # Data Prepration

# In[12]:


# converting yes to 1 and No to 

variable_list =['mainroad','guestroom', 'basement', 'hotwaterheating', 'airconditioning','prefarea']
def binary_map(x) :
    return x.map({'yes' :1 , 'no' : 0})

housing_df[variable_list] = housing_df[variable_list].apply(binary_map)


# In[13]:


housing_df.head()


# ### Dummy variables:

# In[14]:


# Lets us create a dummy variable for furnishing status as 3 level values

status = pd.get_dummies(housing_df['furnishingstatus'], drop_first = True)
status.head()


# In[15]:


housing_df = pd.concat([housing_df, status], axis = 1)


# In[16]:


housing_df.head()


# In[17]:


housing_df.drop(['furnishingstatus'], axis = 1, inplace = True)


# In[18]:


housing_df.head()


# # Splitting the data into Test Train Split

# In[19]:


df_train, df_test = train_test_split(housing_df, train_size =0.7, test_size =0.3, random_state = 100)


# In[20]:


df_train.shape


# # Rescaling the features:

# In[21]:


scaler = MinMaxScaler()

# applying the scaler only to below variable
num_var = ['price', 'area','bedrooms','bathrooms','stories','parking']

df_train[num_var] = scaler.fit_transform(df_train[num_var])


# In[22]:


df_train.head()


# In[23]:


df_train.describe()


# All the values are in the range o and 1.

# In[24]:


# Lets us check the correlation of train data :

plt.figure(figsize = (10,8))
sns.heatmap(df_train.corr(), annot = True, cmap= 'YlGnBu')


# ### we see high correlation between price and area , price and bathrooms, bedrooms and stories and many more. 

# # Dividing X and Y for model building:

# In[25]:


y_train = df_train.pop('price')
x_train = df_train


# # Building a linear model :

# ### We will be using two methods : 1 . using statsmodels.api 2. using RFE

# # Method 1 : Using statsmodels.api

# In[26]:


import statsmodels.api as sm


# In[27]:


# area

x_train_sm = sm.add_constant(x_train[['area']])

lr_1 = sm.OLS(y_train, x_train_sm).fit()


# In[28]:


lr_1.params


# In[29]:


print(lr_1.summary())

Variable Area just explains 28% variance
# In[30]:


plt.scatter(x_train_sm.iloc[:,1], y_train)
plt.plot(x_train_sm.iloc[:,1], 0.126894 + 0.462192*x_train_sm.iloc[:,1], 'r')
plt.show()


# Through the line is passing through the data , we see that area could explain only 28% variance. so let us add another variable

# In[31]:


# Area and bathjrooms

x_train_sm  = sm.add_constant(x_train[['area', 'bathrooms']])

lr_2 = sm.OLS(y_train, x_train_sm).fit()


# In[32]:


lr_2.params


# In[33]:


print(lr_2.summary())


# Adjusted R-squared increased from 28.1% to 47.7% . let us add one more variable and check:

# In[34]:


# Area , bathrooms, and bedrooms

x_trains_sm = sm.add_constant(x_train[['area', 'bedrooms', 'bathrooms']])

lr_3 = sm.OLS(y_train, x_train_sm).fit()


# In[35]:


lr_3.params


# In[36]:


print(lr_3.summary())


# ### Lets us do the other way - Let us build the model by adding all the variables to the model and drop those which are insignificant: 

# In[37]:


x_train.columns


# In[38]:


x_train_sm = sm.add_constant(x_train)
lr_4 = sm.OLS(y_train, x_train_sm).fit()


# In[39]:


lr_4.params


# In[40]:


print(lr_4.summary())


# We see that , certain variables have p-values > 0.05. Before dropping any variables, lets us check VIF as well :

# # VIF:

# In[41]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[42]:


vif = pd.DataFrame()
vif["Features"] = x_train.columns
vif["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range (x_train.shape[1])]
vif["VIF"] = round(vif["VIF"], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# Lets us drop variable semi-furnished as  p-value of semi-furnished is 0.938

# In[43]:


# Dropping the variable semi- furnished

x = x_train.drop('semi-furnished', axis = 1)


# In[44]:


x.columns


# In[45]:


x_sm = sm.add_constant(x)
lr_5 = sm.OLS(y_train, x_sm).fit()


# In[46]:


print(lr_5.summary())


# Now Bedrooms and Basement looks insignificant . lets us check VIF

# In[47]:


vif = pd. DataFrame()
vif["Fetaures"] = x.columns
vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[48]:


# Lets us drop bedrooms

x = x.drop('bedrooms', axis = 1)


# In[49]:


x_sm = sm.add_constant(x)
lr_6 = sm.OLS(y_train, x_sm).fit()


# In[50]:


print(lr_6.summary())


# P-values of all variables looks fine . let us check VIF.

# In[51]:


vif= pd.DataFrame()
vif['Features'] = x.columns
vif['VIF']= [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# # Residual Analysis Of Train Data

# In[52]:


y_train_pred = lr_6.predict(x_sm)


# In[53]:


residual = y_train- y_train_pred


# In[54]:


sns.distplot(residual, bins = 20)


# Error terms are normally distributed.

# ### Making Prediction Using The Final Model:

# In[55]:


# these variables we scalled in Train data ... so let us scale the same variables in test data as well .

num_var =['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']


df_test[num_var]= scaler.transform(df_test[num_var])


# In[56]:


df_test.describe()


# In[57]:


y_test = df_test.pop('price')
x_test = df_test


# In[58]:


x_test_sm = sm.add_constant(x_test)


# In[59]:


x_test_sm = x_test_sm.drop(['semi-furnished', 'bedrooms'], axis = 1)


# In[60]:


y_pred = lr_6.predict(x_test_sm)


# # Model Evaluation :

# In[61]:


plt.scatter(y_test, y_pred)


# In[62]:


lr_6.summary()


# # Method 2 : Using RFE:

# ### Splitting The Data Into Train Split

# In[63]:


df_train, df_test = train_test_split(housing_df, train_size = 0.7, test_size = 0.3,random_state= 100)


# In[64]:


df_train.shape


# In[65]:


df_test.shape


# ### Scaling Of The Data:

# In[66]:


var_list =['price','area','bedrooms','bathrooms','stories','parking']
scaler = MinMaxScaler()

df_train[var_list] = scaler.fit_transform(df_train[var_list])


# In[67]:


df_train.describe()


# ### Dividing X and Y Model Building:

# In[68]:


y_train = df_train.pop('price')
x_train = df_train


# # RFE

# In[69]:


from sklearn.feature_selection import RFE


# In[70]:


from sklearn.linear_model import LinearRegression


# In[71]:


lm = LinearRegression()
lm.fit(x_train, y_train)


rfe = RFE(lm,n_features_to_select=10)
rfe = rfe.fit(x_train, y_train)


# In[72]:


list(zip(x_train.columns,rfe.support_, rfe.ranking_))


# In[73]:


support_col = x_train.columns[rfe.support_]
support_col


# In[74]:


discarded_col = x_train.columns[~rfe.support_]
discarded_col


# ### Building The Model Using Supported Columns:

# In[75]:


x_train_rfe = x_train[support_col]


# In[76]:


x_train_rfe_sm = sm.add_constant(x_train_rfe)


# In[77]:


lr_rfe = sm.OLS(y_train, x_train_rfe_sm).fit()


# In[78]:


lr_rfe.params


# In[ ]:





# In[79]:


print(lr_rfe.summary())


# Variable bedrooms is significant

# In[80]:


x_train_rfe_1 = x_train_rfe.drop(['bedrooms'], axis = 1)


# In[81]:


x_train_rfe_new = sm.add_constant(x_train_rfe_1)


# In[82]:


lr_rfe_1 = sm.OLS(y_train, x_train_rfe_new).fit()


# In[83]:


print(lr_rfe_1.summary())


# All the P -values looks significant. let us check VIF

# In[84]:


vif = pd.DataFrame()
vif['Feature']= x_train_rfe_1.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe_1.values,i) for i in range (x_train_rfe_1.shape[1])]
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ### Residual Analysis

# In[85]:


y_train_pred = lr_rfe_1.predict(x_train_rfe_new)

res = y_train - y_train_pred


# In[86]:


sns.distplot(res, bins= 20)


# Error terms are normally distributed.

# ## Making Predictions Using The Final Model

# In[87]:


var_list =['price', 'area','bedrooms','bathrooms','stories','parking']

df_test[var_list] = scaler.transform(df_test[var_list])


# In[88]:


y_test = df_test.pop('price')
x_test = df_test


# In[89]:


col = x_train_rfe_1.columns


# In[90]:


x_test_new = x_test[col]


# In[91]:


x_test_new.columns


# In[92]:


x_test_rfe = sm.add_constant(x_test_new)


# In[93]:


y_pred = lr_rfe_1.predict(x_test_rfe)


# ### Model Evaluation

# In[94]:


plt.scatter(y_test, y_pred)


# In[95]:


print(lr_rfe_1.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




