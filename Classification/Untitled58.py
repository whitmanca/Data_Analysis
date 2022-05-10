#!/usr/bin/env python
# coding: utf-8

# # D209 Task 2: Decision Tree Regressor
# ### By Chase Whitman

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Set pandas option to view all columns
pd.set_option('display.max_columns', None)


# ## Read in data

# In[2]:


df = pd.read_csv(r"C:\Users\chase\OneDrive\Documents\WGU\Courses\D208\churn_clean.csv")


# In[3]:


df.shape


# In[4]:


df.columns


# ## Data Preprocessing

# In[5]:


print('Shape of dataframe:')
print('Before drop:', df.shape)

# Drop columns that contain unique values
unique_columns = ['CaseOrder', 'Customer_id', 'Interaction', 'UID']

# Multiple columns are used to decribe location. Timezone will be used as a regional reference.
# The additional location columns will be dropped.
location_columns = ['City', 'State','County', 'Zip', 'Lat', 'Lng', 'Job']

# Apply the drop
df.drop(columns=(unique_columns+location_columns), inplace=True)

print('After drop:', df.shape)


# In[6]:


# Columns that contain binary values (Y/N)
binary_columns = ['Churn', 'Techie', 'Port_modem', 'Tablet', 'Phone', 'Multiple', 
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

# Convert values to 1s/0s
for col in binary_columns:
    df[col] = df[col].map(dict(Yes=1, No=0))

# Examine results
df[binary_columns].head()


# In[7]:


# One hot encode columns with multiple unique values
dummy_columns = ['Area', 'TimeZone']
df = pd.get_dummies(df, columns=dummy_columns, drop_first=True)


# In[8]:


# One hot encode Internet Service based on Fiber Optic and DSL values
df['InternetService_FiberOptic'] = np.where(df.InternetService == 'Fiber Optic', 1, 0)
df['InternetService_DSL'] = np.where(df.InternetService == 'DSL', 1, 0)

# Examine results
print(df[['InternetService', 'InternetService_DSL', 'InternetService_FiberOptic']].head(8))

# Drop original Internet Service column
df.drop(columns='InternetService', inplace=True)


# In[9]:


# One hot encode Gender based on Male and Female
df['Gender_Male'] = np.where(df.Gender == 'Male', 1, 0)
df['Gender_Female'] = np.where(df.Gender == 'Female', 1, 0)

# Examine results
print(df[['Gender', 'Gender_Male', 'Gender_Female']].head(10))

# Drop original Gender column
df.drop(columns='Gender', inplace=True)


# In[10]:


# Reduce Marital categories into two types: 
#   Married:1 
#   Not Married:0
df['Marital'] = np.where(df.Marital == 'Married',1,0)


# In[11]:


# One hot encode Payment Method based on automatic payments or checks
df['PaymentMethod_Auto'] = np.where(np.logical_or(df.PaymentMethod == 'Credit Card (automatic)', 
                       df.PaymentMethod == 'Bank Transfer(automatic)'), 1, 0)
df['PaymentMethod_Check'] = np.where(np.logical_or(df.PaymentMethod == 'Mailed Check', 
                       df.PaymentMethod == 'Electronic Check'), 1, 0)

# Examine results
print(df[['PaymentMethod', 'PaymentMethod_Auto', 'PaymentMethod_Check']].head(8))

# Drop original Payment Method column
df.drop(columns='PaymentMethod', inplace=True)


# In[12]:


# Convert Contract type into numerical values based on the year value
df['Contract'] = df['Contract'].replace({'Month-to-month':0, 'One year':1, 'Two Year':2})


# In[13]:


# Examine Results
df.head()


# In[ ]:





# ## Split data

# In[14]:


X = df.drop('Tenure',1).values
y = df['Tenure'].values


# In[15]:


# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.3, 
                                                            random_state=42)
print('Training data: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Test data: X:{}, y:{}'.format(X_test.shape, y_test.shape))


# In[ ]:





# ## Decision Tree Regression

# In[16]:


# Create pipeline steps
steps = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(LassoCV())),
    ('regressor', DecisionTreeRegressor(random_state=101))
])
# Define grid search parameters
parameters = {'regressor__max_depth': np.arange(1,100)}

grid_dtr = GridSearchCV(steps, parameters)
grid_dtr.fit(X_train, y_train)


# In[17]:


# Evaluate grid search parameter results
fs = grid_dtr.best_estimator_.named_steps['feature_selection']
features_names = df.drop(columns='Tenure').columns.to_list()
features = np.array(features_names)
selected_features = features[fs.get_support()]
print('The feature selection step eliminated {} features from the model.'.format(len(features_names) - len(selected_features)))
print('The following features were selected:\n', selected_features)

print('\nThe grid search determined that a max_depth value of {} produces the best model.'.format(
grid_dtr.best_estimator_.named_steps['regressor'].max_depth))


# In[ ]:





# ## Model Evaluation

# In[18]:


def print_results(predictions):
    print('MAE: {}'.format(mean_absolute_error(y_test, predictions)))
    print('MSE: {}'.format(mean_squared_error(y_test, predictions)))
    print('R-squared: {}'.format(r2_score(y_test, predictions)))


# In[19]:


# Make predictions and print results
y_pred = grid_dtr.predict(X_test)

print_results(y_pred)


# In[20]:


# Plot decision tree
plt.figure(figsize=(14,14))
plot_tree(grid_dtr.best_estimator_.named_steps['regressor'], 
          max_depth=3, feature_names=selected_features, fontsize=12)
plt.show()


# In[ ]:





# ### Export datsets to Excel

# In[21]:


# Create training set dataframe
# df_train = pd.DataFrame(data=X_train_raw, columns=df.drop('Churn',1).columns)
# df_train['Churn'] = pd.Series(y_train)

# Create test set dataframe
# df_test = pd.DataFrame(data=X_test_raw, columns=df.drop('Churn',1).columns)
# df_test['Churn'] = pd.Series(y_test)


# In[22]:


# Export each dataframe to excel on different sheets
# writer = pd.ExcelWriter(r'C:\Users\chase\OneDrive\Documents\WGU\Courses\D209\D209_datsets.xlsx', engine='xlsxwriter')
# frames = {'Cleaned full': df,
#          'Training set': df_train,
#          'Test set': df_test}
# for sheet, frame in frames.items():
#     frame.to_excel(writer, sheet_name=sheet)
# writer.save()

