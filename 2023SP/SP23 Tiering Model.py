#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# 
# [1. SQL](#sql)<br>
# [2. EDA](#eda)<br>
# [3. Modeling](#modeling)<br>
# [4. Evaluation](#eval)<br>

# In[15]:


import pandas as pd
import numpy as np
# from sklearn import preprocessing
from pycaret import classification
import datetime
import matplotlib.pyplot as plt 
import os 
# plt.rc("font", size=14)
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import sklearn.metrics as metrics
import seaborn as sns
from sklearn.metrics import brier_score_loss
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)
# from pycaret.classification import *
# %matplotlib inline
import pyodbc


# <a id='sql'></a>
# # SQL Query
# [Top](#top)<br>

# In[2]:


# Read in the retention model SQL output - contains prior 4 terms of data, including upcoming/current term
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=10.27.251.83;'
                      'Database=OpenBook;'
                      'Trusted_Connection=yes;')
retention_query = open(r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\retention_model_obdata_23SP.sql', 'r')
df = pd.read_sql_query(retention_query.read(),conn)
conn.close()


# In[5]:


# Feature engineering placement tests to get normalized highest placement test score
sc_X = StandardScaler()
sc_X = sc_X.fit_transform(df[['aleks', 'sat_english', 'sat_math', 'act_composite', 'cccrtw']])
sc_X = pd.DataFrame(data=sc_X, columns=['aleks_scaled', 'sat_english_scaled', 'sat_math_scaled', 'act_composite_scaled'
                                        , 'cccrtw_scaled'])
data_scaled = pd.concat([df, sc_X], axis = 1)
data_scaled['max_placement_score'] = data_scaled[['aleks_scaled','sat_english_scaled', 'sat_math_scaled', 'act_composite_scaled'
                                        , 'cccrtw_scaled']].max(axis=1)


# In[6]:


# Split into data for training set and current term data to create predictions
data_train = pd.DataFrame(data_scaled[data_scaled['Start Term'].isin(['2022SP', '2020SP', '2021SP'])].reset_index(drop = True))
data_current = pd.DataFrame(data_scaled[data_scaled['Start Term'] == '2023SP'].reset_index(drop = True))


# In[ ]:


# Save csv file as backup for future evaluation
# get the current date
today = datetime.datetime.now().strftime("%Y-%m-%d")

# save the DataFrame to a CSV file with the current date in the filename
data_train.to_csv(r"C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\evaluation_data\data_train_{}.csv".format(today))


# In[7]:


print(data_train.shape)
print(data_current.shape)


# <a id='eda'></a>
# 
# # EDA
# [Top](#top)<br>

# In[ ]:


# Variables to test include:

# new or continuing student
# placement tests
# waived out of placement test
# home college (DA, KK, MX, OH, TR, WR, other, ref = HW)
# first reg date
# gender (male, other, ref = female)
# age
# ethnicity (black, asian, white, other, ref = hispanic)
# star status (free tuition)
# degree (certificate, terminal, coursetaker, ref=transfer)
# full time status
# ever early college
# athletic indicator
# pell eligibility status


# In[8]:


data_train.describe()


# In[9]:


# see which variables might show large variation between retained/not retained
data_train.groupby('Retained').mean()


# In[16]:


# check for null values
plt.figure(figsize=(20,14))
sns.heatmap(data_train.isnull().T, cbar = False)


# In[17]:


# Visualize retained/not retained 
sns.countplot(x='Retained',data=data_train, palette='hls')


# In[18]:


# view distribution for different numeric variables by changing the x variable
sns.histplot(data = data_train, x='count_tests', binwidth = 1)


# In[20]:


# view counts for categorical variables by changing the x variable
sns.countplot(x = 'ptest_exists', data = data_train)


# In[21]:


# Visualize correlations between data
plt.figure(figsize=(20,14))
sns.heatmap(data_train.corr())


# <a id='modeling'></a>
# 
# # Modeling
# [Top](#top)<br>

# In[22]:


# bring in training data snapshot
data_train = pd.read_csv(r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\evaluation_data\data_train_2023-02-03.csv',index_col=0)
data_train.head()


# In[23]:


# Notes on pycaret classification modeling:
# data split into train/test set
# numeric/categorical features inferred (should have been correctly set up by SQL query)
# 10 fold cross validation used
# numeric imputation done using mean value, categorical done using mode
# 70/30 imbalanced data set corrected via SMOTE

from pycaret.classification import *
classification_setup = classification.setup(data = data_train, fix_imbalance = True, target='Retained'
                                            , ignore_features = ['student id', 'Start Term', 'aleks', 'sat_english', 'sat_math', 'act_composite', 'cccrtw', 'aleks_scaled'
                                            , 'sat_english_scaled', 'sat_math_scaled', 'act_composite_scaled', 'cccrtw_scaled', 'Gateway/Bridge status', 'Home_College']
                                            , numeric_features = ['count_tests'], normalize = True, session_id = 1
                                           )


# In[141]:


# view transformed data if desired
# get_config("X")


# In[24]:


# testing
classification.compare_models(include = ['lightgbm', 'xgboost', 'gbc', 'catboost'])


# In[ ]:


# tests different sklearn classification models to determine high performers
classification.compare_models(include = ['lightgbm', 'xgboost', 'gbc', 'catboost'])


# In[25]:


# testing
created_model = create_model('catboost')


# In[ ]:


# use highest performer
created_model = create_model('catboost')


# In[26]:


# testing 
tuned_model = tune_model(created_model)


# In[ ]:


tuned_model = tune_model(created_model)


# In[28]:


# testing
evaluate_model(tuned_model)


# In[ ]:


# review model metrics
evaluate_model(tuned_model)


# In[7]:


# visualize how different variables affect the prediction
interpret_model(tuned_model)


# In[8]:


# if you want to see how the model makes a prediction for a specific student, change the observation number to that student row in the training dataset
interpret_model(tuned_model, plot = 'reason', observation = 0)


# In[39]:


# create final model to be used on current term data, as well as evaluation plots of training data for comparison to current term data after the term ends
final_model = finalize_model(tuned_model)
save_model(final_model, 'sp23_retention_model_TODAYSDATE')


# In[431]:


# save plots for later evaluation - 
plot_model(final_model, plot = 'calibration', save = True) # save calibration curve
plot_model(final_model, plot = 'confusion_matrix', save = True) # save confusion matrix

# move the files to the evaluation folder
foldername = r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\evaluation_data'
conmatrixfile = r'Confusion Matrix.png'
calibrationfile = r'Calibration Curve.png'
os.rename(conmatrixfile, os.path.join(foldername, conmatrixfile))
os.rename(calibrationfile, os.path.join(foldername, calibrationfile))

# rename the files
os.rename(os.path.join(foldername, conmatrixfile), os.path.join(foldername, 'confusion_matrix_TODAYSDATE.png'))
os.rename(os.path.join(foldername, calibrationfile), os.path.join(foldername, 'calibration_curve_TODAYSDATE.png'))


# <a id='eval'></a>
# 
# # Evaluation of Training vs Production outcomes
# ##### Compares training data evaluation metrics to current term predictions. To be run once current term retention numbers start posting
# - [Evaluation metrics (Accuracy, AUC, precision/recall)](#eval_metrics)
# - [Confusion Matrix](#confusion)
# - [Calibration curve of test v actual](#calibration)
# - [Brier score of test v actual](#brier) <br>
# 
# [Top](#top)

# In[29]:


# model used for training
final_model = load_model('sp23_retention_model_20230203')

# load data that was used to train the model to get evaluation metrics
data_train_snapshot = pd.read_csv(r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\evaluation_data\data_train_2023-02-03.csv',index_col=0)

# load data that was fed into the final model and used to tier students at the start of the current term
current_term_eval_set = pd.read_csv(r'C:\Users\dmcdonnell2\OneDrive - City Colleges of Chicago\Python Projects\git\retention_model\2023SP\evaluation_data\current_term_eval_snapshot_2023-02-03.csv',index_col=0)

# re-run retention query to get retention actuals
# df = pd.read_sql_query(retention_query.read(),conn)
df['student id'] = df['student id'].astype(int)

# simulate retention actuals with below code
df['Retained'] = np.random.choice([0,1], size=len(df), p = [.4, .6]) # change p for simulated retention outputs. p = [.4, .6] means 40% chance of not retained, 60% chance of retained

# merge retention actuals into snapshot from start of the term that got predicted retention scores
ret_actuals = pd.merge(current_term_eval_set, df[['student id', 'Start Term', 'Retained']], how = 'left', on = ['student id', 'Start Term'], suffixes = ['_old', ''])
ret_actuals['Retained'] = ret_actuals['Retained'].fillna(0)
ret_actuals['Retained'] = ret_actuals['Retained'].astype(int)

# get dataframe for model predictions
ret_actuals_model = ret_actuals.drop(['Retained_old', 'Label', 'Score_0', 'Score_1', 'hc_percentile', 'tier'], axis=1)


# ##### Evaluation metrics (Accuracy, AUC, precision/recall) <a id='eval_metrics'></a>

# In[30]:


# Training data metrics using snapshot and final model
training_metrics = predict_model(final_model, raw_score = True, data = data_train_snapshot)

# Current term data using predicted vs. actual retention rates
current_set = predict_model(final_model, raw_score = True, data = ret_actuals_model)


# ##### Confusion Matrix <a id='confusion'></a> <br>

# ##### Training set
# ![title](evaluation_data/confusion_matrix_2023-02-03.png)

# ##### Current term data

# In[31]:


plt.figure(figsize=(7, 4))
sns.heatmap(pd.crosstab(ret_actuals['Retained'], ret_actuals['Label']), annot = True, fmt ='0')
plt.xlabel('Predicted Retained')
plt.ylabel('Actual Retained')
plt.show()


# ##### Calibration curve <a id='calibration'></a>

# 
# ##### Training set
# ![title](evaluation_data/calibration_curve_2023-02-03.png)

# In[32]:


# First, sort the dataframe by the 'Score' column
ret_actuals.sort_values(by='Score_1', inplace=True)

# Create bins of equal size
bins = np.linspace(0, 1, 11)# min(ret_actuals['Score_1']), max(ret_actuals['Score_1']), 10)

# Create a new column for the binned scores
ret_actuals['Score_Binned'] = pd.cut(ret_actuals['Score_1'], bins)

# Calculate the average score for each bin
bin_avg = ret_actuals.groupby('Score_Binned').mean()['Retained']
bin_avg_predicted = ret_actuals.groupby('Score_Binned').mean()['Score_1']

# create figure
plt.figure(figsize=(14,7))

# Plot calibration curve
plt.plot(bin_avg_predicted, bin_avg, marker='o', linestyle='--', color='b')

# Plot reference line
plt.plot([0, 1], [0, 1], marker='o', linestyle='--', color='k')

# Add labels
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('Predicted Positive Rate', fontsize=14)

# Add title
plt.title('Calibration Curve', fontsize=16)

# Add legend
plt.legend(loc="lower right", fontsize=14)

plt.show()


# ##### Brier score <a id='brier'></a>

# In[33]:


# calculate brier score on training set (compares retention rate to predicted retention rate in training set)
brier_score_train = brier_score_loss(training_metrics['Retained'], training_metrics['Score_1'])

# calculate reference brier score on training set (compares retention rate to reference rate in training set)
training_metrics['Retained_reference_rate'] = .61 # manually adjust this value to equal the average retention rate in the training set, typically 60-75%
brier_score_train_ref = brier_score_loss(training_metrics['Retained'], training_metrics['Retained_reference_rate'])

# calculate brier skill score on training set (compares brier score train to brier score train ref to see if model is more skilled than simply imputing average retention rate)
bss_train = 1 - (brier_score_train / brier_score_train_ref)

print('Training data - brier score = ', round(brier_score_train, 3))
print('Training data - brier score reference (', training_metrics['Retained_reference_rate'][0]*100, '% reference retention rate) = ', round(brier_score_train_ref, 3), sep = '')
print('Training data - brier skill score = ', round(bss_train, 3))


# In[34]:


# calculate brier score on current term data (compares retention rate to predicted retention rate for current term)
brier_score_actual = brier_score_loss(ret_actuals['Retained'], ret_actuals['Score_1'])

# calculate reference brier score on training set (compares retention rate to reference rate in training set)
ret_actuals['Retained_reference_rate'] = .61 # manually adjust this value to equal the average retention rate in the training set, typically 60-75%
brier_score_actual_ref = brier_score_loss(ret_actuals['Retained'], ret_actuals['Retained_reference_rate'])

# calculate brier skill score on training set (compares brier score train to brier score train ref to see if model is more skilled than simply imputing average retention rate)
bss_actual = 1 - (brier_score_actual / brier_score_actual_ref)

print('Current term data - brier score = ', round(brier_score_actual, 3))
print('Current term data - brier score reference (', ret_actuals['Retained_reference_rate'][0]*100, '% reference retention rate) = ', round(brier_score_actual_ref, 3), sep = '')
print('Current term data - brier skill score = ', round(bss_actual, 3))

