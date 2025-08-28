#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import seaborn as sn
from datetime import datetime
from pybaseball import statcast, playerid_reverse_lookup


# In[15]:


today_date = datetime.today().strftime('%Y-%m-%d')

statcast_data=statcast(start_dt="2025-03-27", end_dt=today_date)
batter_ids = statcast_data['batter'].unique().tolist()

# Retrieve the batter names using playerid_reverse_lookup
batter_names_df = playerid_reverse_lookup(batter_ids, key_type='mlbam')

# Combine first and last names into a single column
batter_names_df['batter_name'] = batter_names_df['name_first'] + ' ' + batter_names_df['name_last']

# Merge the batter names into the Statcast data
merged_data = statcast_data.merge(batter_names_df[['key_mlbam', 'batter_name']], 
                                  how='left', 
                                  left_on='batter', 
                                  right_on='key_mlbam')

# Drop the redundant 'key_mlbam' column
merged_data.drop(columns=['key_mlbam'], inplace=True)
data = merged_data


# In[16]:


data.columns


# In[17]:


filt_data = data[['batter_name', 'events', 'launch_speed', 'launch_angle', 'delta_run_exp']]
filt_data.head(10)


# In[18]:


filt_data['events'].unique()


# In[19]:


## Calculating weights for outcomes using run expectancy:

event_group = filt_data.groupby('events').agg(
    avg_run_exp = pd.NamedAgg(column = 'delta_run_exp', aggfunc = 'mean'))
event_group_filt = event_group[event_group.index.isin(['single', 'double', 'triple', 'walk', 'hit_by_pitch', 'home_run'])]

# Rescaling so mean average run expectancy is 1
og_mean = event_group_filt['avg_run_exp'].mean()
event_group_filt['avg_run_rescaled'] = event_group_filt['avg_run_exp']/og_mean
event_group_filt


# In[20]:


ev_la_data = filt_data.dropna()
ev_la_data.head()


# In[21]:


# Visualizing relationship between EV and run exp:
plt.scatter(ev_la_data['launch_speed'], ev_la_data['delta_run_exp'])
plt.xlabel("Exit Velocity (mph)")
plt.ylabel("Run Expectancy Change")


# In[22]:


plt.scatter(ev_la_data['launch_angle'], ev_la_data['delta_run_exp'])
plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Run Expectancy Change")


# In[23]:


# Creating a random forest model to create "scores" for a certain launch angle and exit velocity:

# train test split (0.8)
X = ev_la_data[['launch_speed', 'launch_angle']]
y = ev_la_data['delta_run_exp']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 123)


# In[24]:


## Analyzing correlations of features
corr_matrix = X_train.corr()
plt.figure(figsize = (20,20))
plt.rcParams.update({'font.size': 40})
sn.heatmap(corr_matrix, annot = True)

# We see fairly limited correlation between EV and LA


# In[25]:


## Examining histograms and distributions of features
plt.rcParams.update({'font.size': 10})
X_train.hist(bins = 15, figsize = (12,12))


# Both are slightly left-tailed, fairly normal though


# In[26]:


# Looking at distribution of Change in Run Expectancy
y_train.hist(bins = 20)

# We see it is highly centered around -0.2 (which is about average for a normal out)


# In[27]:


# Preprocessing:
colnames = list(X_train.columns)
ct = make_column_transformer(
    (StandardScaler(), colnames))

# Fitting X_train with scaled values, transforming both X_train and X_test
transformed_X_train = ct.fit_transform(X_train)
transformed_X_test = ct.transform(X_test)

## Creating transformed data frames
X_train_transformed = pd.DataFrame(transformed_X_train, columns = colnames)
X_test_transformed = pd.DataFrame(transformed_X_test, columns = colnames)
X_train_transformed.head()


# In[28]:


# Pipeline for random forest model:
pipe_rf = make_pipeline(ct, RandomForestRegressor(random_state = 123, n_jobs = -1))


# In[29]:


## Hyperparameter optimization for random forest regressor
param_grid_rf = {"randomforestregressor__max_depth" : [2,4,6,8,10,12,14,16,18,20],
               "randomforestregressor__n_estimators" : [2,4,6,8,10,12,14,16,18,20]}
random_search_rf = RandomizedSearchCV(pipe_rf, param_grid_rf, n_iter = 100, cv = 5, n_jobs = -1,random_state = 123,
                                  scoring = 'neg_root_mean_squared_error')
random_search_rf.fit(X_train, y_train)
results_rf = pd.DataFrame(random_search_rf.cv_results_).set_index("rank_test_score").sort_index()
results_rf.T


# In[36]:


## Looking at optimized hyperparameters
params=results_rf.T.iloc[6][1]
n_estimators = params['randomforestregressor__n_estimators']
max_depth = params['randomforestregressor__max_depth']
results_rf.T.iloc[6][1]


# In[37]:


# Creating new pipeline
pipe_rf1 = make_pipeline(ct, RandomForestRegressor(random_state = 123, n_jobs = -1, n_estimators = n_estimators, max_depth = max_depth))


# In[38]:


# Fitting and calculating test score:
pipe_rf1.fit(X_train, y_train)
test_predict = pipe_rf1.predict(X_test)
test_score = np.sqrt(mean_squared_error(y_test, pipe_rf1.predict(X_test)))
test_score


# In[39]:


# Predicting on dataset values
ev_la = ev_la_data[['launch_speed', 'launch_angle']]

ev_la_preds = pipe_rf1.predict(ev_la)

# Scaling to ensure all values are > 0 using logistic function if prediction < 0 and exponential transformation
# to ensure all predictions > 0 have value > 1.

def scale_predictions(preds):
    # Scales logistically if pred < 0, exponentially + 1 if pred > 0
    scaled = np.where(preds < 0, 1 / (1 + np.exp(-preds)), np.exp(preds) + 1)
    return scaled
scaled_ev_la_preds = scale_predictions(ev_la_preds)
ev_la_data['run_coef'] = scaled_ev_la_preds
ev_la_data.head()


# In[40]:


event_group_filt = event_group_filt[['avg_run_rescaled']]
event_group_filt = event_group_filt.rename(columns = {'avg_run_rescaled':'coef'})
event_group_filt.to_csv("data/outcome_coefs.csv")
event_group_filt


# In[41]:


# Calculating plate appearances for each player

pa = filt_data.groupby('batter_name').count()
pa_tot = pa[['events']]
pa_tot.head(10)


# In[42]:


# Combining outcome coefficients with run coefficients as score contribution
def calculate_contribution(row):
    event_type = row['events']
    rc = row['run_coef']
    coefficient = event_group_filt.loc[event_type, 'coef'] if event_type in event_group_filt.index else 0
    return coefficient * rc

ev_la_data['score_contribution'] = ev_la_data.apply(calculate_contribution, axis=1)
ev_la_success = ev_la_data[ev_la_data['events'].isin(['single', 'double', 'triple', 'walk', 'hit_by_pitch', 'home_run'])]
ev_la_success.head(10)


# In[43]:


# Formula for metwOBA:
# (0.33*BB + 0.55*HBP + 0.70*1B*RC + 1.11*2B*RC + 1.40*3B*RC + 1.91*HR*RC)/PA
# Where RC is the previously calculated run coefficient from EV and LA

player_scores = ev_la_success.groupby('batter_name')['score_contribution'].sum().reset_index()
final_data = player_scores.merge(pa_tot, on='batter_name')

# Calculate final scores
final_data['metwOBA'] = final_data['score_contribution'] / final_data['events']
final_data = final_data.rename(columns = {'events':'PA', 'batter_name':'Player'})

# Filtering to plate appearances above 30% quantile 
pa_quant_30 = final_data['PA'].quantile(0.3)
final_data = final_data[final_data['PA'] > pa_quant_30]
final_data = final_data.set_index('Player')
final_data = final_data[['metwOBA']]
final_data = final_data.sort_values(by = 'metwOBA', ascending = False)
final_data.head(15)


# In[44]:


final_data.to_csv("data/metwOBAlb.csv")


# In[ ]:




