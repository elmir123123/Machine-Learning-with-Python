#!/usr/bin/env python
# coding: utf-8

# In[2]:


# timeit

# Student Name : ELMIR ALYRZAEV
# Cohort       : 4 

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

# use this space for all of your package imports
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
################################################################################
# Load Data
################################################################################

# use this space to load the original dataset
# MAKE SURE TO SAVE THE ORIGINAL FILE AS original_df
# Example: original_df = pd.read_excel('Apprentice Chef Dataset.xlsx')

original_df = pd.read_excel("Apprentice_Chef_Dataset.xlsx")

Chef = original_df
################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

# use this space for all of the feature engineering that is required for your
# final model

# if your final model requires dataset standardization, do this here as well

# setting outlier thresholds
Total_meals_hi         = 190
Unique_meals_hi        = 9
Contact_w_customer_lo  = 2
Contact_w_customer_hi  = 13
Avg_time_site_hi       = 150
Cancellation_before_hi = 5
Cancellation_after_hi  = 2
Mobile_login_lo        = 5
Mobile_login_hi        = 6
Pc_login_lo            = 1
Pc_login_hi            = 2
Weekly_plan_hi         = 30
Early_deliveries_hi    = 6
Late_deliveries_hi     = 8
Avg_prep_hi            = 280
Largest_order_lo       = 2
Largest_order_hi       = 8
Master_class_hi        = 2
Median_rating_hi       = 4
Avg_clicks_lo          = 8
Avg_clicks_hi          = 18
Total_photos_hi        = 470

#BASED ON THE BOXPLOT AND THE .95 QUANTILE REVENUE SEEMS TO CUT OFF AT 4500 
Revenue_hi             = 4500


# setting trend-based thresholds
Total_meals_change_hi     = 250 # data scatters above this point
Unique_meals_change_hi   = 9 # data scatters above this point
Contacts_w_customer_change_hi   = 10  # trend changes above this point
Avg_time_visit_change_hi  = 300 # data scatters above this point
Cancellation_before_change_hi  = 10 # data scatters above this point
Mobile_login_change_hi   = 6  # trend changes above this point
Early_deliveries_change_hi   = 3  # trend changes above this point
Late_deliveries_change_hi   = 11  # trend changes above this point
Avg_prep_vid_change_hi     = 290 # data scatters above this point
Largest_order_change_hi   = 5  # trend changes above this point
Master_classes_change_hi   = 1  # trend changes above this point
Median__meal_rating_change_hi   = 4  # trend changes above this point
Avg_clicks_per_visit_change_hi   = 10  # trend changes above this point
Total_photos_viewed_change_hi   = 500 # data scatters above this point
Weekly_plan_change_at = 16 # data scatters above this point

#####################################################
Mobile_number_change_at = 1 # only different at 5
Cancellation_after_change_at = 0 # zero inflated
Pc_login_change_at      = 1 # big changes at 1
Weekly_plan_change_at = 0 # zero inflated
Refrigerated_change_at   = 0 # zero inflated (especielly when looking at value counts 1726 zeros and 220 ones)
##############################################################################
## Feature Engineering (outlier thresholds)                                 ##
##############################################################################

# developing features (columns) for outliers


# UNIQUE_MEALS_PURCH 
Chef['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = Chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][Chef['UNIQUE_MEALS_PURCH'] > Unique_meals_hi]

Chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


# MASTER_CLASSES_ATTENDED
Chef['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = Chef.loc[0:,'out_MASTER_CLASSES_ATTENDED'][Chef['MASTER_CLASSES_ATTENDED'] > Master_class_hi]

Chef['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                 value      = 1,
                                 inplace    = True)


# MEDIAN_MEAL_RATING
Chef['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = Chef.loc[0:,'out_MEDIAN_MEAL_RATING'][Chef['MEDIAN_MEAL_RATING'] > Median_rating_hi]

Chef['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                 value      = 1,
                                 inplace    = True)

# REVENUE
Chef['out_REVENUE'] = 0
condition_hi = Chef.loc[0:,'out_REVENUE'][Chef['REVENUE'] > Revenue_hi]

Chef['out_REVENUE'].replace(to_replace = condition_hi,
                                 value      = 1,
                                 inplace    = True)



##############################################################################
## Feature Engineering (trend changes)                                      ##
##############################################################################

# developing features (columns) for outliers

# TOTAL_MEALS_ORDERED
Chef['change_TOTAL_MEALS_ORDERED'] = 0
condition = Chef.loc[0:,'change_TOTAL_MEALS_ORDERED'][Chef['TOTAL_MEALS_ORDERED'] > Total_meals_change_hi]

Chef['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition,
                                   value      = 1,
                                   inplace    = True)



# UNIQUE_MEALS_PURCH
Chef['change_UNIQUE_MEALS_PURCH'] = 0
condition = Chef.loc[0:,'change_UNIQUE_MEALS_PURCH'][Chef['UNIQUE_MEALS_PURCH'] > Unique_meals_change_hi]

Chef['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)


# CONTACTS_W_CUSTOMER_SERVICE
Chef['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0

condition = Chef.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][Chef['CONTACTS_W_CUSTOMER_SERVICE'] > Contacts_w_customer_change_hi]

Chef['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)


# AVG_PREP_VID_TIME
Chef['change_AVG_PREP_VID_TIMEE'] = 0

condition = Chef.loc[0:,'change_AVG_PREP_VID_TIMEE'][Chef['AVG_PREP_VID_TIME'] > Avg_prep_vid_change_hi]

Chef['change_AVG_PREP_VID_TIMEE'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)


# MEDIAN_MEAL_RATING
Chef['change_MEDIAN_MEAL_RATING'] = 0

condition = Chef.loc[0:,'change_MEDIAN_MEAL_RATING'][Chef['MEDIAN_MEAL_RATING'] > Median__meal_rating_change_hi]

Chef['change_MEDIAN_MEAL_RATING'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)

# AVG_CLICKS_PER_VISIT
Chef['change_AVG_CLICKS_PER_VISIT'] = 0
condition = Chef.loc[0:,'change_AVG_CLICKS_PER_VISIT'][Chef['AVG_CLICKS_PER_VISIT'] > Avg_clicks_per_visit_change_hi]

Chef['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition,
                                value      = 1,
                                inplace    = True)


# CONTACTS_W_CUSTOMER_SERVICE
Chef['change_TOTAL_PHOTOS_VIEWED'] = 0

condition = Chef.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][Chef['TOTAL_PHOTOS_VIEWED'] > Total_photos_viewed_change_hi]

Chef['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition,
                                      value      = 1,
                                      inplace    = True)



################################################################################
# Train/Test Split
################################################################################

# use this space to set up testing and validation sets using train/test split

# Note: Be sure to set test_size = 0.25

import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

x_variables = ['CROSS_SELL_SUCCESS',
'TOTAL_MEALS_ORDERED',
'UNIQUE_MEALS_PURCH',
'CONTACTS_W_CUSTOMER_SERVICE',
'AVG_TIME_PER_SITE_VISIT',
'AVG_PREP_VID_TIME',
'LARGEST_ORDER_SIZE',
'MASTER_CLASSES_ATTENDED',
'MEDIAN_MEAL_RATING',
'TOTAL_PHOTOS_VIEWED',
'out_UNIQUE_MEALS_PURCH',
'out_MASTER_CLASSES_ATTENDED',
'out_MEDIAN_MEAL_RATING',
'out_REVENUE',
'change_TOTAL_MEALS_ORDERED',
'change_UNIQUE_MEALS_PURCH',
'change_CONTACTS_W_CUSTOMER_SERVICE', 
'change_AVG_PREP_VID_TIMEE',
'change_MEDIAN_MEAL_RATING',
'change_AVG_CLICKS_PER_VISIT',
'change_TOTAL_PHOTOS_VIEWED']

Chef_data   = Chef.loc[ : , x_variables]


# Preparing the target variable
Chef_target = Chef.loc[:, 'REVENUE']

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# use this space to instantiate, fit, and predict on your final model
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.preprocessing import StandardScaler # standard scaler

# INSTANTIATING a StandardScaler() object
scaler = StandardScaler()


# FITTING the scaler with housing_data
scaler.fit(Chef_data)


# TRANSFORMING our data after fit (interpreting the data as below(negative) or above(positive) average)
X_scaled = scaler.transform(Chef_data)


# converting scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

#Indicating columns for each variable from # to "text"
X_scaled_df.columns = Chef_data.columns

# checking the results
X_scaled_df.describe().round(2)

############################################################
# TRAIN_TEST SPLIT USING THE STANDARDIZED DATA
X_train, X_test, y_train,y_test = train_test_split(
            X_scaled_df,
            Chef_target,
            test_size = 0.25,
            random_state = 222)
############################################################
# creating lists for training set accuracy and test set accuracy
training_accuracy = []
test_accuracy = []


# building a visualization of 1 to 50 neighbors
neighbors_settings = range(1, 21)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# plotting the visualization
fig, ax = plt.subplots(figsize=(12,8))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# finding the optimal number of neighbors
opt_neighbors = test_accuracy.index(max(test_accuracy)) + 1
print(f"""The optimal number of neighbors is {opt_neighbors}""")

################################################################################
################################################################################
# Final Model Score (score)
# INSTANTIATING a model with the optimal number of neighbors
knn_stand = KNeighborsRegressor(algorithm = 'auto',
                   n_neighbors = opt_neighbors)



# FITTING the model based on the training data
knn_stand.fit(X_train, y_train)



# PREDITCING on new data
knn_stand_pred = knn_stand.predict(X_test)



# SCORING the results
print('Training Score:', knn_stand.score(X_train,y_train).round(4))
print('Testing Score:',  knn_stand.score(X_test, y_test).round(4))

# saving scoring data for future use
knn_stand_score_train =  knn_stand.score(X_train,y_train).round(4)
knn_stand_score_test  = knn_stand.score(X_test, y_test).round(4)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = knn_stand_score_test



# In[ ]:




