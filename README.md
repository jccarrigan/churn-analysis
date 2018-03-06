# Churn Analysis for a Major San Francisco Ride Sharing Company

## Motivation

A ride-sharing company (Company X) is interested in predicting rider retention.
To help explore this question, we are analyzing a sample dataset of a cohort of
users who signed up for an account in January 2014. The data was pulled on July
1, 2014; we consider a user retained if they were “active” (i.e. took a trip)
in the preceding 30 days (from the day the data was pulled). In other words, a
user is "active" if they have taken a trip since June 1, 2014.

The goal is to help understand **what factors are
the best predictors for retention**, and offer suggestions to operationalize
those insights to help Company X. 

**Note:** Due to the sensitive nature of the data in question, the company name and data will not be included in this analysis.

# Data 

Here is a detailed description of the data:

- `city`: city this user signed up in phone: primary device for this user
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD`
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday


## Data Cleaning

Describe cleaning process

 - How did you compute the target?

## Exploratory Data Analysis 

Scatter Matrix

Violin Plot 

Correlation Matrix/HeatMap

## Modeling 

Describe models used, tested
  - What model did you use in the end? Why?
  - Alternative models you considered? Why are they not good enough?
  - What performance metric did you use to evaluate the *model*? Why?
  

| --------------------------------------|:-----:|  
| LogisticRegression Accuracy:          | 0.736 |
| LogisticRegression Precision:         | 0.736 |   
| LogisticRegression Recall:            | 0.847 | 
| LogisticRegression F1:                | 0.787 |
| --------------------------------------|:-----:|  
| AdaBoostClassifier Accuracy:          | 0.804 |
| AdaBoostClassifier Precision:         | 0.804 |
| AdaBoostClassifier Recall:            | 0.862 |   
| AdaBoostClassifier F1:                | 0.832 |
| --------------------------------------|:-----:|  
| GradientBoostingClassifier Accuracy:  | 0.807 |
| GradientBoostingClassifier Precision: | 0.807 |
| GradientBoostingClassifier Recall:    | 0.862 |
| GradientBoostingClassifier F1:        | 0.834 |
| --------------------------------------|:-----:|  
| RandomForestClassifier Accuracy:      | 0.807 |
| RandomForestClassifier Precision:     | 0.807 |
| RandomForestClassifier Recall:        | 0.808 |
| RandomForestClassifier F1:            | 0.807 |
| --------------------------------------|:-----:|  

## Results

| --------------------------------------|:-----:|
GradientBoostingClassifier Accuracy:    | 0.807 |
GradientBoostingClassifier Precision:   | 0.807 |
GradientBoostingClassifier Recall:      | 0.869 |
GradientBoostingClassifier F1:          | 0.837 |
| --------------------------------------|:-----:|

ROC plot

Feature Importance

Partial Dependency Plot

## Conclusions

Wrap up with recommendations for company 

- Based on insights from the model, what actionable plans do you propose to
    reduce churn?
  - What are the potential impacts of implementing these plans or decisions?
    What performance metrics did you use to evaluate these *decisions*, why?
   
 

  
  
