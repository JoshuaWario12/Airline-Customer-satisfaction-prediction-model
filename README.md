 **Airline Customer Satisfaction Prediction**

**Project Overview**

This repository contains a complete end-to-end machine learning pipeline for predicting airline customer satisfaction. The goal is to classify each passenger as either “Satisfied” or “Dissatisfied” based on demographic information, flight details, and in-flight service ratings. A Random Forest classifier is used to achieve high accuracy and robustness.

**Table of Contents**

1. Dataset
2. Getting Started
3. Notebook Summary

   3.1 Introduction

   3.2 Exploratory Data Analysis and Data Transformation

   3.3 Feature Engineering and Encoding

   3.4 Model Training (Random Forest)

   3.5 Initial Model Evaluation

   3.6 Model Improvement with Hyperparameter Tuning

   3.7 Final Model Evaluation
   
4. Results

---

**1. Dataset**
   
   The dataset used in this project was obtained from Kaggle. Each row represents a single passenger’s feedback, containing:

* Demographic Information: age, gender, customer type, type of travel
* Flight Details: travel class, flight distance, departure delay, arrival delay
* In-Flight Service Ratings: inflight wifi service, check-in service, inflight entertainment, seat comfort, food and drink, online services (online boarding, online support), cleanliness, baggage handling
* Target Variable: satisfaction (binary: “Satisfied” or “Dissatisfied”)

Before modeling, categorical variables were encoded (label encoding for binary categories and one-hot encoding for multi-class features), and missing values (for example, in Arrival Delay in Minutes) were imputed with median values.

---

**2. Getting Started**

Prerequisites

* Python version 3.7 or higher
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

Install required packages by running:

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

Repository Structure

```
data/
    Invistico_Airline.csv        (Original dataset in CSV format)

notebooks/
    predicting customer airline satisfaction.ipynb  
        (Jupyter Notebook containing all code and explanations)

README.md          (This file)
```

---

**3. Notebook Summary**

3.1 Introduction

* Problem Definition: Classify passengers as satisfied or dissatisfied.
* Chosen Algorithm: Random Forest, because it is robust to overfitting, handles non-linear relationships well, and provides strong predictive performance.
* Objectives:

  • Load and clean the dataset

  • Perform exploratory data analysis (EDA) and data transformation

  • Train a Random Forest classifier

  • Evaluate the initial model

  • Improve model performance through hyperparameter tuning

  • Evaluate and interpret the final model

3.2 Exploratory Data Analysis and Data Transformation

* Initial Exploration: Display first few rows with `head()` to understand data structure. Check data types with `dtypes` and missing values using `isnull().sum()`.
* Missing Value Handling: Fill missing values in “Arrival Delay in Minutes” with the median.
* Correlation Analysis: Plot a correlation heatmap to identify highly correlated feature pairs (for example, “Arrival Delay in Minutes” & “Departure Delay in Minutes”, “Food and Drink” & “Seat Comfort”, various online service ratings, “Cleanliness” & “Baggage Handling”). Examine class imbalance in the target variable using a count plot.

3.3 Feature Engineering and Encoding

* Target Encoding: Convert `satisfaction` (“Satisfied”/“Dissatisfied”) into numeric labels (`0` and `1`) using `LabelEncoder`.
* Categorical Feature Encoding:
  • Label-encode binary categorical columns such as `Gender`, `Customer Type`, and `Type of Travel`.
  • One-hot encode multi-class fields (for example, `Travel Class`).
* Feature Scaling: Use `StandardScaler` to standardize numerical features (flight distance, delay minutes, rating scores) so all features are on a similar scale.

3.4 Model Training (Random Forest)

* Train/Test Split: Split data into 80% training and 20% testing sets using `train_test_split` with `stratify=y` to preserve class proportions.
* Baseline Model: Train a `RandomForestClassifier` with `class_weight='balanced'` to address class imbalance. Achieve approximately 96% accuracy on initial evaluation.

3.5 Initial Model Evaluation

* Confusion Matrix (Initial Model):
  • True Negatives = 11,318
  • False Positives = 441
  • False Negatives = 657
  • True Positives = 13,560
* Classification Report (Initial Model):
  • Class 0 (Dissatisfied): Precision = 0.95, Recall = 0.96, F1-score = 0.95, Support = 11,759
  • Class 1 (Satisfied): Precision = 0.97, Recall = 0.95, F1-score = 0.96, Support = 14,217
  • Overall Accuracy = 0.96

3.6 Model Improvement with Hyperparameter Tuning

* RandomizedSearchCV Setup: Define search space for key hyperparameters—`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`. Use `RandomizedSearchCV` with 10 parameter combinations and 3-fold cross-validation to find the best combination.
* Best Parameters Found:
  • `n_estimators = 200`
  • `min_samples_split = 2`
  • `min_samples_leaf = 1`
  • `max_features = 'log2'`
  • `max_depth = None`
* Resulting Performance: Accuracy remains at 96%, but the number of misclassifications (false positives and false negatives) decreases slightly, indicating a more stable, confident model.

3.7 Final Model Evaluation

* Confusion Matrix (After Tuning):
  • True Negatives = 11,321
  • False Positives = 438
  • False Negatives = 652
  • True Positives = 13,565
* Classification Report (After Tuning):
  • Class 0 (Dissatisfied): Precision = 0.95, Recall = 0.96, F1-score = 0.95, Support = 11,759
  • Class 1 (Satisfied): Precision = 0.97, Recall = 0.95, F1-score = 0.96, Support = 14,217
  • Overall Accuracy = 0.96
* Key Insight: Although the overall accuracy remains at 96%, the number of misclassifications decreased (false positives from 441 to 438, false negatives from 657 to 652). The model is more consistent and confident after tuning.
---
**4. Results**

* Final Model Accuracy: 0.96
* Precision, Recall, and F1-Scores for both classes are all at or above 0.95.
* Hyperparameter tuning reduced misclassifications and improved model confidence.
