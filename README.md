# Bank Marketing Data Analysis

This repository contains code and documentation for analyzing **Bank Marketing** data to predict customer outcomes. The primary goal is to determine whether a client will subscribe to a term deposit (binary classification of `y`).

[In depth Paper](https://docs.google.com/document/d/1ClUWeWzALbtbg5AYqc63n704XW-FhzYmfPudI__XT5w/edit?tab=t.0)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Methods & Workflow](#methods--workflow)
4. [Mini Holdout Cheat Sheet](#mini-holdout-cheat-sheet)
5. [Scoring & Metrics](#scoring--metrics)
6. [Installation & Requirements](#installation--requirements)
7. [Usage](#usage)
8. [Limitations & Next Steps](#limitations--next-steps)
9. [License & Credits](#license--credits)

---

## Project Overview

- **Objective**: Predict whether a customer will subscribe to a bank’s term deposit (the `y` variable).
- **Dataset Size**: 37,069 records (rows).
- **Key Techniques**:
  - Data cleaning and one-hot encoding of categorical features.
  - Feature engineering (e.g., scaling of numeric columns).
  - Training and evaluating a **Decision Tree Classifier**.
  - Using a **mini holdout set** for an additional check on model performance.

---

## Data Description

Each record represents a marketing contact with a bank client. The **main features** include:

- **Demographics**: `age`, `job`, `marital`, `education`.
- **Loan & Credit**: `default` (has credit in default?), `housing` (housing loan?), `loan` (personal loan?).
- **Contact Details**: `contact` (contact type), `month`, `day_of_week`.
- **Campaign Performance**:  
  - `campaign`: number of contacts during the current campaign.  
  - `pdays`: days passed since the client was last contacted (-1 or 999 indicate no prior contact).  
  - `previous`: number of contacts prior to this campaign.  
  - `poutcome`: outcome of the previous campaign (`nonexistent`, `failure`, or `success`).
- **Economic Indicators**: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
- **Target**: `y` indicates whether the customer subscribed to a term deposit (`yes`/`no`).

**Note**: The dataset appears to have no missing values in the traditional sense (e.g., `NaN`).

---

## Methods & Workflow

1. **Data Loading & Inspection**  
   - Loaded the main dataset from a public GitHub URL.  
   - Confirmed shape, column types, and no null values.

2. **Feature Selection & Engineering**  
   - Chose certain columns (e.g., `job`, `marital`, `education`, `poutcome`, `pdays`, etc.) based on domain knowledge and correlation.  
   - Scaled numeric columns (`age`, `campaign`, `pdays`, etc.) to a `[0,1]` range for improved model performance.

3. **Encoding**  
   - Applied **One-Hot Encoding** (`pd.get_dummies`) to categorical variables.  
   - Dropped the first dummy column (`drop_first=True`) to avoid multicollinearity.

4. **Splitting**  
   - Divided data into `train` and `test` sets (70%/30% split).

5. **Model Building**  
   - **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`) with a `max_depth=5`.  
   - Trained on `X_train`, `y_train`.  
   - Checked accuracy on `X_test`, `y_test`.

6. **Tree Visualization**  
   - Used `matplotlib` and `sklearn.tree.plot_tree` to visualize the trained decision tree.

---

## Mini Holdout Cheat Sheet

To further validate the model (beyond the initial train/test split):

1. **Mini Holdout Data**  
   - A small subset (`bank_holdout_test_mini.csv` + `bank_holdout_test_mini_answers.csv`) used to check model generalization.

2. **Preparation**  
   - Loaded the mini holdout features and **applied the same transformations** (dummy encoding, scaling, etc.) as the training data.  
   - Aligned columns between the mini holdout and training set.

3. **Scoring**  
   - Ran `clf.score(prediction_set, real_test)` to measure how well the classifier performs on the mini holdout.  
   - The performance on this subset was notably **different** from the main test set, suggesting further investigation might be needed.

---

## Scoring & Metrics

- **Accuracy**: `clf.score(X_test, y_test)`  
  - E.g., ~0.896 (or ~89.6%) on the main test set.
- **Precision, Recall, F1** (for more nuanced classification analysis):  
  ```python
  from sklearn.metrics import recall_score, precision_score, f1_score
  
  predicted = clf.predict(X_test)
  precision = precision_score(y_test, predicted)
  recall    = recall_score(y_test, predicted)
  f1        = f1_score(y_test, predicted)
