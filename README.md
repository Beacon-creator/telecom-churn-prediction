# Customer Churn Prediction in the Nigerian Telecom Sector

This project was completed as part of the AI Community Africa Data Science Track capstone project. The goal is to build and compare machine learning models that predict customer churn in the Nigerian telecom sector.

## Project Overview

Customer churn is a major challenge for telecom providers because it is usually more expensive to acquire a new customer than to retain an existing one. By predicting which customers are likely to churn, a telecom company can take early action through targeted retention offers, loyalty rewards, or improved customer support.

The main question answered in this project is:

**Does Logistic Regression or Random Forest produce a higher accuracy score when predicting customer churn?**

## Datasets

The project uses two datasets:

- `telecom_demographics_nigeria.csv`: customer demographic and account information
- `telecom_usage.csv`: customer usage behaviour and churn labels

The datasets are merged using the shared `customer_id` column.

The target variable is:

- `churn = 1`: customer churned
- `churn = 0`: customer did not churn

## Tools And Libraries

The project was completed using:

- Python
- Jupyter Notebook
- pandas
- numpy
- scikit-learn

## Project Workflow

The project follows four main steps:

1. Load and merge the datasets
2. Explore the merged data and calculate churn rate
3. Preprocess the data by encoding categorical variables, dropping unsuitable columns, and scaling features
4. Train and evaluate Logistic Regression and Random Forest models

## Data Preparation

The two datasets were merged into a single DataFrame called `churn_df`.

The merged dataset contained:

- 6,500 rows
- 14 columns
- No missing values

The churn distribution was:

- Non-churned customers: 5,197
- Churned customers: 1,303

This gives a churn rate of approximately **20.05%**, meaning the dataset is imbalanced.

## Model Training

Two supervised machine learning models were trained:

- Logistic Regression
- Random Forest Classifier

The data was split into training and testing sets using an 80/20 split with `random_state=42`.

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report

The accuracy scores for each model were:

| Logistic Regression | 0.79 |
| Random Forest | 0.7885 |

Based on accuracy, Logistic Regression performed slightly better than Random Forest.

```python
higher_accuracy = "LogisticRegression"
```

## Key Insight

Although Logistic Regression had the higher accuracy score, accuracy alone is not enough for this problem. The dataset is imbalanced, with about 80% of customers not churning and only about 20% churning.

Both models had a recall and F1-score of `0.00` for class `1`, which represents churned customers. This means the models failed to correctly identify customers who actually churned.
