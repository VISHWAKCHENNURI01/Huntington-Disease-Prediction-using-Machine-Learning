# Huntington-Disease-Prediction-using-Machine-Learning

## Project Overview

This project aims to predict Huntington Disease progression using Machine Learning techniques. Multiple classification algorithms were implemented, evaluated, and compared to identify the best-performing model. This project will predict the category of disease in three stages like Primary Cause or Early Stage,Trans-acting Modifier or Middle Stage,Cis-acting Modifier or Advanced Stage.

The project follows an end-to-end Machine Learning and MLOps workflow including:

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Building
* Hyperparameter Tuning
* Model Evaluation
* Model Serialization
* Streamlit Deployment
* MLflow Experiment Tracking
* GitHub Version Control

---

## Dataset

The dataset contains patient-related clinical and demographic information used to predict disease status. The target variable is a Multi-Class Classification Problem means the target variable has three target values.

### Features

* Age
* Gender
* Clinical Measurements
* Cognitive Scores
* Motor Scores
* Other medical indicators

### Target Variable

* Disease Category / Disease Progression Stage

---

## Machine Learning Models

The following models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes

---

## Hyperparameter Tuning

Hyperparameter optimization was performed using:

* GridSearchCV
* RandomizedSearchCV

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Cross Validation Score
* Confusion Matrix

---

## Project Structure

project/

├── data/

├── notebooks/

├── src/

│ ├── preprocessing.py

│ ├── train.py

│ └── predict.py

├── models/

│ └── model.pkl

├── app/

│ └── app.py

├── logs/

├── requirements.txt

├── README.md

---

## MLOps Components

### Version Control

GitHub is used for source code management and version control.

### Experiment Tracking

MLflow is used to track:

* Parameters
* Metrics
* Model Versions

### Logging & Monitoring

Prediction logs and model performance monitoring are maintained for reproducibility and model governance.

---

## Model Deployment

The trained model is deployed using Streamlit.

Run the application:

streamlit run app.py

---

## Installation

Clone the repository:

git clone <repository-url>

Navigate to project directory:

cd project

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py

---

## Results

The models were compared using classification metrics and cross-validation. Logistic Regression was selected as the baseline model due to its balanced performance and good generalization capability.

Machine Learning | MLOps
