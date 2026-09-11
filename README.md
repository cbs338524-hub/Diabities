Diabetes Prediction System

A professional Machine Learning web application for predicting diabetes
risk from patient health parameters using Logistic Regression and Flask.

Disclaimer: This project is for educational and demonstration
purposes only. It is not a medical diagnostic tool and should not
replace advice from a qualified healthcare professional.

Project Overview

This project demonstrates a complete supervised machine learning
workflow:

Data cleaning and preprocessing

Feature selection

Target encoding

Train/test splitting

Feature standardization

Logistic Regression classification

Probability-based prediction

Model evaluation

Flask web application integration

Features

Machine Learning

Logistic Regression

StandardScaler preprocessing

80/20 stratified train-test split

Automatic data cleaning

Missing-value handling

Diabetes Yes/No target encoding

Model Evaluation

The application calculates:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Patient Prediction

Users can enter:

Patient Name

Age

Blood Pressure

Blood Sugar

BMI

The application returns:

Prediction result

Diabetes probability

Risk level

Prediction timestamp

Risk Classification

Probability        Risk Level

Below 30%          Low Risk
30% to below 70%   Moderate Risk
70% or above       High Risk

These thresholds are application-defined demonstration thresholds and
are not clinical diagnostic thresholds.

Dataset

The project uses hospital.csv.

Relevant columns:

Column           Description

Patient_Name     Patient name/identifier
Age              Patient age
Blood_Pressure   Blood pressure
Blood_Sugar      Blood sugar
BMI              Body Mass Index
Diabetes         Target variable: Yes/No

The model uses:

Age
Blood_Pressure
Blood_Sugar
BMI

Target:

Diabetes

Machine Learning Workflow

hospital.csv
    ↓
Data Cleaning
    ↓
Feature Selection
    ↓
Target Encoding
    ↓
Train/Test Split
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
Model Evaluation
    ↓
Flask Web Application
    ↓
Patient Prediction

Technologies Used

Python

Pandas

Scikit-learn

Flask

HTML

CSS

Jinja2

Project Structure

The current Flask setup keeps index.html in the same folder as
app.py.

training class/
│
├── app.py
├── hospital.csv
├── index.html
└── README.md

Installation

Install the required Python packages:

pip3 install pandas flask scikit-learn

Or:

python3 -m pip install pandas flask scikit-learn

Run the Application

Open the terminal in the project folder:

cd "training class"

Run:

python3 app.py

Then open:

http://127.0.0.1:5000

Important: Do not open index.html directly with file://. It must
be rendered through Flask because it contains Jinja template variables.

Model Evaluation

Accuracy measures the proportion of correct predictions.

Precision measures how many predicted positive cases were actually
positive.

Recall measures how many actual positive cases were correctly
detected.

F1 Score balances precision and recall.

ROC-AUC measures the model's ability to distinguish between the two
classes.

Confusion Matrix provides True Negative, False Positive, False
Negative, and True Positive counts.

Feature Interpretation

The Logistic Regression coefficients provide an interpretable indication
of how the standardized features influence the model's prediction.

A positive coefficient increases the estimated log-odds of the positive
class, while a negative coefficient decreases them, with other features
held constant.

Input Validation

The application validates:

Age            : 1 - 120
Blood Pressure : 40 - 250
Blood Sugar    : 20 - 600
BMI            : 10 - 80

These are application input-validation ranges, not medical reference
ranges.

Future Improvements

Random Forest and XGBoost comparison

Cross-validation

Hyperparameter tuning

ROC curve visualization

Precision-Recall curve

Feature importance charts

Prediction history

Downloadable reports

Database integration

REST API

Cloud deployment

Responsive mobile interface

Learning Outcomes

This project demonstrates practical understanding of:

Supervised Machine Learning

Binary Classification

Logistic Regression

Feature Scaling

Train/Test Splitting

Classification Metrics

Confusion Matrix

ROC-AUC

Flask

HTML/Jinja integration

Machine Learning deployment concepts

Author

AIML Student

Developed as a practical Machine Learning and Flask web application
project.

License

This project is intended for educational and portfolio purposes.

Medical Disclaimer

The application provides a machine-learning-based risk estimate from the
supplied dataset. It is not a medical diagnosis. Health-related
decisions should be made with guidance from a qualified healthcare
professional.
