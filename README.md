# California Housing Prices – EDA and Machine Learning Models

## Table of Contents
- [Project Overview](#project-overview)
- [How to Run the Code](#how-to-run-the-code)
- [Project Components](#project-components)
  - [Data Handling](#data-handling)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Classification Extension](#classification-extension)
  - [Error Handling](#error-handling)
- [Steps Performed in the Jupyter Notebook](#steps-performed-in-the-jupyter-notebook)
  - [Step 1: Data Loading and Inspection](#step-1-data-loading-and-inspection)
  - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
  - [Step 3: Feature Engineering](#step-3-feature-engineering)
  - [Step 4: Model Training and Evaluation](#step-4-model-training-and-evaluation)
  - [Step 5: Visualization](#step-5-visualization)
  - [Step 6: Model Saving](#step-6-model-saving)
- [Conclusion](#conclusion)

---

## Project Overview

The **California Housing Prices Application** is a data science project designed to analyze and predict median house values in California using historical housing data.  
This project performs detailed **Exploratory Data Analysis (EDA)** and builds predictive **Machine Learning models** using Linear Regression and Random Forests.  
It also explores a classification approach by categorizing housing prices into multiple classes (Low, Medium, High).

The entire workflow — from data preprocessing to model saving — is implemented in **Python**, and all experiments are performed in **Jupyter Notebook** and **VS Code**.

---

## How to Run the Code

Follow these steps to set up and execute the project on your system:

### 1️⃣ Download the Project Files
Ensure that you have the following files:
- `main.py`
- `housing.csv` (dataset)
- `requirements.txt`
- `project.ipynb` (for Jupyter-based workflow)

The following folders will be automatically created or used:
Model/ → to store trained models
data/ → contains dataset (housing.csv)
images/ → stores visualizations and plots
src/ → contains source code modules

yaml
Copy code

---

### 2️⃣ Install Required Libraries
Make sure Python is installed on your system (>=3.8).  
Then open the terminal in your project directory and run:

```bash
pip install -r requirements.txt
3️⃣ Run the Application Script
Run the full EDA and model training pipeline:

bash
Copy code
python main.py
OR run interactively in Jupyter Notebook:

bash
Copy code
jupyter notebook project.ipynb
 Project Components
 Data Handling
Uses a DataHandler/EDA class to:

Load the dataset (housing.csv)

Display missing values, data types, and descriptive statistics

Handle missing values and perform encoding

Generate new engineered features like:

rooms_per_household

bedrooms_per_room

population_per_household

 Exploratory Data Analysis (EDA)
Includes:

Histograms of numerical features

Correlation heatmap

Geographical scatter plots (longitude vs latitude)

Boxplots for house values by ocean proximity

Outlier analysis and data distribution plots

 Model Training and Evaluation
Two regression models are trained and compared:

Linear Regression – baseline model

Random Forest Regressor – tuned for higher accuracy

Each model is evaluated using:

RMSE (Root Mean Squared Error)

R² Score

Visualization of Actual vs Predicted Values

Residual plots and Feature Importances

 Classification Extension
The continuous median_house_value variable is binned into classes (Low, Medium, High)
A RandomForestClassifier is then trained to predict price category, with:

Classification Report (Precision, Recall, F1-score)

Confusion Matrix Visualization

 Error Handling
Checks if dataset file (housing.csv) exists

Handles missing or invalid values gracefully

Ensures positive numeric inputs during preprocessing

Auto-handles infinity and NaN from feature engineering

 Steps Performed in the Jupyter Notebook
Step 1: Data Loading and Inspection
Load data using the EDA class.

Display head, info, and descriptive statistics.

Step 2: Data Preprocessing
Handle missing values, encode categorical features, and impute numeric columns.

Prepare data for model input.

Step 3: Feature Engineering
Add new attributes for better correlation and prediction accuracy.

Step 4: Model Training and Evaluation
Train Linear Regression and Random Forest Regressor models.

Evaluate using RMSE, R², and visualize results.

Step 5: Visualization
Generate histograms, heatmaps, scatter plots, and feature importance charts.

Step 6: Model Saving
Save trained models and scaler in the Model/ folder using joblib.

 Conclusion
The California Housing Prices EDA and Machine Learning Models project demonstrates a complete Data Science workflow:

Data Cleaning & Exploration

Feature Engineering

Predictive Modeling & Evaluation

Model Saving for Reuse

This project provides a strong foundation for advanced predictive analytics and deployment-ready housing price estimation systems.

 Author
Rana Ali Husnain
 BSCS 5th Semester | Data Science & Machine Learning Enthusiast
