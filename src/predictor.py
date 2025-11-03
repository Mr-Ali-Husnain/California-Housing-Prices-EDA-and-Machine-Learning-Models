import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np


# ======================== #
#   Data Preprocessing
# ======================== #
class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def encode_and_split(self, target):
        print("\nEncoding categorical features...")
        le = LabelEncoder()
        self.df['ocean_proximity'] = le.fit_transform(self.df['ocean_proximity'])

        X = self.df.drop(columns=[target])
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data split into train and test sets.")
        return X_train, X_test, y_train, y_test

# ======================== #
#   Model Selection
# ======================== #  
class ModelSelector:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def compare_models(self):
        print("\n🔍 Comparing Selected Regressor Models...\n")

        # ✅ Define a smaller set of good models
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "Support Vector Regressor": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor()
        }

        results = []

        # ✅ Train & evaluate each model
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            score = model.score(self.X_test, self.y_test)
            cv_score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=3))
            results.append({
                "Model": name,
                "R² Test Score": round(score, 3),
                "Cross-Val Score": round(cv_score, 3)
            })

        # ✅ Display results in clean table
        results_df = pd.DataFrame(results).sort_values(by="R² Test Score", ascending=False)
        print(results_df)
        print("\n✅ Model comparison complete — fewer models, faster execution!")
        return results_df

# ======================== #
#   Model Training
# ======================== #
class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_best_model(self):
        print("\nTraining RandomForestRegressor as best model...")
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print("Model trained successfully.")

        import pickle
        with open("Model/rf_regressor.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Trained model saved successfully in 'Model/rf_regressor.pkl'")

        return model, y_pred


# ======================== #
#   Model Evaluation
# ======================== #
class ModelEvaluator:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def evaluate(self):
        print("\nModel Performance:")
        print(f"MAE: {mean_absolute_error(self.y_test, self.y_pred):.2f}")
        print(f"MSE: {mean_squared_error(self.y_test, self.y_pred):.2f}")
        print(f"R² Score: {r2_score(self.y_test, self.y_pred):.2f}")

    def plot_confusion_matrix(self):
        y_true = np.round(self.y_test / 50000).astype(int)
        y_pred = np.round(self.y_pred / 50000).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Binned Target)")
        plt.show()

