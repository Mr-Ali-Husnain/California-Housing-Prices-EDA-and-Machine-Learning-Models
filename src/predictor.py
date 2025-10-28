import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import os

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model_path = "Model/"
        os.makedirs(self.model_path, exist_ok=True)

    def prepare_data(self):
        df = self.df.drop("ocean_proximity", axis=1)
        X = df.drop("median_house_value", axis=1)
        y = df["median_house_value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_regression_models(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train_scaled, y_train)
        y_pred_lr = lin_reg.predict(X_test_scaled)

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        y_pred_rf = rf_reg.predict(X_test)

        print("\n📈 Linear Regression:")
        print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
        print("R²:", r2_score(y_test, y_pred_lr))

        print("\n🌲 Random Forest Regression:")
        print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
        print("R²:", r2_score(y_test, y_pred_rf))

        joblib.dump(lin_reg, os.path.join(self.model_path, "lin_reg.joblib"))
        joblib.dump(rf_reg, os.path.join(self.model_path, "rf_regressor.joblib"))
        joblib.dump(scaler, os.path.join(self.model_path, "scaler.joblib"))
        print("\n✅ Models saved in 'Model/' folder.")

    def classify_house_value(self):
        df = self.df.copy()
        df["value_category"] = pd.cut(df["median_house_value"],
                                      bins=[0, 150000, 300000, 500000],
                                      labels=["Low", "Medium", "High"])
        X = df.drop(["median_house_value", "ocean_proximity", "value_category"], axis=1)
        y = df["value_category"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        print("\n🎯 Classification Report:")
        print(classification_report(y_test, y_pred))
        joblib.dump(rf_clf, os.path.join(self.model_path, "rf_classifier.joblib"))
