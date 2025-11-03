# src/data_handler.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os

# ======================== #
#    DataLoader
# ======================== #
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("Data Loaded Successfully!\n")
        return self.df

    def show_info(self):
        print("\nDataset Info:")
        return self.df.info()

    def describe_data(self):
        print("\nStatistical Summary:")
        return self.df.describe()

# ======================== #
#    DataCleaner
# ======================== #
class DataCleaner:
    def __init__(self, df):
        self.df = df

    def check_missing(self):
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        return self.df.isnull().sum()

    def fill_missing(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print("Missing values filled.")
        return self.df

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {before - len(self.df)} duplicates.")
        return self.df

    def remove_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1, Q3 = self.df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            self.df = self.df[(self.df[col] >= Q1 - 1.5*IQR) & (self.df[col] <= Q3 + 1.5*IQR)]
        print("Outliers removed.")
        return self.df

# ======================== #
#   Data Analysis
# ======================== #
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def correlation_matrix(self):
        print("\nCorrelation Matrix:")
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        print(corr)
        return corr

    def highest_correlations(self, target, top_n=5):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if target not in numeric_df.columns:
            print(f"'{target}' is not numeric! Convert or encode it before correlation.")
            return None

        corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
        print(f"\nTop {top_n} correlated features with '{target}':")
        print(corr.head(top_n))
        return corr.head(top_n)

# ======================== #
#   Data Visualization (with auto-save)
# ======================== #
class DataVisualizer:
    def __init__(self, df):
        self.df = df
        # Folder to store all generated plots
        self.output_dir = "Visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_plot(self, filename):
        """Helper function to save the current plot."""
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        print(f"Plot saved: {path}")
        plt.close()

    def plot_distributions(self):
        self.df.hist(figsize=(12, 8), bins=20)
        plt.suptitle("Feature Distributions", fontsize=16)
        self._save_plot("feature_distributions.png")

    def correlation_heatmap(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        self._save_plot("correlation_heatmap.png")

    def ocean_proximity_count(self):
        plt.figure(figsize=(7, 5))
        sns.countplot(x='ocean_proximity', data=self.df)
        plt.title("Ocean Proximity Count")
        self._save_plot("ocean_proximity_count.png")

    def income_vs_value(self):
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='median_income', y='median_house_value', data=self.df, alpha=0.5)
        plt.title("Income vs House Value")
        plt.xlabel("Median Income")
        plt.ylabel("Median House Value")
        self._save_plot("income_vs_value.png")

    def boxplot_features(self):
        numeric_cols = self.df.select_dtypes(include='number').columns
        plt.figure(figsize=(15,8))
        self.df[numeric_cols].boxplot()
        plt.title("Boxplot of Numeric Features (Outlier Detection)")
        plt.xticks(rotation=45)
        self._save_plot("boxplot_features.png")

    def geo_distribution(self):
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='longitude', y='latitude', hue='median_house_value', data=self.df, palette='coolwarm', alpha=0.6)
        plt.title("Geographical Distribution of House Prices")
        self._save_plot("geo_distribution.png")

    def avg_price_per_ocean(self):
        avg_prices = self.df.groupby('ocean_proximity')['median_house_value'].mean().sort_values()
        plt.figure(figsize=(7,5))
        sns.barplot(x=avg_prices.index, y=avg_prices.values, palette='viridis')
        plt.title("Average House Value by Ocean Proximity")
        plt.xlabel("Ocean Proximity")
        plt.ylabel("Average House Value")
        self._save_plot("avg_price_per_ocean.png")

