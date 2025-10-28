import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class HousingEDA:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        print("📂 Loading dataset...")
        self.df = pd.read_csv(self.filepath)
        print(self.df.head())
        print("\n✅ Data Loaded Successfully.")
        return self.df

    def clean_data(self):
        print("\n🧹 Cleaning Data...")
        self.df.dropna(inplace=True)
        self.df["rooms_per_household"] = self.df["total_rooms"] / self.df["households"]
        self.df["bedrooms_per_room"] = self.df["total_bedrooms"] / self.df["total_rooms"]
        self.df["population_per_household"] = self.df["population"] / self.df["households"]
        print("✅ Data cleaned and new features added.")
        return self.df

    def plot_distributions(self):
        print("\n📊 Plotting Histograms...")
        self.df.hist(figsize=(10, 8))
        plt.tight_layout()
        plt.savefig("images/histograms.png")
        plt.show()

    def correlation_heatmap(self):
        print("\n🔥 Correlation Heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.savefig("images/corr_heatmap.png")
        plt.show()

    def geographical_plot(self):
        print("\n🌍 Geo Plot...")
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df["longitude"], self.df["latitude"],
                    alpha=0.4, c=self.df["median_house_value"], cmap="viridis")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Geographical Distribution of House Prices")
        plt.colorbar(label="Median House Value")
        plt.savefig("images/geo_scatter.png")
        plt.show()
