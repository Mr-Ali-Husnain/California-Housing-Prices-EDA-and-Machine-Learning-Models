from src.data_handler import HousingEDA
from src.predictor import ModelTrainer

def main():
    filepath = "data/housing.csv"

    eda = HousingEDA(filepath)
    df = eda.load_data()
    df = eda.clean_data()
    eda.plot_distributions()
    eda.correlation_heatmap()
    eda.geographical_plot()

    trainer = ModelTrainer(df)
    trainer.train_regression_models()
    trainer.classify_house_value()

if __name__ == "__main__":
    main()
