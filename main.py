# ======================== #
#        main.py
# ======================== #

from src.data_handler import (
    DataLoader,
    DataCleaner,
    DataAnalyzer,
    DataVisualizer
)
from src.predictor import (
    DataPreprocessor,
    ModelSelector,
    ModelTrainer,
    ModelEvaluator
)

# ============ EXECUTION PIPELINE ============ #
if __name__ == "__main__":
    # --- 1️ Load Data ---
    file_path = "Data/housing.csv"      # change path if needed
    target_col = "median_house_value"

    loader = DataLoader(file_path)
    df = loader.load_data()
    loader.show_info()
    loader.describe_data()

    # --- 2️ Clean Data ---
    cleaner = DataCleaner(df)
    cleaner.check_missing()
    df = cleaner.fill_missing()
    df = cleaner.remove_duplicates()
    df = cleaner.remove_outliers()

    # --- 3️ Analyze Data ---
    analyzer = DataAnalyzer(df)
    analyzer.correlation_matrix()
    analyzer.highest_correlations(target_col)

    # --- 4️ Visualize Data (auto-saved in Visualizations folder) ---
    viz = DataVisualizer(df)
    viz.plot_distributions()
    viz.correlation_heatmap()
    viz.ocean_proximity_count()
    viz.income_vs_value()
    viz.boxplot_features()
    viz.geo_distribution()
    viz.avg_price_per_ocean()

    # --- 5️ Preprocess & Split ---
    pre = DataPreprocessor(df)
    X_train, X_test, y_train, y_test = pre.encode_and_split(target_col)

    # --- 6️ Model Comparison (LazyPredict) ---
    selector = ModelSelector(X_train, X_test, y_train, y_test)
    selector.compare_models()

    # --- 7️ Train Best Model + Save Pickle ---
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    model, y_pred = trainer.train_best_model()

    # --- 8️ Evaluate Model ---
    evaluator = ModelEvaluator(y_test, y_pred)
    evaluator.evaluate()
    evaluator.plot_confusion_matrix()

    print("\nFull EDA & ML pipeline executed successfully!")
