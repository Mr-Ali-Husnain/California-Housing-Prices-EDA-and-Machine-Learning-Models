from src.predictor import HousePricePredictor

predictor = HousePricePredictor()
sample = [ -122.23, 37.88, 41.0, 880.0, 129.0, 322.0, 126.0, 8.3252, 1 ]  # example row
result = predictor.predict(sample)
print("Predicted House Price:", result)
