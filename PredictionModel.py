import joblib
from transformer_custom import DataPreprocessing

class Model:

    def __init__(self,columns):
        self.model = joblib.load("modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
