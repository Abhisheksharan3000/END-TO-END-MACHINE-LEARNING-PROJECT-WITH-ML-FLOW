import joblib
import numpy as np
import pandas as pd
from pathlib import path

class PredictionPipeline:
    
    def __init__(self):
        self.model = joblib.load(path('artifacts/model_trainer/model.joblib'))
        
    def predict(self,data):
        prediciton = self.model.predict(data)
        
        return prediciton
    