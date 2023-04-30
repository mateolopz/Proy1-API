from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from transformer_custom import DataPreprocessing

from DataModel import DataModel

app = FastAPI()
vectorizador1 = joblib.load("vectorizador.joblib")

@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = joblib.load("modelo.joblib")
    model.set_params(use_custom_transformer__vectorizador=vectorizador1)
    resp = pd.DataFrame()
    result = model.predict(df)
    resp["respuesta"]=pd.Series(result)
    return resp

