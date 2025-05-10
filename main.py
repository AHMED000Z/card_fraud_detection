from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import load
import numpy as np

app = FastAPI()

# Mount frontend directory
app.mount("/static", StaticFiles(directory="Front"), name="static")

# Load model and scaler
model = load("rf_model.joblib")
scaler = load("scaler.joblib")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("Front/index.html", "r") as f:
        return f.read()

# API schema
class InputData(BaseModel):
    features: list[float]

# Predict route
@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)
    return {"prediction": int(prediction[0])}
