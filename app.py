from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import PredictionPipeline

app = FastAPI(
    title="Manufacturing Parts Prediction API",
    version="1.0.0",
    description="Predict Parts Per Hour using Linear Regression"
)


class PredictionRequest(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float


@app.get("/")
def home():
    return {
        "message": "Manufacturing Parts Prediction API is Running"
    }


@app.post("/predict")
def predict(data: PredictionRequest):
    pipeline = PredictionPipeline()

    prediction = pipeline.predict(data.model_dump())

    return {
        "Predicted_Parts_Per_Hour": round(float(prediction), 2)
    }