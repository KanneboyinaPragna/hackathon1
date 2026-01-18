from fastapi import FastAPI
from app.schema import DeliveryInput
from app.model_loader import model
import numpy as np

app = FastAPI(title="Delivery Time Prediction API")

@app.get("/")
def home():
    return {"message": "Delivery Time Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict-delivery-time")
def predict_delivery_time(data: DeliveryInput):

    features = np.array([[
        data.road_traffic_density,
        data.delivery_person_rating,
        data.vehicle_condition,
        data.type_of_order,
        data.type_of_vehicle
    ]])

    prediction = model.predict(features)[0]

    # Extra logic (this makes your project different)
    confidence = "High" if prediction < 40 else "Medium"
    sla_status = "On Track" if prediction <= 45 else "At Risk"

    return {
        "estimated_delivery_time_minutes": round(float(prediction), 2),
        "confidence_level": confidence,
        "sla_status": sla_status
    }
