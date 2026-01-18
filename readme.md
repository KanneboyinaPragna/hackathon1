# Delivery Duration Prediction using Machine Learning
## Problem Statement

Accurately estimating delivery duration is challenging due to varying traffic conditions, courier performance, and package characteristics.
Inaccurate ETAs lead to poor customer experience and SLA violations.

This project aims to predict delivery duration using machine learning based on route traffic, courier characteristics, and order/package details.
## Solution Overview

We developed a machine learning-based system that predicts delivery duration using historical delivery data.
The trained model is exposed through REST APIs using FastAPI, enabling real-time delivery time predictions.
## Dataset

Dataset used: Zomato Delivery Operations Analytics Dataset (Kaggle)

The dataset includes:
- Road traffic density
- Courier ratings and vehicle condition
- Order and package type
- Actual delivery time taken (in minutes)

This dataset closely represents real-world delivery and logistics operations.
## Features and Target

### Input Features
- Road_traffic_density
- Delivery_person_Ratings
- Vehicle_condition
- Type_of_order
- Type_of_vehicle

### Target Variable
- Delivery_time_taken (minutes)
## Tech Stack

- Python
- FastAPI (Backend & APIs)
- scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Swagger UI (API Testing)
- GitHub (Version Control)
## Project Architecture

1. Dataset is collected and preprocessed
2. Machine learning regression model is trained
3. Trained model is saved using joblib
4. FastAPI loads the trained model
5. API receives delivery-related inputs
6. API returns estimated delivery duration
## API Details

### Endpoint
POST /predict-delivery-time

### Sample Request
```json
{
  "road_traffic_density": 2,
  "delivery_person_rating": 4.3,
  "vehicle_condition": 3,
  "type_of_order": 1,
  "type_of_vehicle": 2
}
{
  "estimated_delivery_time_minutes": 36.5
}

---

## 🔹 STEP 11: Add How to Run the Project

Add:

```md
## How to Run the Project

1. Clone the GitHub repository
2. Create and activate a Python virtual environment
3. Install dependencies:
   pip install -r requirements.txt
4. Run the FastAPI server:
   uvicorn app.main:app --reload
5. Open Swagger UI:
   //http://127.0.0.1:8000/docs
## Team Members

- Member 1: Machine Learning & Model Development
- Member 2: Backend & API Development
- Member 3: Dataset Analysis, Documentation & Demo
## Conclusion

This project demonstrates an end-to-end machine learning solution for delivery duration prediction.
By combining traffic, courier, and order characteristics, the system provides accurate and realistic delivery time estimates.
