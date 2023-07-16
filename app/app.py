import auth
import pandas as pd
from utils import get_logger, get_model_from_gcs
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKey
from pydantic import BaseModel

# Create a logger
logger = get_logger()

# Get the model
model = get_model_from_gcs()


# Define API models
class Property(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: float
    n_bathroom: float
    latitude: float
    longitude: float


class Valuation(BaseModel):
    predicted_price: int


# Create FastAPI app
app = FastAPI()


# Home page
@app.get("/")
async def home():
    return {"message": "Property Valuation API", "model_version": 0.1}


# Define API routes
@app.post("/predict", response_model=Valuation)
async def predict_property_valuation(
    payload: Property, api_key: APIKey = Depends(auth.get_api_key)
):
    try:
        logger.info(f"Received a request - {payload.dict()}")
        request_df = pd.DataFrame([payload.dict()])
        prediction = round(model.predict(request_df)[0])
        logger.info(f"Prediction result - {prediction}")
        result = {"predicted_price": prediction}
    except Exception as e:
        logger.error(f"Prediction failed - Request: {payload.dict()}, Error: {str(e)}")
        raise HTTPException(status_code=400, detail="Prediction failed.")
    return result
