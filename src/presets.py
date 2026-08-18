"""
Preset Delivery Scenarios & Test Data Sampler
Adaptive Food Delivery ETA Prediction System
"""

import pandas as pd
import random

PRESET_SCENARIOS = {
    "⚡ Express Lunch Delivery (Ideal Conditions)": {
        "description": "Short distance delivery during daytime with clear sunny skies, low traffic density, and a highly rated rider on a scooter.",
        "restaurant_lat": 12.9716,
        "restaurant_lon": 77.5946,
        "delivery_lat": 12.9850,
        "delivery_lon": 77.6050,
        "delivery_person_age": 30,
        "delivery_person_ratings": 4.9,
        "weather_condition": "Sunny",
        "traffic_density": "Low",
        "vehicle_type": "scooter",
        "vehicle_condition": 3,
        "order_type": "Meal",
        "multiple_deliveries": 0,
        "festival": "No",
        "city": "Urban",
        "order_hour": 13,
        "pickup_hour": 13,
        "weekend": 0,
        "month": 3,
    },
    "🌧️ Severe Monsoon Storm & Peak Rush (High Risk)": {
        "description": "Long-distance dinner delivery under torrential storm conditions, severe road traffic jams, high kitchen backlog, and overloaded rider.",
        "restaurant_lat": 12.9352,
        "restaurant_lon": 77.6245,
        "delivery_lat": 13.0827,
        "delivery_lon": 77.5877,
        "delivery_person_age": 22,
        "delivery_person_ratings": 3.8,
        "weather_condition": "Stormy",
        "traffic_density": "Jam",
        "vehicle_type": "bicycle",
        "vehicle_condition": 1,
        "order_type": "Buffet",
        "multiple_deliveries": 3,
        "festival": "Yes",
        "city": "Metropolitian",
        "order_hour": 20,
        "pickup_hour": 21,
        "weekend": 1,
        "month": 7,
    },
    "🌙 Late Night Fast Run (Normal Conditions)": {
        "description": "Midnight order with minimal road traffic, clear conditions, and single delivery assignment on an electric scooter.",
        "restaurant_lat": 12.9279,
        "restaurant_lon": 77.6271,
        "delivery_lat": 12.9550,
        "delivery_lon": 77.6500,
        "delivery_person_age": 27,
        "delivery_person_ratings": 4.6,
        "weather_condition": "Cloudy",
        "traffic_density": "Low",
        "vehicle_type": "electric_scooter",
        "vehicle_condition": 2,
        "order_type": "Snack",
        "multiple_deliveries": 0,
        "festival": "No",
        "city": "Urban",
        "order_hour": 23,
        "pickup_hour": 23,
        "weekend": 0,
        "month": 4,
    },
    "🎪 Festival Evening Rush (Medium-High Load)": {
        "description": "Major festival celebration evening with elevated order volumes, buffet catering order, and medium congestion.",
        "restaurant_lat": 13.0358,
        "restaurant_lon": 77.5970,
        "delivery_lat": 12.9600,
        "delivery_lon": 77.6400,
        "delivery_person_age": 34,
        "delivery_person_ratings": 4.7,
        "weather_condition": "Windy",
        "traffic_density": "High",
        "vehicle_type": "motorcycle",
        "vehicle_condition": 2,
        "order_type": "Buffet",
        "multiple_deliveries": 2,
        "festival": "Yes",
        "city": "Metropolitian",
        "order_hour": 19,
        "pickup_hour": 20,
        "weekend": 1,
        "month": 10,
    },
    "🛵 Suburban Drinks Run (Medium Distance)": {
        "description": "Afternoon beverage dropoff in semi-urban neighborhood with moderate travel and standard rider profile.",
        "restaurant_lat": 12.9100,
        "restaurant_lon": 77.6000,
        "delivery_lat": 12.8700,
        "delivery_lon": 77.5900,
        "delivery_person_age": 29,
        "delivery_person_ratings": 4.5,
        "weather_condition": "Fog",
        "traffic_density": "Medium",
        "vehicle_type": "motorcycle",
        "vehicle_condition": 2,
        "order_type": "Drinks",
        "multiple_deliveries": 1,
        "festival": "No",
        "city": "Semi-Urban",
        "order_hour": 16,
        "pickup_hour": 16,
        "weekend": 0,
        "month": 2,
    }
}


_CACHED_DF = None

def _get_dataset_df(dataset_path: str = "dataset/processed/riders_features.csv") -> pd.DataFrame:
    global _CACHED_DF
    if _CACHED_DF is None:
        _CACHED_DF = pd.read_csv(dataset_path)
    return _CACHED_DF


def load_random_sample_from_dataset(dataset_path: str = "dataset/processed/riders_features.csv") -> tuple[dict, float]:
    """
    Samples a random historical delivery record from riders_features.csv
    Returns: (input_dict, actual_time_taken)
    """
    df = _get_dataset_df(dataset_path)
    sample = df.sample(n=1, random_state=random.randint(0, 100000)).iloc[0]
    
    # Reverse mapping for categorical values if needed
    weather_cond = sample.get("Weather_conditions", "Sunny")
    if pd.isna(weather_cond) or "conditions " in str(weather_cond):
        weather_cond = str(weather_cond).replace("conditions ", "").strip()
    
    traffic_dense = sample.get("Road_traffic_density", "Medium")
    if pd.isna(traffic_dense) or str(traffic_dense).strip() == "":
        traffic_dense = "Medium"
        
    vehicle_t = sample.get("Type_of_vehicle", "motorcycle")
    if pd.isna(vehicle_t):
        vehicle_t = "motorcycle"
        
    order_t = sample.get("Type_of_order", "Meal")
    if pd.isna(order_t):
        order_t = "Meal"
        
    city_val = sample.get("City", "Metropolitian")
    if pd.isna(city_val):
        city_val = "Metropolitian"
        
    fest_val = sample.get("Festival", "No")
    if pd.isna(fest_val):
        fest_val = "No"

    input_dict = {
        "description": f"Historical delivery record ID: {sample.get('ID', 'N/A')} (Delivery Person: {sample.get('Delivery_person_ID', 'N/A')})",
        "restaurant_lat": float(abs(sample.get("Restaurant_latitude", 12.9716))),
        "restaurant_lon": float(abs(sample.get("Restaurant_longitude", 77.5946))),
        "delivery_lat": float(abs(sample.get("Delivery_location_latitude", 12.9352))),
        "delivery_lon": float(abs(sample.get("Delivery_location_longitude", 77.6245))),
        "delivery_person_age": int(sample.get("Delivery_person_Age", 28)),
        "delivery_person_ratings": float(sample.get("Delivery_person_Ratings", 4.7)),
        "weather_condition": str(weather_cond).strip(),
        "traffic_density": str(traffic_dense).strip(),
        "vehicle_type": str(vehicle_t).strip(),
        "vehicle_condition": int(sample.get("Vehicle_condition", 2)),
        "order_type": str(order_t).strip(),
        "multiple_deliveries": int(sample.get("multiple_deliveries", 1)),
        "festival": str(fest_val).strip(),
        "city": str(city_val).strip(),
        "order_hour": int(sample.get("Order_Hour", 14)),
        "pickup_hour": int(sample.get("Pickup_Hour", 14)),
        "weekend": int(sample.get("Weekend", 0)),
        "month": int(sample.get("Month", 3)),
    }
    
    actual_time = float(sample.get("Time_taken (min)", 0))
    return input_dict, actual_time
