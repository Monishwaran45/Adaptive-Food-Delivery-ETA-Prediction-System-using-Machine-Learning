"""
Feature Engineering and Stage-Specific Fusion Module
Adaptive Food Delivery ETA Prediction System
"""

import math
import numpy as np
import pandas as pd

PEAK_MAP = {"Normal": 0, "Breakfast": 1, "Lunch": 2, "Dinner": 3}
FESTIVAL_MAP = {"No": 0, "Yes": 1}
CITY_MAP = {"Metropolitian": 0, "Semi-Urban": 1, "Urban": 2}
ORDER_MAP = {"Buffet": 0, "Drinks": 1, "Meal": 2, "Snack": 3}
TRAFFIC_MAP = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
WEATHER_MAP = {"Sunny": 1, "Cloudy": 2, "Windy": 3, "Fog": 4, "Stormy": 5, "Sandstorms": 6}
VEHICLE_MAP = {"bicycle": 1, "electric_scooter": 2, "scooter": 3, "motorcycle": 4}

EXPECTED_35_FEATURES = [
    'Traffic_Score', 'Workload', 'Multiple_Deliveries', 'Peak', 'Festival',
    'Rider_Experience', 'Ratings', 'Traffic_Workload', 'Demand_Index', 'Rider_Load',
    'Trip_Distance', 'Traffic', 'Vehicle', 'Vehicle_Condition', 'Restaurant_Lat',
    'Restaurant_Lon', 'Travel_Index', 'Vehicle_Index', 'Efficiency', 'Weather',
    'City', 'Order', 'Restaurant_Demand', 'Weather_Delay', 'Lat', 'Lon',
    'Experience', 'Delivery_Index', 'Weather_Impact', 'Experience_Index',
    'Order_Hour', 'Pickup_Hour', 'Weekend', 'Month', 'Delivery_person_Age'
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two latitude/longitude points in km."""
    lat1, lon1 = float(lat1), float(lon1)
    lat2, lon2 = float(lat2), float(lon2)

    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 4)


def compute_peak_period(hour: int) -> str:
    if 7 <= hour <= 10:
        return "Breakfast"
    elif 12 <= hour <= 15:
        return "Lunch"
    elif 18 <= hour <= 22:
        return "Dinner"
    return "Normal"


def build_raw_delivery_dict(
    restaurant_lat: float, restaurant_lon: float, delivery_lat: float, delivery_lon: float,
    delivery_person_age: int, delivery_person_ratings: float, weather_condition: str,
    traffic_density: str, vehicle_type: str, vehicle_condition: int, order_type: str,
    multiple_deliveries: int, festival: str, city: str, order_hour: int, pickup_hour: int,
    weekend: int = 0, month: int = 3, trip_distance_km: float = None,
) -> dict:
    if trip_distance_km is None:
        trip_distance_km = haversine_distance(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)

    peak_period_name = compute_peak_period(order_hour)
    traffic_score = TRAFFIC_MAP.get(traffic_density, 2)
    weather_score = WEATHER_MAP.get(weather_condition, 1)
    vehicle_score = VEHICLE_MAP.get(vehicle_type.lower().replace(" ", "_"), 3)
    peak_code = PEAK_MAP.get(peak_period_name, 0)
    festival_code = FESTIVAL_MAP.get(festival, 0)
    city_code = CITY_MAP.get(city, 0)
    order_code = ORDER_MAP.get(order_type, 2)
    rider_experience = float(delivery_person_age) * float(delivery_person_ratings)
    workload = float(multiple_deliveries)

    return {
        "restaurant_lat": float(restaurant_lat), "restaurant_lon": float(restaurant_lon),
        "delivery_lat": float(delivery_lat), "delivery_lon": float(delivery_lon),
        "trip_distance_km": float(trip_distance_km), "delivery_person_age": int(delivery_person_age),
        "delivery_person_ratings": float(delivery_person_ratings), "traffic_score": traffic_score,
        "weather_score": weather_score, "vehicle_score": vehicle_score,
        "vehicle_condition": int(vehicle_condition), "multiple_deliveries": int(multiple_deliveries),
        "workload": workload, "rider_experience": rider_experience, "peak_code": peak_code,
        "festival_code": festival_code, "city_code": city_code, "order_code": order_code,
        "order_hour": int(order_hour), "pickup_hour": int(pickup_hour), "weekend": int(weekend),
        "month": int(month), "peak_period_name": peak_period_name,
    }


def generate_adaptive_fusion_features(inputs: dict) -> pd.DataFrame:
    traffic_score = inputs["traffic_score"]
    workload = inputs["workload"]
    mult_deliv = inputs["multiple_deliveries"]
    peak = inputs["peak_code"]
    festival = inputs["festival_code"]
    rider_exp = inputs["rider_experience"]
    ratings = inputs["delivery_person_ratings"]

    traffic_workload = traffic_score * workload
    demand_index = traffic_score * (peak + 1)
    rider_load = rider_exp / (mult_deliv + 1)

    trip_distance = inputs["trip_distance_km"]
    vehicle_score = inputs["vehicle_score"]
    veh_condition = inputs["vehicle_condition"]
    rest_lat = inputs["restaurant_lat"]
    rest_lon = inputs["restaurant_lon"]
    travel_index = trip_distance * traffic_score
    vehicle_index = vehicle_score * veh_condition
    efficiency = ratings * vehicle_score

    weather_score = inputs["weather_score"]
    city_code = inputs["city_code"]
    order_code = inputs["order_code"]
    restaurant_demand = peak + festival
    weather_delay = weather_score * (peak + 1)

    deliv_lat = inputs["delivery_lat"]
    deliv_lon = inputs["delivery_lon"]
    delivery_index = trip_distance * traffic_score
    weather_impact = weather_score * trip_distance
    experience_index = rider_exp / (traffic_score + 1)

    data = {
        "Traffic_Score": [traffic_score], "Workload": [workload], "Multiple_Deliveries": [mult_deliv],
        "Peak": [peak], "Festival": [festival], "Rider_Experience": [rider_exp], "Ratings": [ratings],
        "Traffic_Workload": [traffic_workload], "Demand_Index": [demand_index], "Rider_Load": [rider_load],
        "Trip_Distance": [trip_distance], "Traffic": [traffic_score], "Vehicle": [vehicle_score],
        "Vehicle_Condition": [veh_condition], "Restaurant_Lat": [rest_lat], "Restaurant_Lon": [rest_lon],
        "Travel_Index": [travel_index], "Vehicle_Index": [vehicle_index], "Efficiency": [efficiency],
        "Weather": [weather_score], "City": [city_code], "Order": [order_code],
        "Restaurant_Demand": [restaurant_demand], "Weather_Delay": [weather_delay], "Lat": [deliv_lat],
        "Lon": [deliv_lon], "Experience": [rider_exp], "Delivery_Index": [delivery_index],
        "Weather_Impact": [weather_impact], "Experience_Index": [experience_index],
        "Order_Hour": [inputs["order_hour"]], "Pickup_Hour": [inputs["pickup_hour"]],
        "Weekend": [inputs["weekend"]], "Month": [inputs["month"]],
        "Delivery_person_Age": [inputs["delivery_person_age"]],
    }
    return pd.DataFrame(data)[EXPECTED_35_FEATURES]
