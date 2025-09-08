from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
from functools import partial
import pandas as pd
import numpy as np
import joblib
import json
import pickle

location_centroids = pd.read_pickle("data/working_data/datalocation_centroids.pkl")

df = pd.read_csv("data/working_data/data_cleaned_20250827.csv")

# Leemos nuestro modelo
rent_model = joblib.load("models/RandomForestRegressor_rent_production_202509011542.joblib")
sale_model = joblib.load("models/RandomForestRegressor_sale_production_202509011610.joblib")


with open("data/size_bins_by_detailedType.json", 'r') as f:
    size_bins_by_detailedType = json.load(f)

# Inicializa el geolocalizador globalmente
geolocator = Nominatim(user_agent="malaga-app-geocoder")
reverse = partial(geolocator.reverse, language="es", addressdetails=True)

with open("data/app_data/encoders.pkl", 'rb') as f:
    encoders = pickle.load(f)


def get_size_options(df):
    df_grouped = df.groupby('detailedType')['size'].agg(['min', 'max']).reset_index()
    dic_min_maz_sizes = df_grouped.set_index('detailedType').T.to_dict()
    return dic_min_maz_sizes

def haversine(lat1, lon1, lat2, lon2):
    """Distancia Haversine en km"""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def get_location_info_from_coords(lat, lon, fallback_distance_km=2.0):
    """
    Devuelve municipio, distrito y barrio a partir de coordenadas (lat, lon)
    """
    try:
        location = geolocator.reverse((lat, lon), language="es", timeout=10)
        address = location.raw.get("address", {})

        municipality = address.get("municipality") or address.get("town") or address.get("city")
        district = (
            address.get("city_district")
            or address.get("suburb")
            or address.get("borough")
        )
        neighborhood = (
            address.get("neighbourhood")
            or address.get("residential")
            or address.get("quarter")
        )

        if municipality and district and neighborhood:
            return municipality, district, neighborhood

    except Exception as e:
        print(f"Nominatim error: {e}")

    # Fallback: buscar la coordenada más cercana en los centroides
    point = (lat, lon)

    def get_distance(row):
        return geodesic(point, (row['latitude'], row['longitude'])).kilometers

    location_centroids['distance'] = location_centroids.apply(get_distance, axis=1)

    closest = location_centroids.sort_values('distance').iloc[0]
    if closest['distance'] <= fallback_distance_km:
        return closest['municipality'], closest['district'], closest['neighborhood']
    else:
        return None, None, None
    
def get_coords_from_location(municipality, district, neighborhood, location_centroids_df):
    match = location_centroids_df[
        (location_centroids_df['municipality'] == municipality) &
        (location_centroids_df['district'] == district) &
        (location_centroids_df['neighborhood'] == neighborhood)
    ]
    
    if not match.empty:
        return round(match['latitude'].iloc[0], 5), round(match['longitude'].iloc[0], 5)
    else:
        return None, None
    
def get_floor_options(df):
    df_grouped = df.groupby('propertyType')['floor'].agg(['min', 'max']).reset_index()
    dic_min_max_floor = {
        row['propertyType']: {"min": int(row['min']), "max": int(row['max'])}
        for _, row in df_grouped.iterrows()
    }
    return dic_min_max_floor
    
def assign_size_range(row):
    detailed_type = row['detailedType']
    size = row['size']
    bins = size_bins_by_detailedType.get(detailed_type)

    if bins is None:
        return 'Unknown'

    bins = sorted(set(bins))
    
    if bins[-1] != float('inf'):
        bins = bins + [float('inf')]

    labels = [
        f"{int(bins[i])}-{int(bins[i+1])}" if bins[i+1] != float('inf')
        else f"{int(bins[i])}+"
        for i in range(len(bins)-1)
    ]

    cat = pd.cut([size], bins=bins, labels=labels, include_lowest=True)

    return cat[0] if not pd.isna(cat[0]) else 'Unknown'
    
def estimate_price_by_area(df, operation, status, property_type, size, municipality, district, neighborhood):
    df['size_range'] = df.apply(assign_size_range, axis=1)
    bins = size_bins_by_detailedType.get(property_type)
    if bins is None:
        return None

    labels = [f"{int(bins[i])}-{int(bins[i+1])}" if bins[i+1] != float('inf') else f"{int(bins[i])}+" for i in range(len(bins)-1)]
    size_range = pd.cut([size], bins=bins, labels=labels)[0]

    filtered_df = df[
        (df['operation'] == operation) &
        (df['status'] == status) &
        (df['propertyType'] == property_type) &
        (df['size_range'] == size_range) &
        (df['municipality'] == municipality) &
        (df['district'] == district) &
        (df['neighborhood'] == neighborhood)
    ]

    return filtered_df['priceByArea'].mean()
    
def get_price_prediction(propertyType, operation, size, rooms, bathrroms, municipality, district, neighborhood, latitude, longitude, status, detailedType, lift, floor, parkingSpace):
    """
    Makes the prediction using the appropiate model
    """

    new_Development = False
    isParkingSpaceIncludedInPrice = False

    if status == "newdevelopment":
        new_Development = True
    if parkingSpace == True:          
        isParkingSpaceIncludedInPrice = True

    priceByArea = estimate_price_by_area(df, operation, status, propertyType, size, municipality, district, neighborhood)

    data = {'propertyType': propertyType, 'size': size, 'rooms':rooms, 'bathrooms':bathrroms, 'municipality':municipality, 'district':district, 'neighborhood': neighborhood,
        'latitude':latitude, 'longitude':longitude, 'status': status, 'newDevelopment':new_Development,'priceByArea':priceByArea, 'detailedType':detailedType, 'floor':floor, 'hasLift':lift, 'hasParkingSpace':parkingSpace, 'isParkingSpaceIncludedInPrice':isParkingSpaceIncludedInPrice}
    
    input_df = pd.DataFrame([data])
    print("Columns in input_df:", input_df.columns.tolist())
    print("Expecting columns:", list(encoders.keys()))


    if operation == 'rent':
        model = rent_model
    else:
        model = sale_model

    print("model input: ",data)

    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    prediction = model.predict(input_df)[0]
    print(f"Price prediction: {prediction}")
    return prediction
    


