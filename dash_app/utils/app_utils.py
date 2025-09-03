from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.distance import geodesic
from functools import partial
import pandas as pd
import numpy as np
import joblib
import json

location_centroids = pd.read_pickle("data/working_data/datalocation_centroids.pkl")

dataframe = pd.read_csv("data/working_data/idealista_data_total_20250805.csv")

# Leemos nuestro modelo
model = joblib.load("models/XGB_sales_model_20250802.pkl")

with open("data/mapping_dict_20250802.json",'r', encoding='utf-8') as f:
    mappings = json.load(f)

# Inicializa el geolocalizador globalmente
geolocator = Nominatim(user_agent="malaga-app-geocoder")
reverse = partial(geolocator.reverse, language="es", addressdetails=True)

with open("../data/size_bins_by_detailedType.json", 'r') as f:
    size_bins_dict = json.load(f)

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
    
def get_floor_options(property_type):
    if property_type in ['chalet', 'countryHouse']:
        return [{'label': '0', 'value': 0}]
    
    elif property_type in ['duplex', 'flat', 'office', 'premise', 'studio']:
        options = [{'label': str(i), 'value': i} for i in range(0, 31)]
        options.append({'label': 'Mezzanine', 'value': 0.5}) 
        return options
    

    elif property_type == 'garage':
        return [{'label': '-2', 'value': -2},
                {'label': '-1', 'value': -1},
                {'label': '0', 'value': 0}]
    
    else:
        return []
    
def estimate_price_by_area(df, operation, property_type, size, municipality, neighborhood):
    bins = size_bins_dict.get(property_type)
    if bins is None:
        return None  # o usar una media general

    labels = [f"{int(bins[i])}-{int(bins[i+1])}" if bins[i+1] != float('inf') else f"{int(bins[i])}+" for i in range(len(bins)-1)]
    size_range = pd.cut([size], bins=bins, labels=labels)[0]

    filtered_df = df[
        (df['operation'] == operation) &
        (df['propertyType'] == property_type) &
        (df['size_range'] == size_range) &
        (df['municipality'] == municipality) &
        (df['neighborhood'] == neighborhood)
    ]

    return filtered_df['avg_price_area_by_type_size_neigh'].mean()
    
def get_price_prediction(propertyType, operation, size, rooms, bathrroms, municipality, district, neighborhood, latitude, longitude, status, lift, floor, parkingSpace):
    """
    Hace la prediccion del precio usando nuestro modelo
    """

    new_Development = False
    isParkingSpaceIncludedInPrice = False

    if status == "newdevelopment":
        new_Development = True
    if parkingSpace == True:          
        isParkingSpaceIncludedInPrice = True

    priceByArea = estimate_price_by_area(dataframe, operation, propertyType, size, municipality, neighborhood)

    model_input_columns = [
        'propertyType', 'operation', 'size', 'rooms', 'bathrooms', 'municipality', 'district', 'neighborhood',
        'latitude', 'longitude', 'status', 'newDevelopment','priceByArea', 'floor', 'hasLift', 'hasParkingSpace', 'isParkingSpaceIncludedInPrice'] 

    data = {'propertyType': propertyType, 'operation': operation, 'size': size, 'rooms':rooms, 'bathrooms':bathrroms, 'municipality':municipality, 'district':district, 'neighborhood': neighborhood,
        'latitude':latitude, 'longitude':longitude, 'status': status, 'newDevelopment':new_Development,'priceByArea':priceByArea, 'floor':floor, 'hasLift':lift, 'hasParkingSpace':parkingSpace, 'isParkingSpaceIncludedInPrice':isParkingSpaceIncludedInPrice}
    
    data_processed = {}
    for col in model_input_columns:
        value = data[col]
        if col in mappings:
            # Aplicar mapeo
            data_processed[col] = mappings[col].get(value, -1)  # Usar -1 si no está en el mapeo
        else:
            # Dejar numéricos y booleanos como están
            data_processed[col] = value

    input_df = pd.DataFrame([data_processed])

    prediction = model.predict(input_df)[0]
    print(f"Predicción de precio: {prediction}")
    return prediction
    


