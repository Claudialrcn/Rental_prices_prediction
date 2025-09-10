"""
────────────────────────────────────────────────────────────
📄 data_cleaning_utils.py
────────────────────────────────────────────────────────────
This file provides utility functions for preprocessing 
and cleaning real estate datasets. 

It includes:
    - The main preprocessing pipeline to clean and prepare 
      data for training and analysis.
    - Helper functions for handling missing values, filling 
      categorical fields, assigning size ranges, filtering 
      outliers, and enriching location data.
"""
#---------------------------- Imports --------------------------
import pandas as pd
import ast
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from functools import partial
from tqdm import tqdm
import time
import json
from datetime import datetime

#---------------------------- Constants --------------------------
MAPPING_FLOOR = {
    "bj": 0,
    "en": 0.5,  
    "st": -1,  
    "ss": -0.5    
}

DEFAULT_PARKING_SIZES = {
    "motorcycle": 3,
    "carAndMotorcycle": 19,
    "compactCar": 11,
    "sedanCar": 13,
    "garage": 15,
    "twoCars": 27
}

with open("../data/size_bins_by_detailedType.json", 'r') as f:
    SIZE_BINS_BY_DETAILED_TYPE = json.load(f)

#---------------------------- Misc Functions --------------------------
def fill_floor(row, modes):
    """
    Fill missing floor values based on property type.

    Parameters:
        row (pd.Series): A row of the DataFrame.
        modes (dict): Dictionary mapping property types to their most frequent floor values.

    Returns:
        int or float: The filled floor value.
    """
    if pd.isna(row["floor"]):
        if row["propertyType"] in ["chalet","countryHouse","office","premise"]:
            return 0
        elif row["propertyType"] == "garage":
            return -1
        else:
            return modes.get(row["propertyType"], 0)
    else:
        return row["floor"]
    
def fill_hasLift(row):
    """
    Determine whether a property should have a lift (elevator) based on its characteristics.

    Parameters:
        row (pd.Series): A row of the DataFrame.

    Returns:
        int: 1 if the property is assigned a lift, 0 otherwise.
    """
    # Rule 1: Chalets and country houses -> no lift
    if row["propertyType"] in ["chalet", "countryHouse"]:
        return 0
    # Rule 2: New developments -> has lift
    elif row.get("newDevelopment") == True:
        return 1
    # Rule 3: Older properties -> check floor
    elif pd.notna(row.get("floor")):
        if row["floor"] > 6:
            return 1
        else:
            return 0
    # Fallback if floor is missing
    else:
        return 0


def fill_garage_size(row):
    """
    Fill missing size values for garage-type properties with predefined defaults.
    
    Parameters:
        row (pd.Series): A row of the DataFrame.

    Returns:
        float or int: The filled or original size value.

    """
    # Check if the size is missing and the property is a garage type
    if pd.isna(row["size"]) and row["detailedType"] in DEFAULT_PARKING_SIZES:
        # Replace NaN with the default size for that detailedType
        return DEFAULT_PARKING_SIZES[row["detailedType"]]
    else:
        # Keep the original size if it's not missing or not a garage
        return row["size"]
    
def fill_missing_districts_neighborhoods_with_geopy(df):
    """
    Fills missing values in 'district' and 'neighborhood' columns
    using geopy reverse geocoding based on latitude and longitude.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'latitude', 'longitude', 
                           'district', and 'neighborhood' columns.

    Returns:
        df (pd.DataFrame): DataFrame with filled values.
    """
    geolocator = Nominatim(user_agent="malaga-geocoder")
    reverse = partial(geolocator.reverse, language="es")

    filled_district = 0
    filled_neighborhood = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if pd.isna(row["district"]) or pd.isna(row["neighborhood"]):
                location = reverse(f"{row['latitude']}, {row['longitude']}")
                address = location.raw.get("address", {})

                # Fill missing district
                if pd.isna(row["district"]):
                    district = (
                        address.get("borough") or
                        address.get("suburb") or
                        address.get("city_district") or
                        address.get("municipality") or
                        address.get("town") or
                        address.get("village")
                    )
                    if district:
                        df.at[index, "district"] = district
                        filled_district += 1

                # Fill missing neighborhood
                if pd.isna(row["neighborhood"]):
                    neighborhood = (
                        address.get("neighbourhood") or
                        address.get("quarter") or
                        address.get("hamlet") or
                        address.get("residential") or
                        address.get("locality")
                    )
                    if neighborhood:
                        df.at[index, "neighborhood"] = neighborhood
                        filled_neighborhood += 1

                time.sleep(0.5)  # Respect API usage limits

        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            print(f"Error at row {index}: {e}")
            continue

    print(f"Total districts filled: {filled_district}")
    print(f"Total neighborhoods filled: {filled_neighborhood}")

    return df

def filter_by_percentile(df, column, central_percent=0.75):
    """
    Filter a DataFrame to remove outliers based on a specified central percentile range.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to filter
        central_percent (float): Central portion of data to keep (between 0 and 1)
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Calculate lower and upper percentiles
    lower_quantile = (1 - central_percent) / 2
    upper_quantile = 1 - lower_quantile
    
    lower_limit = df[column].quantile(lower_quantile)
    upper_limit = df[column].quantile(upper_quantile)
    
    # Filter the DataFrame
    df_filtered = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    
    print(f"Filtering '{column}' to keep the central {central_percent*100:.1f}% of data")
    print(f"Lower limit: {lower_limit}, Upper limit: {upper_limit}")
    print(f"Number of records before filtering: {len(df)}")
    print(f"Number of records after filtering: {len(df_filtered)}\n")
    
    return df_filtered

def assign_size_range(row):
    """
    Assign a size range category to a property based on its detailed type and size.
        
    Parameters:
        row (pd.Series): A row of the DataFrame.

    Returns:
        str: The assigned size range label, or 'Unknown' if not applicable.
    """
    detailed_type = row['detailedType']
    size = row['size']
    bins = SIZE_BINS_BY_DETAILED_TYPE.get(detailed_type)

    if bins is None or pd.isna(size):
        return 'Unknown'

    labels = [
        f"{int(bins[i])}-{int(bins[i+1])}" if bins[i+1] != float('inf') 
        else f"{int(bins[i])}+"
        for i in range(len(bins)-1)
    ]

    try:
        return pd.cut([size], bins=bins, labels=labels, include_lowest=True)[0]
    except Exception:
        return 'Unknown'

#---------------------------- Main Functions --------------------------
def preprocess_train_data(data):
    """
    Preprocess the dataset to prepare it for training and testing ML models.

    Steps performed:
    1. Create DataFrame from raw data
    2. Handle missing values in `status` and `newDevelopment`
    3. Parse `parkingSpace` column
    4. Change `detailedType` field
    5. Clean and impute `floor` values
    6. Impute `hasLift` values
    7. Fill missing `rooms` and `bathrooms` (and drop outliers in bathrooms)
    8. Fill missing garage sizes with default values
    9. Drop duplicate rows
    10. Fill missing `district` and `neighborhood` using geopy
    11. Fill the rest of missing values of `neighborhood`
    12. Filter price outliers separately for rent and sale operations
    13. Assign `size_range` bins based on detailed type
    14. Compute average `priceByArea` grouped by several features
    15. Fill missing `priceByArea` values
    16. Check for remaining missing values and handle them
    """
    df = pd.DataFrame(data)
    columns = ['price', 'propertyType', 'detailedType', 'operation', 'size', 'rooms', 'bathrooms','municipality', 'district', 'neighborhood', 'latitude', 'longitude', 'status', 'newDevelopment', 'ParkingSpace', 'priceByArea', 'floor', 'hasLift']

    # Update rows where status is empty (NaN) and newDevelopment is True.
    df.loc[df["status"].isna() & (df["newDevelopment"] == True), "status"] = "newdevelopment"

    # Fill na values in 'status' with 'unknown'
    df['status'] = df['status'].fillna('unknown')

    # All values of parkingSpace as dict
    df["parkingSpace"] = df["parkingSpace"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    #Creating the new columns
    df["hasParkingSpace"] = df["parkingSpace"].apply(lambda x: x["hasParkingSpace"] if pd.notna(x) else None)

    # Dropping the original column
    df = df.drop(columns=["parkingSpace"])

    # Filling NaN values with False
    df["hasParkingSpace"] = df["hasParkingSpace"].fillna(False).astype(bool)

    # Rewriting detailedType with the content of subTypology if exists, else with typology
    df["detailedType"] = df["detailedType"].apply(
        lambda x: x.get("subTypology") if pd.notna(x) and isinstance(x, dict) and "subTypology" in x else (
            x.get("typology") if pd.notna(x) and isinstance(x, dict) and "typology" in x else None
        )
    )

    # Delete the rows with ati value in floor column
    df = df[df["floor"] != "ati"]

    # Convert floor values using the mapping dictionary
    df["floor"] = df["floor"].replace(MAPPING_FLOOR)

    # Calculate floor by propertyType
    modes = (df.groupby("propertyType")["floor"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            .to_dict())

    # Apply fill_floor function to complete de na values
    df["floor"] = df.apply(lambda r: fill_floor(r, modes), axis=1)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce")

    # Fill hasLift na values using the fill_hasLift function
    df["hasLift"] = df.apply(fill_hasLift, axis=1)

    # Fill na values in rooms and bathrooms with 0
    df["rooms"] = df["rooms"].fillna(0).astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(0).astype(int)
    # Keep only rows with 15 bathrooms or less
    df = df[df['bathrooms'] <= 15]

    # Fill missing size values for garages (all the na values are garages)
    df["size"] = df.apply(fill_garage_size, axis=1)


    # Fill missing districts and neighborhoods using geopy
    df = fill_missing_districts_neighborhoods_with_geopy(df)

    # Change municipality ti "Mijas" when neighborhood equals "Las Lagunas"
    df.loc[df['neighborhood'] == 'Las Lagunas', 'municipality'] = 'Mijas'
    # Replace "Torreblanca del Sol" with "Torreblanca" in the district column
    df["district"] = df["district"].replace(
        {"Torreblanca del Sol": "Torreblanca"}
    )

    # Fill missing or empty neighborhood values with the corresponding district value
    df["neighborhood"] = df["neighborhood"].where(
        df["neighborhood"].notna() & (df["neighborhood"] != ""),  # Keep if not NaN and not empty
        df["district"]  # Otherwise, replace with district
    )

    # Remove outliers by operation
    df_rental_filtered = filter_by_percentile(df[df['operation'] == 'rent'], 'price',0.8)
    df_sale_filtered = filter_by_percentile(df[df['operation'] == 'sale'], 'price',0.8)

    # Combine the filtered rental and sale dataframes
    df = pd.concat([df_rental_filtered, df_sale_filtered], ignore_index=True)

    # Assign size ranges
    df.loc[:,'size_range'] = df.apply(assign_size_range, axis=1)

    # Calculate the average price by operation, detailedType, size, status, municipality, district and neighborhood
    df['avg_price_area_by_type_size_neigh'] = df.groupby(
        ['operation', 'detailedType','status', 'municipality', 'district', 'neighborhood', 'size_range'], observed=True
    )['priceByArea'].transform('mean')

    #Fill nan values in priceByArea with the value in avg_price_area_by_type_size_neigh
    df['priceByArea'] = df['priceByArea'].fillna(df['avg_price_area_by_type_size_neigh'])

    df = df.drop(columns=['avg_price_area_by_type_size_neigh', 'size_range'])  # Drop the helper column

    df = df.drop_duplicates()

    # Null check
    missing_rows = df[df.isnull().any(axis=1)]
    n_missing = len(missing_rows)

    print(f"✅ Preprocessing completed. Rows with missing values: {n_missing}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    if n_missing == 0:
        file_path = f"../data/working_data/cleaned_data_complete_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print("✅ No missing values remain. Data saved to:", file_path)

    elif n_missing < 20:
        df = df.dropna()
        file_path = f"../data/working_data/cleaned_data_droppedna_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print(f"⚠️ Dropped {n_missing} rows with missing values. Data saved to:", file_path)
    else:
        file_path = f"../data/working_data/incomplete_cleaned_data_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print(f"❌ Warning: {n_missing} rows with missing values remain. Data saved to:", file_path)






