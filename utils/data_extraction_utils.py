import os
from dotenv import load_dotenv
import base64
import requests
import json
from datetime import datetime
import numpy as np

# Load variables from .env file
load_dotenv()

# Geting API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("SECRET")

# Fixed parameters for filtered searches
BASE_URL = "https://api.idealista.com/3.5/es" # Base URL to search in Spain
LOCATION = "0-EU-ES-29" # Location code for Málaga
MAX_ITEMS = 50 # Maximum number of items to retrieve in one request 50
SORT = 'desc'

# Data path
DATA_PATH = "data/results"

def get_oauth_token():
    """
    This function retrieves the OAuth token from the Idealista API.
    It uses the API_KEY and SECRET to authenticate and obtain the token.
    """
    # Encode the API_KEY and SECRET
    message = API_KEY + ":" + API_SECRET

    # Base64 encode the message
    auth = base64.b64encode(message.encode("ascii")).decode("ascii")

    # Set up headers for the request
    headers_dic = {'Authorization': 'Basic ' + auth,
                   'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
}

    # Set up parameters for the request
    params_dic = {'grant_type': 'client_credentials',
                  'scope': 'read'}

    # Make the request to get the token
    request_call = requests.post(
        "https://api.idealista.com/oauth/token",
        headers=headers_dic,
        data=params_dic
    )

    # Parse the response to get the token
    token = json.loads(request_call.text)['access_token']
    
    return token

def get_search_url(operation: str, property_type: str) -> str:
    """
    Build and return the base search URL for the Idealista API.

    params: 
        operation : str
            Type of operation to search for (e.g., "sale" or "rent").
        property_type : str
            Type of property to search for (e.g., "homes", "offices", "garages").

    returns:
        A formatted URL string containing the base search endpoint with fixed
        parameters and placeholders for pagination.
    """
    url = (
        f"{BASE_URL}/search?"
        f"operation={operation}&"
        f"maxItems={MAX_ITEMS}&"
        f"locationId={LOCATION}&"
        f"propertyType={property_type}&"
        f"sort={SORT}&"
        f"language=es&"
        f"numPage=%s"
    )
    return url

def get_data_from_api(url, pagination):
    """
    This function retrieves data from the Idealista API.
    It uses the provided URL to make a request to the API and get the data.
    """
    # Get the OAuth token
    token = get_oauth_token()

    # Set up headers for the request
    headers_dic = {'Content-Type': 'application/x-www-form-urlencoded',
                   'Authorization': 'Bearer ' + token,
                   }

    # Make the request to get the data
    request_call = requests.post(url, headers=headers_dic)

    # Parse the response
    try:
        data = request_call.json()
    except json.JSONDecodeError:
        print("Error: la respuesta no es JSON válida.")
        print("Texto recibido:", request_call.text)
        return None

    # File name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"../data/extracted_data/idealista-data-{timestamp}-{pagination}.json"

    # Save as  JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON guardado en {filename}")

    return data

def fetch_property_data(operation: str, property_type: str, max_pages: int):
    """
    Fetch all property data for a given operation and property type,
    respecting the pagination limits.
    
    Parameters
    ----------
    operation : str
        "sale" or "rent"
    property_type : str
        "homes", "offices", "premises", "garages"
    max_pages : int
        Maximum number of pages to fetch
    
    Returns
    -------
    list
        List of property dictionaries
    """
    results = []

    url_base = get_search_url(operation, property_type)
    
    # First page
    url1 = url_base % 1
    result = get_data_from_api(url1, 1)
    results += result['elementList']
    total_pages = result.get("totalPages", 1)
    requests_count = 1

    
    # Remaining pages
    for page in range(2, min(max_pages, total_pages) + 1):
        url = url_base % page
        result = get_data_from_api(url, page)
        results += result['elementList']
        requests_count += 1
    
    return results, requests_count


def get_complete_data_json(with_old_data=True):
    """
    This function make 100 requests to the Idealista API. The requests are divided like:
        - 30 requests for sale homes
        - 30 requests for rent homes
        - 5 requests for sale offices
        - 5 requests for rent offices
        - 8 requests for sale premises
        - 8 requests for rent premises
        - 7 requests for sale garages
        - 7 requests for rent garages

    Always keeps an accumulated historical file with ALL data ever fetched.

    parameters:
        with_old_data : bool
            If True, loads historical data and merges with the new results for processing.
            If False, processes only new data, but still appends them to the historical file.
    """
    
    historical_path = os.path.join(DATA_PATH, "merged_data.json")
    latest_path = os.path.join(DATA_PATH, "latest_run.json")
    
    # Charge historical data if it exists
    if os.path.exists(historical_path):
        with open(historical_path, "r", encoding="utf-8") as f:
            historical_data = json.load(f)
    else:
        historical_data = []

    # Choose base data to start
    if with_old_data:
        results = historical_data.copy()
    else:
        results = []

    requests_count = 0

    for operation, prop_type, max_pages in [
        ("sale", "homes", 30),
        ("rent", "homes", 30),
        ("sale", "offices", 5),
        ("rent", "offices", 5),
        ("sale", "premises", 7),
        ("rent", "premises", 8),
        ("sale", "garages", 6),
        ("rent", "garages", 7)
    ]:
        data, reqs = fetch_property_data(operation, prop_type, max_pages)
        results += data
        requests_count += reqs
        print(f"Fetched {len(data)} properties for {prop_type} {operation} in {reqs} requests.")
        

    remaining_requests = 100 - requests_count
    if remaining_requests > 0:
        print(f"Remaining requests to fill: {remaining_requests}")
        extra_per_type = remaining_requests // 2

        # Extra pages for homes sale
        url_base = get_search_url("sale", "homes")
        for page in range(31, 31 + extra_per_type):
            result = get_data_from_api(url_base % page, page)
            results += result['elementList']
        

        # Extra pages for homes rent
        url_base = get_search_url("rent", "homes")
        for page in range(31, 31 + extra_per_type):
            result = get_data_from_api(url_base % page, page)
            results += result['elementList']

    if with_old_data:
        with open(historical_path, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
            print(f"Historical data updated in {historical_path}")
    else:
        with open(latest_path, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
            print(f"Latest data properly saved in {latest_path}")

        with open(historical_path, 'w', encoding='utf-8') as outfile:
            json.dump(historical_data + results, outfile, indent=4, ensure_ascii=False)
            print(f"Historical data updated in {historical_path}")

    return results
     